import os
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageFont, ImageDraw
from collections import deque
import time
import torch
import torch.nn as nn
import timm
import json
import pyautogui
import sys

# =========================================================================
# CONFIGURATION AND CONSTANTS
# =========================================================================

MODEL_DIR = './vmamba_models'
MODEL_EPOCH = 3
INITIAL_LANGUAGE = 'ASL'

# PREDICTION STABILIZATION PARAMETERS
PREDICTION_HISTORY_SIZE = 20
STABILITY_THRESHOLD = 15
CONFIDENCE_THRESHOLD = 0.90
SENTENCE_ADD_DELAY = 1.0

SCREENSHOT_DIR = 'screenshots'
LOG_FILE = 'prediction_log_vmamba.txt'

# Arabic character mapping
ARABIC_CHAR_MAP = {
    'ain': 'ع', 'al': 'ال', 'aleff': 'ا', 'bb': 'ب', 'dal': 'د', 'dha': 'ظ', 'dhad': 'ض', 'fa': 'ف', 
    'gaaf': 'ق', 'ghain': 'غ', 'ha': 'ه', 'haa': 'ح', 'jeem': 'ج', 'kaaf': 'ك', 'khaa': 'خ', 'la': 'لا', 
    'laam': 'ل', 'meem': 'م', 'nun': 'ن', 'ra': 'ر', 'saad': 'ص', 'seen': 'س', 'sheen': 'ش', 'taa': 'ت', 
    'taad': 'ط', 'tha': 'ث', 'thh': 'ث', 'tah': 'ط', 'ya': 'ي', 'yaa': 'ي', 'zay': 'ز', 'zal': 'ذ', 
    'UNKNOWN': '?', 
}

# Language configurations
language_configs = {
    'ASL': {'code': 'EN', 'display': 'ASL'},
    'ArSL': {'code': 'AR', 'display': 'العربية'},
    'TSL': {'code': 'TR', 'display': 'TSL'},
}

# Load fonts
try:
    font_path = "arial.ttf"
    font_size_char = 75
    font_size_label = 30
    font_size_ui = 18

    font = ImageFont.truetype(font_path, font_size_char, encoding="utf-8")
    label_font = ImageFont.truetype(font_path, font_size_label, encoding="utf-8")
    ui_font_tuple = ('Arial', font_size_ui, 'bold')
    
except IOError:
    print("Warning: Font files not found. Using default fonts.")
    font = ImageFont.load_default()
    label_font = ImageFont.load_default()
    ui_font_tuple = ('TkDefaultFont', 14)

if not os.path.exists(SCREENSHOT_DIR):
    os.makedirs(SCREENSHOT_DIR)

# =========================================================================
# VMamba Model
# =========================================================================
class CustomMambaClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(CustomMambaClassifier, self).__init__()
        self.backbone = timm.create_model('convnext_base', pretrained=pretrained, features_only=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.backbone.feature_info[-1]['num_chs'], num_classes)
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.avgpool(x)
        x = self.head(x)
        return x

# Global variables
model = None
class_map = {}
class_names = []
current_language = 'ASL'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prediction_history = deque(maxlen=PREDICTION_HISTORY_SIZE)
last_added_character = ""
last_add_time = time.time()

# MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=2)

def take_screenshot():
    try:
        screenshot = pyautogui.screenshot() 
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join(SCREENSHOT_DIR, filename)
        screenshot.save(filepath)
        print(f"Screenshot saved: {filepath}")
    except Exception as e:
        print(f"Error taking screenshot: {e}")

def load_resources(lang):
    global model, class_map, class_names, current_language, prediction_history
    
    lang_code = language_configs[lang]['code']
    model_path = os.path.join(MODEL_DIR, f'vmamba_{lang_code}_epoch_{MODEL_EPOCH}.pth')
    class_map_path = os.path.join(MODEL_DIR, f'class_map_{lang_code}.json')
    
    print(f"Loading [{lang}]... Model: {model_path}")

    # Load Class Map
    if not os.path.exists(class_map_path):
        messagebox.showerror("ERROR", f"Class map file '{class_map_path}' not found.")
        return False
    
    try:
        with open(class_map_path, 'r', encoding='utf-8') as f:
            class_map = json.load(f)
            
        num_classes = len(class_map)
        class_names = sorted(class_map, key=class_map.get)
        print(f"[{lang}] Number of classes: {num_classes}")
    except Exception as e:
        messagebox.showerror("ERROR", f"Error loading class map: {e}")
        return False

    # Load Model
    if not os.path.exists(model_path):
        messagebox.showerror("ERROR", f"Model file '{model_path}' not found.")
        model = None
        return False
    
    try:
        new_model = CustomMambaClassifier(num_classes=num_classes, pretrained=False).to(device)
        new_model.load_state_dict(torch.load(model_path, map_location=device))
        new_model.eval()
        
        model = new_model
        current_language = lang
        prediction_history.clear()
        print(f"VMamba model loaded successfully. (Language: {lang})")
        return True
        
    except Exception as e:
        messagebox.showerror("ERROR", f"Error loading model: {e}")
        model = None
        return False

def get_display_character(latin_label, lang):
    if lang == 'ArSL':
        return ARABIC_CHAR_MAP.get(latin_label.lower(), latin_label)
    return latin_label

def get_display_language_label(lang_code):
    return language_configs[lang_code]['display']

def get_stable_prediction(current_pred_idx):
    global prediction_history
    
    if current_pred_idx is not None:
        prediction_history.append(current_pred_idx)
    
    if not prediction_history:
        return None, 0 
    
    unique, counts = np.unique(list(prediction_history), return_counts=True)
    stable_char_idx = unique[np.argmax(counts)]
    stable_count = counts[np.argmax(counts)]
    
    if 0 <= stable_char_idx < len(class_names):
        stable_char = class_names[stable_char_idx]
    else:
        stable_char = None
        
    if stable_count >= STABILITY_THRESHOLD:
        return stable_char, stable_count
    else:
        return stable_char, stable_count

def process_frame(frame):
    global last_added_character, last_add_time, model
    
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    predicted_character = None
    max_confidence = 0.0
    stable_char = None 
    stable_count = 0
    current_pred_idx = None
    
    language_display_text = get_display_language_label(current_language)
    
    # Draw language label
    if current_language == 'ArSL':
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        text_position = (10, 10) 
        text_color = (255, 0, 0)
        
        try:
            draw.text(text_position, language_display_text, font=label_font, fill=text_color, direction="rtl")
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            cv2.putText(frame, language_display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, language_display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Hand detection and prediction
    if results.multi_hand_landmarks and model is not None:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Calculate bounding box
        x_list = [lm.x for lm in hand_landmarks.landmark]
        y_list = [lm.y for lm in hand_landmarks.landmark]
        
        padding = 50
        x1 = int(min(x_list) * W)
        y1 = int(min(y_list) * H)
        x2 = int(max(x_list) * W)
        y2 = int(max(y_list) * H)
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        max_dim = max(x2-x1, y2-y1)
        square_side = max_dim + 2 * padding

        crop_x1 = max(0, center_x - square_side // 2)
        crop_y1 = max(0, center_y - square_side // 2)
        crop_x2 = min(W, center_x + square_side // 2)
        crop_y2 = min(H, center_y + square_side // 2)
        
        hand_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if hand_img.size > 0:
            h, w, _ = hand_img.shape
            max_size = max(h, w)
            
            square_img = np.zeros((max_size, max_size, 3), dtype=np.uint8)
            start_y = (max_size - h) // 2
            start_x = (max_size - w) // 2
            square_img[start_y:start_y + h, start_x:start_x + w] = hand_img
            
            resized_hand = cv2.resize(square_img, (224, 224))
            
            # Preprocess image
            image_tensor = torch.from_numpy(resized_hand).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            normalized_hand = (image_tensor - mean) / std
            normalized_hand = normalized_hand.to(device)

            # Model prediction
            with torch.no_grad():
                outputs = model(normalized_hand.unsqueeze(0))
                probabilities = model.softmax(outputs)
            
            max_confidence, predicted_index = torch.max(probabilities, 1)
            max_confidence = max_confidence.item()
            predicted_index = predicted_index.item()
            
            current_character = class_names[predicted_index]
            current_pred_idx = predicted_index

            # Stabilization check
            stable_char, stable_count = get_stable_prediction(current_pred_idx)
            
            # Sentence addition logic
            current_time = time.time()
            is_stable = stable_count >= STABILITY_THRESHOLD
            
            if is_stable and max_confidence >= CONFIDENCE_THRESHOLD:
                display_char = get_display_character(stable_char, current_language)
                if display_char != last_added_character:
                    if current_time - last_add_time >= SENTENCE_ADD_DELAY:
                        sentence_var.set(sentence_var.get() + display_char)
                        last_added_character = display_char
                        last_add_time = current_time
                else:
                    last_add_time = current_time

            # Draw bounding box and character
            color = (0, 255, 0) if is_stable and max_confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), color, 4)
            
            display_char = get_display_character(stable_char, current_language) if stable_char and is_stable and stable_count >= STABILITY_THRESHOLD else (get_display_character(current_character, current_language) if 'current_character' in locals() else "...")
            
            # Use PIL for text rendering
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            text_position = (crop_x1, max(0, crop_y1 - 80))
            direction_mode = "rtl" if current_language == 'ArSL' else "ltr"
            
            try:
                text_bbox = draw.textbbox(text_position, display_char, font=font, direction=direction_mode)
                draw.rectangle(text_bbox, fill=(255, 255, 255))
                draw.text(text_position, display_char, font=font, fill=(0, 0, 0), direction=direction_mode)
                frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except:
                cv2.putText(frame, display_char, (crop_x1, max(0, crop_y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                
    else:
        prediction_history.clear()
        stable_char = "Waiting for hand..."
        
    # Update character display
    if stable_char and stable_count >= STABILITY_THRESHOLD:
        display_char = get_display_character(stable_char, current_language)
    elif 'current_character' in locals():
        display_char = get_display_character(current_character, current_language)
    else:
        display_char = "Waiting..."
        
    char_var.set(display_char)
    
    return frame

# =========================================================================
# TKINTER UI - TAMAMEN DÜZELTİLMİŞ KISIM
# =========================================================================

def update_video_feed():
    ret, frame = cap.read()
    if not ret:
        video_label.after(10, update_video_feed)
        return

    frame = cv2.resize(frame, (800, 600))
    frame = cv2.flip(frame, 1)

    processed_frame = process_frame(frame)
    
    photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)))
    video_label.config(image=photo)
    video_label.photo = photo
    video_label.after(10, update_video_feed)

def change_language_and_load(lang):
    if load_resources(lang):
        justify = 'right' if lang == 'ArSL' else 'left'
        char_entry.config(justify=justify)
        sentence_entry.config(justify=justify)
        update_button_style(lang)
        clear_characters()

def update_button_style(active_lang):
    buttons = {
        'ArSL': ar_button,
        'TSL': tr_button,
        'ASL': en_button,
    }
    for lang, button in buttons.items():
        if lang == active_lang:
            button.config(style='Active.TButton')
        else:
            button.config(style='TButton')

def clear_characters():
    sentence_var.set("")
    char_var.set("")
    global last_added_character, last_add_time
    last_added_character = ""
    last_add_time = time.time()
    prediction_history.clear()

def delete_last_character():
    current_sentence = sentence_var.get()
    if current_sentence:
        sentence_var.set(current_sentence[:-1])
    clear_characters()

def check_screenshot_key(event):
    if event.char.lower() == 's':
        take_screenshot()

# Create main window
root = tk.Tk()
root.title("👋 Sign Language Recognition - VMamba Pro")
root.geometry("1180x750") 
root.resizable(False, False)

# Style configuration
style = ttk.Style()
style.theme_use('clam')

# Configure styles
style.configure('TFrame', background='#f0f0f0')
style.configure('TLabel', background='#f0f0f0', font=('Arial', 14))
style.configure('TButton', font=('Arial', 14, 'bold'), padding=10)
style.configure('Active.TButton', background='#28a745', foreground='white')

# Create main container with padding - BURASI DÜZELTİLDİ
main_container = ttk.Frame(root, padding="10")
main_container.pack(fill='both', expand=True)

# Top control frame
control_frame = ttk.Frame(main_container)
control_frame.pack(fill='x', pady=(0, 10))

char_var = tk.StringVar()
sentence_var = tk.StringVar()

char_label = ttk.Label(control_frame, text="Predicted Character:", font=('Arial', 16, 'bold'))
char_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')

char_entry = tk.Entry(control_frame, textvariable=char_var, font=ui_font_tuple, state='readonly', width=5, justify='left')
char_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')

sentence_label = ttk.Label(control_frame, text="Word:", font=('Arial', 16, 'bold'))
sentence_label.grid(row=0, column=2, padx=(50, 10), pady=10, sticky='w')

sentence_entry = tk.Entry(control_frame, textvariable=sentence_var, font=ui_font_tuple, state='readonly', width=30, justify='left')
sentence_entry.grid(row=0, column=3, padx=10, pady=10, sticky='ew')

control_frame.grid_columnconfigure(3, weight=1)

# Main content area
content_frame = ttk.Frame(main_container)
content_frame.pack(fill='both', expand=True)

# Video frame - BURASI DÜZELTİLDİ (camera_frame YERİNE video_frame)
video_frame = ttk.Frame(content_frame, relief='solid', borderwidth=2)
video_frame.pack(side='left', padx=10, pady=10)

video_label = tk.Label(video_frame, text="Camera Feed is Starting...", background='black', foreground='white', width=800, height=600)
video_label.pack()

# Right panel
right_panel = ttk.Frame(content_frame, width=320)
right_panel.pack(side='right', fill='y', padx=10, pady=10)
right_panel.pack_propagate(False)

ttk.Label(right_panel, text="Language Selection", font=('Arial', 16, 'bold')).pack(pady=(10, 20))

# Language buttons
ar_button = ttk.Button(right_panel, text="AR ArSL (العربية)", command=lambda: change_language_and_load('ArSL'), width=22)
ar_button.pack(pady=10)

tr_button = ttk.Button(right_panel, text="TR TSL (Turkish)", command=lambda: change_language_and_load('TSL'), width=22)
tr_button.pack(pady=10)

en_button = ttk.Button(right_panel, text="US ASL (American)", command=lambda: change_language_and_load('ASL'), width=22)
en_button.pack(pady=10)

# Operation buttons
operation_frame = ttk.Frame(right_panel)
operation_frame.pack(pady=20)

delete_button = ttk.Button(operation_frame, text="← Delete", command=delete_last_character, width=12)
delete_button.grid(row=0, column=0, padx=5, pady=10)

clear_button = ttk.Button(operation_frame, text="Clear", command=clear_characters, width=12)
clear_button.grid(row=0, column=1, padx=5, pady=10)

root.bind('<Key>', check_screenshot_key)
root.focus_set()

# =========================================================================
# START APPLICATION
# =========================================================================
if __name__ == '__main__':
    if os.path.exists(LOG_FILE):
        try:
            os.remove(LOG_FILE)
            print(f"{LOG_FILE} cleaned.")
        except:
            print(f"Warning: {LOG_FILE} could not be cleaned.")
    
    if load_resources(current_language):
        update_button_style(current_language)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Camera not starting. Check connection.")
            sys.exit()

        update_video_feed()
        root.mainloop()
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Startup Error. Check model files and path.")
        sys.exit()