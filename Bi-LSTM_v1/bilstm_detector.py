import cv2
import mediapipe as mp
import numpy as np
import pickle
import tkinter as tk
from tkinter import ttk 
from PIL import Image, ImageTk, ImageFont, ImageDraw
import time
import os
import pyautogui 
import sys
from tensorflow.keras.models import load_model

# --- CONFIGURATION AND ARABIC TRANSLATION MAP ---

ARABIC_CHAR_MAP = {
    'ain': 'ع', 'al': 'ال', 'aleff': 'ا', 'bb': 'ب', 'dal': 'د', 'dha': 'ظ', 'dhad': 'ض', 'fa': 'ف', 
    'gaaf': 'ق', 'ghain': 'غ', 'ha': 'ه', 'haa': 'ح', 'jeem': 'ج', 'kaaf': 'ك', 'khaa': 'خ', 'la': 'لا', 
    'laam': 'ل', 'meem': 'م', 'nun': 'ن', 'ra': 'ر', 'saad': 'ص', 'seen': 'س', 'sheen': 'ش', 'taa': 'ت', 
    'taad': 'ط', 'tha': 'ث', 'thh': 'ث', 'tah': 'ط', 'ya': 'ي', 'yaa': 'ي', 'zay': 'ز', 'zal': 'ذ', 
    'UNKNOWN': '?', 
}

# Load font file 
try:
    font_path = "a.ttf"
    if not os.path.exists(font_path):
        font_path = "arial.ttf" 
    
    font_size_char = 75
    font_size_ui = 18

    # Font for predicted character (large)
    font = ImageFont.truetype(font_path, font_size_char, encoding="utf-8") 
    # Font for UI elements / Language Label (smaller)
    font_size_label = 30
    label_font = ImageFont.truetype(font_path, font_size_label, encoding="utf-8")
    # Font for confidence text (small)
    font_small = ImageFont.truetype(font_path, 20, encoding="utf-8")
    
    ui_font_tuple = ('Arial', font_size_ui, 'bold')
    
except IOError:
    print(f"Error: Font files ({font_path} or arial.ttf) not found. Default fonts will be used.")
    font = ImageFont.load_default()
    label_font = ImageFont.load_default()
    font_small = ImageFont.load_default()
    ui_font_tuple = ('TkDefaultFont', 14)
    
LOG_FILE = 'prediction_log_bilstm.txt'
SCREENSHOT_DIR = 'screenshots'
# Model dosyalarının bulunduğu klasör
MODELS_DIR = r'C:\Users\ridva\.conda\envs\vmamba_env\vmamba_proje\Bi-LSTM_v1' 

language_configs = {
    'ASL': {'model_file': 'bilstm_EN.h5', 'label_map_file': 'label_map_EN.pkl', 'feature_size': 42, 'hands_required': 1},
    'ArSL': {'model_file': 'bilstm_AR.h5', 'label_map_file': 'label_map_AR.pkl', 'feature_size': 42, 'hands_required': 1},
    'TSL': {'model_file': 'bilstm_TR.h5', 'label_map_file': 'label_map_TR.pkl', 'feature_size': 84, 'hands_required': 2}, 
}

current_language = 'TSL'
model = None
labels_dict = None

if not os.path.exists(SCREENSHOT_DIR):
    os.makedirs(SCREENSHOT_DIR)
    
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=2)

hand_present = False
current_latin_label = "" 
current_character = "" 
hand_start_time = None
last_predicted_char = "" 

PREDICTION_CONFIDENCE_THRESHOLD = 0.7

# --- CORE FUNCTIONS ---

def take_screenshot():
    """Takes and saves a screenshot."""
    try:
        screenshot = pyautogui.screenshot() 
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join(SCREENSHOT_DIR, filename)
        screenshot.save(filepath)
        print(f"Screenshot saved: {filepath}")
    except Exception as e:
        print(f"Error taking screenshot: {e}")

def load_resources(lang_code):
    """Loads the Bi-LSTM model and label map for the specified language."""
    global model, labels_dict, current_language

    config = language_configs.get(lang_code)
    if not config:
        print(f"Error: Configuration for {lang_code} not found.")
        return False

    model_path = os.path.join(MODELS_DIR, config['model_file'])
    label_map_path = os.path.join(MODELS_DIR, config['label_map_file'])

    if not os.path.exists(model_path) or not os.path.exists(label_map_path):
        print(f"Error: {lang_code} model/label file not found.")
        print(f"Checked Paths: Model: {model_path} | Label: {label_map_path}")
        return False

    try:
        # Bi-LSTM modelini yükle
        model = load_model(model_path)
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")

        with open(label_map_path, 'rb') as f:
            labels_dict = pickle.load(f)
            
        current_language = lang_code
        print(f"[{lang_code}] Bi-LSTM model and labels loaded successfully.")
        print(f"Number of classes: {len(labels_dict)}")
        print(f"Hands required: {config['hands_required']}")
        return True

    except Exception as e:
        print(f"Error loading {lang_code} model: {e}")
        model = None
        labels_dict = None
        return False

def log_prediction_time(duration_ms, prediction):
    """Logs the prediction time and result to a text file."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')} | Language: {current_language} | Prediction: {prediction} | Duration: {duration_ms:.2f} ms\n")
        
def get_display_character(latin_label, lang):
    """Converts the Latin label from the model to the correct display character based on the language."""
    if lang == 'ArSL':
        return ARABIC_CHAR_MAP.get(latin_label.lower(), latin_label)
    return latin_label
    
def get_display_language_label(lang_code):
    """Returns the display text for the active language."""
    if lang_code == 'ArSL':
        return 'العربية' 
    elif lang_code == 'TSL':
        return 'TSL'
    elif lang_code == 'ASL':
        return 'ASL'
    return lang_code

def process_frame(frame):
    """Processes the video frame, extracts hand features, performs prediction, and updates the image."""
    global hand_present, current_latin_label, current_character, hand_start_time, last_predicted_char

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    predicted_latin_label = "" 
    predicted_display_char = "" 
    all_x_coords = []
    all_y_coords = []
    prediction_confidence = 0.0
    
    language_display_text = get_display_language_label(current_language)
    hands_required = language_configs[current_language]['hands_required']
    
    # Initialize PIL image for drawing complex text
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # --- LANGUAGE LABEL RENDERING (Top Left) ---
    
    if current_language == 'ArSL':
        text_position = (10, 10) 
        text_color = (255, 0, 0)
        
        try:
            draw.text(text_position, language_display_text, font=label_font, fill=text_color, direction="rtl")
        except Exception as e:
            cv2.putText(frame, language_display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
    else:
        cv2.putText(frame, 
                    language_display_text, 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8,      
                    (0, 0, 255),
                    2,        
                    cv2.LINE_AA) 
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
    
    # --- HAND DETECTION AND PREDICTION LOGIC ---
    if results.multi_hand_landmarks and model and labels_dict:
        detected_hands = len(results.multi_hand_landmarks)
        
        # Gerekli el sayısı kontrolü
        if detected_hands != hands_required:
            if detected_hands == 1:
                cv2.putText(frame, f"Need {hands_required} hand(s)! Detected: {detected_hands}", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            hand_present = False
            predicted_display_char = f"Need {hands_required} hand(s)"
            char_var.set(predicted_display_char)
            return frame
        
        hand_present = True
        data_aux = [] 
        
        # Tüm elleri işle
        for hand_landmarks in results.multi_hand_landmarks:
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                      mp_drawing_styles.get_default_hand_landmarks_style(), 
                                      mp_drawing_styles.get_default_hand_connections_style())
            
            # Coordinate collection and normalization
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            all_x_coords.extend(x_coords)
            all_y_coords.extend(y_coords)
            
            # Her el için ayrı normalizasyon
            min_x, min_y = min(x_coords), min(y_coords)
            
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min_x)
                data_aux.append(hand_landmarks.landmark[i].y - min_y)

        # Feature vector padding/truncation
        expected_size = language_configs[current_language]['feature_size']
        final_data_vector = data_aux[:expected_size]
        if len(final_data_vector) < expected_size: 
            final_data_vector.extend([0.0] * (expected_size - len(final_data_vector))) 

        if len(final_data_vector) == expected_size:
            try:
                # Model input shape'e göre veriyi reshape et
                # Model (None, 42, 1) bekliyor, biz (1, 42, 1) göndermeliyiz
                input_data = np.array(final_data_vector).reshape(1, expected_size, 1)
                
                # Prediction
                predictions = model.predict(input_data, verbose=0)
                predicted_index = np.argmax(predictions[0])
                prediction_confidence = np.max(predictions[0])
                
                predicted_latin_label = labels_dict.get(int(predicted_index), 'UNKNOWN')
                predicted_display_char = get_display_character(predicted_latin_label, current_language)
                current_latin_label = predicted_latin_label
                current_character = predicted_display_char
                
                print(f"Prediction: {predicted_latin_label}, Confidence: {prediction_confidence:.3f}, Hands: {detected_hands}")
                
                # Bounding Box - tüm elleri kapsayacak şekilde
                if all_x_coords and all_y_coords: 
                    x1 = max(0, int(min(all_x_coords) * W) - 20)
                    y1 = max(0, int(min(all_y_coords) * H) - 20)
                    x2 = min(W, int(max(all_x_coords) * W) + 20)
                    y2 = min(H, int(max(all_y_coords) * H) + 20)
                    
                    # Color based on confidence score
                    if prediction_confidence >= PREDICTION_CONFIDENCE_THRESHOLD:
                        box_color = (0, 255, 0)  # Green - high confidence
                    else:
                        box_color = (0, 165, 255)  # Orange - low confidence
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)
                    
                    # Draw character (with PIL)
                    pil_image_for_char = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw_char = ImageDraw.Draw(pil_image_for_char)
                    
                    text_position = (x1, max(0, y1 - 80)) 
                    direction_mode = "rtl" if current_language == 'ArSL' else "ltr"
                    
                    try:
                        # Draw character
                        text_bbox = draw_char.textbbox(text_position, predicted_display_char, font=font, direction=direction_mode)
                        draw_char.rectangle(text_bbox, fill=(255, 255, 255))
                        draw_char.text(text_position, predicted_display_char, font=font, fill=(0, 0, 0), direction=direction_mode)
                        
                        # Draw confidence score
                        confidence_text = f"Conf: {prediction_confidence:.2f}"
                        confidence_bbox = draw_char.textbbox((x1, y1 - 30), confidence_text, font=font_small)
                        draw_char.rectangle(confidence_bbox, fill=(255, 255, 255))
                        draw_char.text((x1, y1 - 30), confidence_text, font=font_small, fill=(0, 0, 0))
                        
                        # Draw hands info
                        hands_text = f"Hands: {detected_hands}"
                        hands_bbox = draw_char.textbbox((x1, y1 - 50), hands_text, font=font_small)
                        draw_char.rectangle(hands_bbox, fill=(255, 255, 255))
                        draw_char.text((x1, y1 - 50), hands_text, font=font_small, fill=(0, 0, 0))
                        
                        frame = cv2.cvtColor(np.array(pil_image_for_char), cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        print(f"Text drawing error: {e}")
                        cv2.putText(frame, predicted_display_char, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"{prediction_confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
            except Exception as e:
                print(f"Prediction Error: {e}")
                predicted_display_char = "ERROR"
                
    else:
        hand_present = False
        hand_start_time = None
        predicted_display_char = ""
        current_latin_label = ""
        current_character = ""

    # Word formation logic (3 second rule)
    if (hand_present and current_character not in ["", "?", "UNKNOWN", "ERROR"] and 
        prediction_confidence >= PREDICTION_CONFIDENCE_THRESHOLD):
        
        if hand_start_time is None or last_predicted_char != current_character:
            hand_start_time = time.time()
            last_predicted_char = current_character
            
        current_time = time.time()
        
        if hand_start_time is not None and current_time - hand_start_time >= 3:
            sentence_var.set(sentence_var.get() + current_character)
            last_predicted_char = "" 
            hand_start_time = None
    else:
        hand_start_time = None
        last_predicted_char = ""
        
    char_var.set(predicted_display_char)
    
    return frame

# --- TKINTER UI FUNCTIONS ---

def update_video_feed():
    """Main loop updating the camera feed."""
    ret, frame = cap.read()
    if not ret:
        video_label.after(10, update_video_feed)
        return

    frame = cv2.resize(frame, (800, 600))
    frame = cv2.flip(frame, 1)

    processed_frame = process_frame(frame)
    
    # Convert from OpenCV to PIL and Tkinter
    photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)))
    video_label.config(image=photo)
    video_label.photo = photo
    video_label.after(10, update_video_feed)

def change_language_and_load(lang):
    """Changes language and loads the new model."""
    if load_resources(lang):
        justify = 'right' if lang == 'ArSL' else 'left'
        char_entry.config(justify=justify)
        sentence_entry.config(justify=justify)
        update_button_style(lang)
        clear_characters()

def update_button_style(active_lang):
    """Updates the style of the active language button."""
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
    """Clears text boxes and resets the timer."""
    sentence_var.set("")
    char_var.set("")
    global current_latin_label, current_character, hand_start_time, last_predicted_char
    current_latin_label = ""
    current_character = ""
    hand_start_time = None
    last_predicted_char = ""

def delete_last_character():
    """Deletes the last character."""
    current_sentence = sentence_var.get()
    if current_sentence:
        sentence_var.set(current_sentence[:-1])
    clear_characters_timer_only() 

def clear_characters_timer_only():
    """Clears prediction/timer state without clearing the sentence."""
    char_var.set("")
    global current_latin_label, current_character, hand_start_time, last_predicted_char
    current_latin_label = ""
    current_character = ""
    hand_start_time = None
    last_predicted_char = ""

def check_screenshot_key(event):
    """Takes a screenshot when 's' or 'S' is pressed."""
    if event.char.lower() == 's':
        take_screenshot()

# --- TKINTER UI SETUP ---

root = tk.Tk()
root.title("👋 Sign Language Recognition - Bi-LSTM Pro")
root.geometry("1180x750") 
root.resizable(False, False)

# Style Definition
style = ttk.Style()
style.theme_use('clam')
style.configure('TFrame', background='#f0f0f0')
style.configure('TLabel', background='#f0f0f0', font=('Arial', 14))
style.configure('TButton', font=('Arial', 14, 'bold'), padding=10, relief='flat', background='#e1e1e1', foreground='#333333')
style.map('TButton', background=[('active', '#cccccc')], foreground=[('disabled', '#a0a0a0')])
style.configure('Active.TButton', background='#28a745', foreground='white')
style.map('Active.TButton', background=[('active', '#1e7e34')])

# Main Frames
main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill='both', expand=True)

# Top Control Frame
top_control_frame = ttk.Frame(main_frame)
top_control_frame.pack(fill='x', pady=(0, 10))
char_var = tk.StringVar()
sentence_var = tk.StringVar()
char_label = ttk.Label(top_control_frame, text="Predicted Character:", font=('Arial', 16, 'bold'))
char_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')
char_entry = tk.Entry(top_control_frame, textvariable=char_var, font=ui_font_tuple, state='readonly', width=5, justify='left', bd=3, relief='sunken')
char_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')
sentence_label = ttk.Label(top_control_frame, text="Word:", font=('Arial', 16, 'bold'))
sentence_label.grid(row=0, column=2, padx=(50, 10), pady=10, sticky='w')
sentence_entry = tk.Entry(top_control_frame, textvariable=sentence_var, font=ui_font_tuple, state='readonly', width=30, justify='left', bd=3, relief='sunken')
sentence_entry.grid(row=0, column=3, padx=10, pady=10, sticky='ew')
top_control_frame.grid_columnconfigure(3, weight=1)

# Main Content Frame
main_content_frame = ttk.Frame(main_frame)
main_content_frame.pack(fill='both', expand=True)

# Left Panel (Video Feed)
video_frame = ttk.Frame(main_content_frame, relief='solid', borderwidth=2, padding=5)
video_frame.pack(side='left', padx=10, pady=10)
video_label = tk.Label(video_frame, text="Camera Feed is Starting...", background='black', foreground='white', width=800, height=600)
video_label.pack()

# Right Panel (Language Buttons and Operations)
right_panel = ttk.Frame(main_content_frame, width=320, style='TFrame') 
right_panel.pack(side='right', fill='y', padx=10, pady=10)
right_panel.grid_propagate(False)
ttk.Label(right_panel, text="Language Selection", font=('Arial', 16, 'bold')).grid(row=0, column=0, columnspan=2, pady=(10, 20), sticky='ew')

# Language Buttons
ar_button = ttk.Button(right_panel, text="AR ArSL (العربية)", command=lambda: change_language_and_load('ArSL'), width=22) 
ar_button.grid(row=1, column=0, columnspan=2, pady=10, padx=5, sticky='ew')
tr_button = ttk.Button(right_panel, text="TR TSL (Turkish)", command=lambda: change_language_and_load('TSL'), width=22) 
tr_button.grid(row=2, column=0, columnspan=2, pady=10, padx=5, sticky='ew')
en_button = ttk.Button(right_panel, text="US ASL (American)", command=lambda: change_language_and_load('ASL'), width=22) 
en_button.grid(row=3, column=0, columnspan=2, pady=10, padx=5, sticky='ew')

ttk.Label(right_panel, text="").grid(row=4, column=0, columnspan=2, pady=20)
right_panel.grid_columnconfigure(0, weight=1)
right_panel.grid_columnconfigure(1, weight=1)

# Operation Buttons
delete_button = ttk.Button(right_panel, text="← Delete", command=delete_last_character, width=12) 
delete_button.grid(row=5, column=0, padx=5, pady=10, sticky='ew')
clear_button = ttk.Button(right_panel, text="Clear", command=clear_characters, width=12) 
clear_button.grid(row=5, column=1, padx=5, pady=10, sticky='ew')

root.bind('<Key>', check_screenshot_key)
root.focus_set()

# --- MAIN STARTUP ---

if __name__ == '__main__':
    if os.path.exists(LOG_FILE):
        try:
            os.remove(LOG_FILE)
            print(f"{LOG_FILE} is cleaned.")
        except:
            print(f"Warning: {LOG_FILE} could not be cleaned. Continuing.")
    
    # Load resources before starting camera
    if load_resources(current_language):
        update_button_style(current_language)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
             print("Error: Camera is not starting. Check camera connection or permissions.")
             sys.exit()

        update_video_feed()
        root.mainloop()
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Startup Error. Please ensure model files are loaded correctly and path is correct.")
        sys.exit()