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
import sys # Gerekli kütüphaneleri dahil et

# --- CONFIGURATION AND ARABIC TRANSLATION MAP ---

ARABIC_CHAR_MAP = {
    'ain': 'ع', 'al': 'ال', 'aleff': 'ا', 'bb': 'ب', 'dal': 'د', 'dha': 'ظ', 'dhad': 'ض', 'fa': 'ف', 
    'gaaf': 'ق', 'ghain': 'غ', 'ha': 'ه', 'haa': 'ح', 'jeem': 'ج', 'kaaf': 'ك', 'khaa': 'خ', 'la': 'لا', 
    'laam': 'ل', 'meem': 'م', 'nun': 'ن', 'ra': 'ر', 'saad': 'ص', 'seen': 'س', 'sheen': 'ش', 'taa': 'ت', 
    'taad': 'ط', 'tha': 'ث', 'thaa': 'ث', 'tah': 'ط', 'waw': 'و', 'ya': 'ي', 'yaa': 'ي', 'zay': 'ز', 'thal': 'ذ', 
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
    
    ui_font_tuple = ('Arial', font_size_ui, 'bold')
    
except IOError:
    print(f"Error: Font files ({font_path} or arial.ttf) not found. Default fonts will be used.")
    font = ImageFont.load_default()
    label_font = ImageFont.load_default()
    ui_font_tuple = ('TkDefaultFont', 14)
    
LOG_FILE = 'prediction_log_rf.txt'
SCREENSHOT_DIR = 'screenshots'
# Lütfen bu yolu kendi model dosyanızın bulunduğu klasörle güncelleyin!
MODELS_DIR = r'C:\Users\ridva\.conda\envs\vmamba_env\vmamba_proje\random_forest_v1' 

language_configs = {
    'ASL': {'model_file': 'random_forest_EN.pkl', 'label_map_file': 'label_map_EN.pkl', 'feature_size': 42},
    'ArSL': {'model_file': 'random_forest_AR.pkl', 'label_map_file': 'label_map_AR.pkl', 'feature_size': 42},
    'TSL': {'model_file': 'random_forest_TR.pkl', 'label_map_file': 'label_map_TR.pkl', 'feature_size': 84}, 
}

current_language = 'TSL'
model = None
labels_dict = None

if not os.path.exists(SCREENSHOT_DIR):
    os.makedirs(SCREENSHOT_DIR)
    
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# max_num_hands=2 olarak kalmalıdır, TSL modeli iki eli de bekleyebilir
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=2)

hand_present = False
current_latin_label = "" 
current_character = "" 
hand_start_time = None
last_predicted_char = "" 


# --- CORE FUNCTIONS (NameError'ı düzeltmek için bu kısım burada olmalıdır) ---

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
    """Loads the Random Forest model and label map for the specified language."""
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
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(label_map_path, 'rb') as f:
            labels_dict = pickle.load(f)
            
        current_language = lang_code
        print(f"[{lang_code}] Random Forest model and labels loaded successfully.")
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
    
    language_display_text = get_display_language_label(current_language)
    
    # Initialize PIL image for drawing complex text (Arapça etiket ve harf)
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # --- DİL ETİKETİ RENDERLAMA (Sol Üst Köşe) ---
    
    if current_language == 'ArSL':
        # PIL kullanarak Arapça'yı çiz
        text_position = (10, 10) 
        text_color = (255, 0, 0) # Kırmızı (RGB)
        
        try:
            draw.text(text_position, language_display_text, font=label_font, fill=text_color, direction="rtl")
        except Exception as e:
            # Font hatası olursa OpenCV fallback
            cv2.putText(frame, language_display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        
        # PIL'den OpenCV frame'ine geri dön (bu frame üzerine el çizimleri yapılacak)
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # Yeniden başlat
        draw = ImageDraw.Draw(pil_image)
        
    else:
        # TSL ve ASL için basit OpenCV metni kullan
        cv2.putText(frame, 
                    language_display_text, 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8,      
                    (0, 0, 255), # Kırmızı renk (BGR)
                    2,        
                    cv2.LINE_AA) 
        # PIL görüntüsünü sadece prediction için güncel tut
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
    
    # --- El Tespiti ve Tahmin Mantığı ---
    if results.multi_hand_landmarks and model and labels_dict:
        hand_present = True
        data_aux = [] 
        
        for hand_landmarks in results.multi_hand_landmarks:
            
            # Mediapipe El Çizimi
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                      mp_drawing_styles.get_default_hand_landmarks_style(), 
                                      mp_drawing_styles.get_default_hand_connections_style())
            
            # Koordinat Toplama ve Normalizasyon
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            all_x_coords.extend(x_coords)
            all_y_coords.extend(y_coords)
            min_x, min_y = min(x_coords), min(y_coords)
            
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min_x)
                data_aux.append(hand_landmarks.landmark[i].y - min_y)

        # Özellik Vektörü Doldurma/Kesme
        expected_size = language_configs[current_language]['feature_size']
        final_data_vector = data_aux[:expected_size]
        if len(final_data_vector) < expected_size: 
            final_data_vector.extend([0.0] * (expected_size - len(final_data_vector))) 

        if len(final_data_vector) == expected_size:
            try:
                # Tahmin
                prediction = model.predict([np.asarray(final_data_vector)])
                predicted_latin_label = labels_dict.get(int(prediction[0]), 'UNKNOWN')
                predicted_display_char = get_display_character(predicted_latin_label, current_language)
                current_latin_label = predicted_latin_label
                current_character = predicted_display_char
                
                # Bounding Box
                if all_x_coords and all_y_coords: 
                    x1 = max(0, int(min(all_x_coords) * W) - 20)
                    y1 = max(0, int(min(all_y_coords) * H) - 20)
                    x2 = min(W, int(max(all_x_coords) * W) + 20)
                    y2 = min(H, int(max(all_y_coords) * H) + 20)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    
                    # Harfi Çizme (PIL ile)
                    pil_image_for_char = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw_char = ImageDraw.Draw(pil_image_for_char)
                    
                    text_position = (x1, max(0, y1 - 80)) 
                    direction_mode = "rtl" if current_language == 'ArSL' else "ltr"
                    
                    try:
                        text_bbox = draw_char.textbbox(text_position, predicted_display_char, font=font, direction=direction_mode)
                        draw_char.rectangle(text_bbox, fill=(255, 255, 255))
                        draw_char.text(text_position, predicted_display_char, font=font, fill=(0, 0, 0), direction=direction_mode)
                        frame = cv2.cvtColor(np.array(pil_image_for_char), cv2.COLOR_RGB2BGR)
                    except:
                         cv2.putText(frame, predicted_display_char, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                
            except Exception as e:
                # print(f"Prediction Error: {e}")
                predicted_display_char = "ERROR"
                
    else:
        hand_present = False
        hand_start_time = None
        predicted_display_char = ""
        current_latin_label = ""
        current_character = ""

    # 5. Kelime Oluşturma Mantığı (3 saniye kuralı)
    if hand_present and current_character not in ["", "?", "UNKNOWN", "ERROR"]:
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
        # Eğer kamera okumazsa, Tkinter döngüsünü durdurmak yerine kendini tekrar çağırır.
        video_label.after(10, update_video_feed)
        return

    frame = cv2.resize(frame, (800, 600))
    frame = cv2.flip(frame, 1)

    processed_frame = process_frame(frame)
    
    # OpenCV'den PIL'e ve Tkinter'a dönüşüm
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
    """Updates the style of the active language button (Green for active)."""
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
root.title("👋 Sign Language Recognition - Pro")
root.geometry("1180x750") 
root.resizable(False, False)

# 1. Style Definition (ttk)
style = ttk.Style()
style.theme_use('clam')
style.configure('TFrame', background='#f0f0f0')
style.configure('TLabel', background='#f0f0f0', font=('Arial', 14))
style.configure('TButton', font=('Arial', 14, 'bold'), padding=10, relief='flat', background='#e1e1e1', foreground='#333333')
style.map('TButton', background=[('active', '#cccccc')], foreground=[('disabled', '#a0a0a0')])
style.configure('Active.TButton', background='#28a745', foreground='white')
style.map('Active.TButton', background=[('active', '#1e7e34')])

# 2. Main Frames
main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill='both', expand=True)

# 3. Top Control Frame
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

# 4. Main Content Frame
main_content_frame = ttk.Frame(main_frame)
main_content_frame.pack(fill='both', expand=True)

# 5. Left Panel (Video Feed)
video_frame = ttk.Frame(main_content_frame, relief='solid', borderwidth=2, padding=5)
video_frame.pack(side='left', padx=10, pady=10)
video_label = tk.Label(video_frame, text="Camera Feed is Starting...", background='black', foreground='white', width=800, height=600)
video_label.pack()

# 6. Right Panel (Language Buttons and Operations)
right_panel = ttk.Frame(main_content_frame, width=320, style='TFrame') 
right_panel.pack(side='right', fill='y', padx=10, pady=10)
right_panel.grid_propagate(False)
ttk.Label(right_panel, text="Language Selection", font=('Arial', 16, 'bold')).grid(row=0, column=0, columnspan=2, pady=(10, 20), sticky='ew')

# Language Buttons
ar_button = ttk.Button(right_panel, text="ArSL (العربية)", command=lambda: change_language_and_load('ArSL'), width=22) 
ar_button.grid(row=1, column=0, columnspan=2, pady=10, padx=5, sticky='ew')
tr_button = ttk.Button(right_panel, text="TSL (Turkish)", command=lambda: change_language_and_load('TSL'), width=22) 
tr_button.grid(row=2, column=0, columnspan=2, pady=10, padx=5, sticky='ew')
en_button = ttk.Button(right_panel, text="ASL (American)", command=lambda: change_language_and_load('ASL'), width=22) 
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
    
    # Kamerayı başlatmadan önce kaynakları yükle ve NameError'ı engelle
    if load_resources(current_language):
        update_button_style(current_language)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
             print("Error: Camera is not starting. Check camera connection or permissions.")
             sys.exit() # Kamerayı başlatamazsak programı sonlandır

        update_video_feed()
        root.mainloop()
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Startup Error. Please ensure model files are loaded correctly and path is correct.")
        sys.exit()