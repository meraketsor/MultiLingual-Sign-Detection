import cv2
import mediapipe as mp
import numpy as np
import pickle
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageFont, ImageDraw
import time
import os
import pyautogui 
import sys
from tensorflow.keras.models import load_model

# =========================================================================
# YAPILANDIRMA VE SABİTLER
# =========================================================================

# Arapça Harf Haritası
ARABIC_CHAR_MAP = {
    'ain': 'ع', 'al': 'ال', 'aleff': 'ا', 'bb': 'ب', 'dal': 'د', 'dha': 'ظ', 'dhad': 'ض', 'fa': 'ف', 
    'gaaf': 'ق', 'ghain': 'غ', 'ha': 'ه', 'haa': 'ح', 'jeem': 'ج', 'kaaf': 'ك', 'khaa': 'خ', 'la': 'لا', 
    'laam': 'ل', 'meem': 'م', 'nun': 'ن', 'ra': 'ر', 'saad': 'ص', 'seen': 'س', 'sheen': 'ش', 'taa': 'ت', 
    'taad': 'ط', 'tha': 'ث', 'thh': 'ث', 'tah': 'ط', 'ya': 'ي', 'yaa': 'ي', 'zay': 'ز', 'zal': 'ذ', 
    'UNKNOWN': '?',
}

# Font yükleme
try:
    font_path = "a.ttf"
    if not os.path.exists(font_path):
        font_path = "arial.ttf"
    
    font = ImageFont.truetype(font_path, 75, encoding="utf-8")
    ui_font = ('Arial', 18)
except IOError:
    print("Hata: Font dosyası bulunamadı, varsayılan font kullanılıyor.")
    font_small = ImageFont.truetype(font_path, 20, encoding="utf-8")
    ui_font = ('Arial', 18)

# Log Dosyası
LOG_FILE = 'prediction_log_cnn.txt'
SCREENSHOT_DIR = 'screenshots'

# Model Klasörü (Sizin belirttiğiniz yol)
MODELS_DIR = r'C:\Users\ridva\.conda\envs\vmamba_env\vmamba_proje\cnn_v1'

# Dil Yapılandırmaları
language_configs = {
    'ASL': {'model_file': 'cnn_EN.h5', 'label_map_file': 'label_map_EN.pkl', 'feature_size': 42, 'hands_required': 1},
    'ArSL': {'model_file': 'cnn_AR.h5', 'label_map_file': 'label_map_AR.pkl', 'feature_size': 42, 'hands_required': 1},
    'TSL': {'model_file': 'cnn_TR.h5', 'label_map_file': 'label_map_TR.pkl', 'feature_size': 84, 'hands_required': 2}, 
}

# Tahmin Ayarları
# Güven eşiğini biraz düşürdük (0.5) ki model daha kolay tahmin üretebilsin.
# Eğer çok fazla yanlış tahmin olursa bunu artırabilirsiniz (0.7, 0.8 gibi).
PREDICTION_CONFIDENCE_THRESHOLD = 0.50 

# Global Değişkenler
current_language = 'TSL' # Başlangıç dili
model = None
labels_dict = None
hand_present = False
current_latin_label = "" 
current_character = "" 
hand_start_time = None
last_predicted_char = "" 

# Klasör kontrolü
if not os.path.exists(SCREENSHOT_DIR):
    os.makedirs(SCREENSHOT_DIR)
    
# Mediapipe Ayarları
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# Statik mod False (video için), güvenilirlik 0.5
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)


# =========================================================================
# YARDIMCI İŞLEVLER
# =========================================================================

def take_screenshot():
    """Ekran görüntüsü alır ve kaydeder."""
    try:
        screenshot = pyautogui.screenshot() 
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join(SCREENSHOT_DIR, filename)
        screenshot.save(filepath)
        print(f"Screenshot saved: {filepath}")
        messagebox.showinfo("Ekran Görüntüsü", f"Kaydedildi:\n{filepath}")
    except Exception as e:
        print(f"Error taking screenshot: {e}")

def load_resources(lang_code):
    """CNN modelini ve etiket haritasını yükler."""
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
        messagebox.showerror("Hata", f"Model dosyaları bulunamadı!\n{model_path}")
        return False

    try:
        # CNN modelini yükle (Keras)
        model = load_model(model_path)
        print(f"Model yüklendi: {lang_code}")
        print(f"  Input Shape: {model.input_shape}")
        
        # Etiket haritasını yükle (Pickle)
        with open(label_map_path, 'rb') as f:
            labels_dict = pickle.load(f)
            
        current_language = lang_code
        print(f"  Etiketler yüklendi. Sınıf sayısı: {len(labels_dict)}")
        return True

    except Exception as e:
        print(f"Error loading {lang_code} model: {e}")
        messagebox.showerror("Hata", f"Model yüklenirken hata: {e}")
        model = None
        labels_dict = None
        return False

def log_prediction_time(duration_ms, prediction, confidence):
    """Tahmin süresini ve sonucunu loglar."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')} | Lang: {current_language} | Pred: {prediction} ({confidence:.2f}) | Dur: {duration_ms:.2f} ms\n")
        
def get_display_character(latin_label, lang):
    """Model çıktısını ekranda gösterilecek karaktere dönüştürür."""
    if lang == 'ArSL':
        return ARABIC_CHAR_MAP.get(latin_label.lower(), latin_label)
    return latin_label
    
def get_display_language_label(lang_code):
    """Aktif dilin ekrandaki adını döndürür."""
    labels = {'ArSL': 'العربية (ArSL)', 'TSL': 'Türkçe (TSL)', 'ASL': 'English (ASL)'}
    return labels.get(lang_code, lang_code)

def process_frame(frame):
    """Kareyi işler, el tespiti ve tahmin yapar."""
    global hand_present, current_latin_label, current_character, hand_start_time, last_predicted_char

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    predicted_latin_label = "..." 
    predicted_display_char = "..." 
    all_x_coords = []
    all_y_coords = []
    prediction_confidence = 0.0
    detected_hands = 0
    
    language_display_text = get_display_language_label(current_language)
    hands_required = language_configs[current_language]['hands_required']
    feature_size = language_configs[current_language]['feature_size']
    
    # --- 1. Dil Etiketini Çiz ---
    cv2.putText(frame, language_display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    
    # --- 2. El Tespiti ve Veri Hazırlama ---
    if results.multi_hand_landmarks and model and labels_dict:
        detected_hands = len(results.multi_hand_landmarks)
        hand_present = True
        data_aux = [] 
        
        # Tespit edilen her el için
        for hand_landmarks in results.multi_hand_landmarks:
            # Çizim
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                    mp_drawing_styles.get_default_hand_landmarks_style(), 
                                    mp_drawing_styles.get_default_hand_connections_style())
            
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            all_x_coords.extend(x_coords)
            all_y_coords.extend(y_coords)
            
            # Normalizasyon (Bounding Box'a göre değil, elin kendi min/max'ına göre)
            min_x, min_y = min(x_coords), min(y_coords)
            
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min_x)
                data_aux.append(hand_landmarks.landmark[i].y - min_y)

        # --- KRİTİK: Veri Boyutu Ayarlama (Padding/Truncation) ---
        # Modelin beklediği boyuta (feature_size) göre veriyi ayarla.
        
        # Eğer veri fazla ise kes
        if len(data_aux) > feature_size:
            final_data_vector = data_aux[:feature_size]
        # Eğer veri eksik ise (örn: TSL'de 2 el bekleniyor ama 1 el var), sıfırla doldur
        elif len(data_aux) < feature_size:
            padding = [0.0] * (feature_size - len(data_aux))
            final_data_vector = data_aux + padding
        else:
            final_data_vector = data_aux

        # --- 3. Tahmin ---
        if len(final_data_vector) == feature_size:
            try:
                start_time = time.time()
                
                # Modelin input shape'ine göre veriyi şekillendir
                # Genellikle CNN (1, 42, 1) veya (1, 84, 1) bekler
                input_data = np.array(final_data_vector).reshape(1, feature_size, 1)
                
                predictions = model.predict(input_data, verbose=0)
                end_time = time.time()
                
                predicted_index = np.argmax(predictions[0])
                prediction_confidence = np.max(predictions[0])
                
                # Güven Eşiği Kontrolü
                if prediction_confidence >= PREDICTION_CONFIDENCE_THRESHOLD:
                    predicted_latin_label = labels_dict.get(predicted_index, '?') # int index kullanılıyor olabilir
                    if predicted_latin_label == '?':
                        # Bazen sözlük key'leri string olabilir, bir de öyle deneyelim
                        predicted_latin_label = labels_dict.get(str(predicted_index), 'UNKNOWN')

                    predicted_display_char = get_display_character(predicted_latin_label, current_language)
                    
                    current_latin_label = predicted_latin_label
                    current_character = predicted_display_char
                    
                    # Konsola yazdır (Hata ayıklama için)
                    # print(f"Tahmin: {predicted_latin_label} ({prediction_confidence:.2f}) - Eller: {detected_hands}/{hands_required}")
                else:
                    predicted_display_char = "..." # Düşük güven
                
                duration_ms = (end_time - start_time) * 1000
                # log_prediction_time(duration_ms, predicted_latin_label, prediction_confidence)

                # --- 4. Görselleştirme (Kutu ve Metin) ---
                if all_x_coords:
                    x1 = max(0, int(min(all_x_coords) * W) - 20)
                    y1 = max(0, int(min(all_y_coords) * H) - 20)
                    x2 = min(W, int(max(all_x_coords) * W) + 20)
                    y2 = min(H, int(max(all_y_coords) * H) + 20)
                    
                    color = (0, 255, 0) if prediction_confidence >= PREDICTION_CONFIDENCE_THRESHOLD else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                    
                    # PIL ile metin yazma
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_image)
                    
                    # Metin konumu
                    text_x = x1
                    text_y = max(0, y1 - 80)
                    
                    # Tahmini yaz
                    draw.text((text_x, text_y), predicted_display_char, font=font, fill=color)
                    
                    # Güven skorunu yaz
                    conf_text = f"{prediction_confidence:.2f}"
                    draw.text((text_x, y1 - 30), conf_text, font=font_small, fill=(255, 255, 255))
                    
                    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            except Exception as e:
                print(f"Prediction Error: {e}")

    else:
        hand_present = False
        hand_start_time = None
        current_character = ""

    # --- 5. Kelime Oluşturma (3 Saniye Kuralı) ---
    if (hand_present and current_character not in ["", "...", "?", "UNKNOWN"] and 
        prediction_confidence >= PREDICTION_CONFIDENCE_THRESHOLD):
        
        if hand_start_time is None or last_predicted_char != current_character:
            hand_start_time = time.time()
            last_predicted_char = current_character
            
        current_time = time.time()
        
        if hand_start_time is not None and current_time - hand_start_time >= 3:
            # Karakteri cümleye ekle
            current_sentence = sentence_var.get()
            if current_language == 'ArSL':
                sentence_var.set(current_character + current_sentence) # Arapça için başa ekle
            else:
                sentence_var.set(current_sentence + current_character) # Diğerleri için sona ekle
                
            last_predicted_char = "" 
            hand_start_time = None # Zamanlayıcıyı sıfırla
            print(f"Karakter Eklendi: {current_character}")
    else:
        hand_start_time = None
    
    # Arayüzdeki tahmin kutusunu güncelle
    char_var.set(predicted_display_char)
    
    return frame

# =========================================================================
# TKINTER ARAYÜZÜ (Random Forest ile Aynı Stil)
# =========================================================================

# Ana Fonksiyonlar
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
        # Giriş kutusu hizalaması
        justify = 'right' if lang == 'ArSL' else 'left'
        char_entry.config(justify=justify)
        sentence_entry.config(justify=justify)
        
        update_button_style(lang)
        clear_characters()

def update_button_style(active_lang):
    buttons = {'ArSL': ar_button, 'TSL': tr_button, 'ASL': en_button}
    for lang, button in buttons.items():
        if lang == active_lang:
            button.config(style='Active.TButton')
        else:
            button.config(style='TButton')

def clear_characters():
    sentence_var.set("")
    char_var.set("")
    global current_character, hand_start_time
    current_character = ""
    hand_start_time = None

def delete_last_character():
    current_s = sentence_var.get()
    if current_s:
        if current_language == 'ArSL':
             sentence_var.set(current_s[1:])
        else:
             sentence_var.set(current_s[:-1])
    char_var.set("")

def check_screenshot_key(event):
    if event.char.lower() == 's':
        take_screenshot()

# Arayüz Kurulumu
root = tk.Tk()
root.title("👋 Sign Language Recognition - CNN Pro")
root.geometry("1180x750") 
root.resizable(False, False)

# Stil
style = ttk.Style()
style.theme_use('clam')
style.configure('TFrame', background='#f0f0f0')
style.configure('TLabel', background='#f0f0f0', font=('Arial', 14))
style.configure('TButton', font=('Arial', 14, 'bold'), padding=10, relief='flat', background='#e1e1e1', foreground='#333333')
style.map('TButton', background=[('active', '#cccccc')], foreground=[('disabled', '#a0a0a0')])
style.configure('Active.TButton', background='#28a745', foreground='white')
style.map('Active.TButton', background=[('active', '#1e7e34')])

# Ana Çerçeveler
main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill='both', expand=True)

# Üst Kontrol Paneli (Metin Alanları)
top_control_frame = ttk.Frame(main_frame)
top_control_frame.pack(fill='x', pady=(0, 10))

char_var = tk.StringVar()
sentence_var = tk.StringVar()

# Etiketler
char_label = ttk.Label(top_control_frame, text="Predicted Character:", font=('Arial', 16, 'bold'))
char_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')

# Tahmin Alanı
char_entry = tk.Entry(top_control_frame, textvariable=char_var, font=('Arial', 18, 'bold'), state='readonly', width=5, justify='left', bd=3, relief='sunken')
char_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')

# Cümle Alanı
sentence_label = ttk.Label(top_control_frame, text="Word:", font=('Arial', 16, 'bold'))
sentence_label.grid(row=0, column=2, padx=(50, 10), pady=10, sticky='w')

sentence_entry = tk.Entry(top_control_frame, textvariable=sentence_var, font=('Arial', 18, 'bold'), state='readonly', width=30, justify='left', bd=3, relief='sunken')
sentence_entry.grid(row=0, column=3, padx=10, pady=10, sticky='ew')
top_control_frame.grid_columnconfigure(3, weight=1)

# Ana İçerik Paneli
main_content_frame = ttk.Frame(main_frame)
main_content_frame.pack(fill='both', expand=True)

# Sol Panel: Video
video_frame = ttk.Frame(main_content_frame, relief='solid', borderwidth=2, padding=5)
video_frame.pack(side='left', padx=10, pady=10)
video_label = tk.Label(video_frame, text="Camera Feed...", background='black', foreground='white', width=800, height=600)
video_label.pack()

# Sağ Panel: Butonlar
right_panel = ttk.Frame(main_content_frame, width=320, style='TFrame') 
right_panel.pack(side='right', fill='y', padx=10, pady=10)
right_panel.grid_propagate(False)

ttk.Label(right_panel, text="Language Selection", font=('Arial', 16, 'bold')).grid(row=0, column=0, columnspan=2, pady=(10, 20), sticky='ew')

# Dil Butonları
ar_button = ttk.Button(right_panel, text="AR ArSL (العربية)", command=lambda: change_language_and_load('ArSL'), width=22) 
ar_button.grid(row=1, column=0, columnspan=2, pady=10, padx=5, sticky='ew')

tr_button = ttk.Button(right_panel, text="TR TSL (Turkish)", command=lambda: change_language_and_load('TSL'), width=22) 
tr_button.grid(row=2, column=0, columnspan=2, pady=10, padx=5, sticky='ew')

en_button = ttk.Button(right_panel, text="US ASL (American)", command=lambda: change_language_and_load('ASL'), width=22) 
en_button.grid(row=3, column=0, columnspan=2, pady=10, padx=5, sticky='ew')

ttk.Label(right_panel, text="").grid(row=4, column=0, columnspan=2, pady=20)
right_panel.grid_columnconfigure(0, weight=1)
right_panel.grid_columnconfigure(1, weight=1)

# İşlem Butonları
delete_button = ttk.Button(right_panel, text="← Delete", command=delete_last_character, width=12) 
delete_button.grid(row=5, column=0, padx=5, pady=10, sticky='ew')

clear_button = ttk.Button(right_panel, text="Clear", command=clear_characters, width=12) 
clear_button.grid(row=5, column=1, padx=5, pady=10, sticky='ew')

root.bind('<Key>', check_screenshot_key)
root.focus_set()

# --- BAŞLATMA ---
if __name__ == '__main__':
    if load_resources(current_language):
        update_button_style(current_language)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
             messagebox.showerror("Hata", "Kamera açılamadı.")
             sys.exit()

        update_video_feed()
        root.mainloop()
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Başlatma hatası: Model dosyaları bulunamadı.")
        sys.exit()