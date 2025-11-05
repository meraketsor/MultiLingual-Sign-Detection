import cv2
import mediapipe as mp
import numpy as np
import pickle
import tkinter as tk
from PIL import Image, ImageTk, ImageFont, ImageDraw
import time
import os

# --- YAPILANDIRMA VE ARAPÇA ÇEVİRİ HARİTASI ---

# Arapça modelinizin çıktı etiketlerini (Latince) gerçek Arapça harflerine çeviren harita.
# Bu harita, AR modelinizin çıktı etiketleriyle eşleşmelidir.
ARABIC_CHAR_MAP = {
    'ain': 'ع',
    'al': 'ل',
    'aleff': 'ا',
    'bb': 'ب',
    'dal': 'د',
    'dha': 'ذ',
    'dhad': 'ض',
    'fa': 'ف',
    'gaaf': 'ق',
    'ghain': 'غ',
    'ha': 'ه',
    'haa': 'ح',
    'jeem': 'ج',
    'kaaf': 'ك',
    'khaa': 'خ',
    'la': 'ل', # Genellikle aynı harfi temsil edebilir, modelinize göre düzeltin.
    'laam': 'ل',
    'meem': 'م',
    'nun': 'ن',
    'ra': 'ر',
    'saad': 'ص',
    'seen': 'س',
    'sheen': 'ش',
    'ta': 'ط',
    'taa': 'ت',
    'thaa': 'ث',
    'thal': 'ذ', # Genellikle aynı harfi temsil edebilir, modelinize göre düzeltin.
    'toot': 'ط', # Alternatif Tı harfi için, modelinize göre düzeltin.
    'waw': 'و',
    'ya': 'ي',
    'yaa': 'ي',
    'zay': 'ز',
    'UNKNOWN': '?', # Bilinmeyen durumlar için
    # TR ve EN dilleri için bu haritaya gerek yoktur.
}


# Font dosyasını yükle
# 'a.ttf' dosyasının projenin ana dizininde olduğundan emin ol.
try:
    # Kullanıcının belirttiği fontu dene
    font = ImageFont.truetype("a.ttf", 75, encoding="utf-8")
    ui_font = ('a.ttf', 18)
except IOError:
    # Bulunamazsa Arial'a geri dön
    print("a.ttf font dosyası bulunamadı, Arial fontu kullanılacak.")
    # Windows/Linux'ta varsayılan bir fontu kullanmaya çalış
    try:
        font = ImageFont.truetype("arial.ttf", 75)
        ui_font = ('Arial', 18)
    except IOError:
        # Eğer arial da yoksa, basit bir font kullan
        font = ImageFont.load_default() 
        ui_font = ('Arial', 18) # Tkinter varsayılanı

# Tahmin sürelerinin kaydedileceği dosya
LOG_FILE = 'prediction_log.txt'

# Modellerin ve etiket haritalarının ana dizini
MODELS_DIR = r'C:\Users\ridva\.conda\envs\vmamba_env\vmamba_proje\decision_tree_v1' 

# Dil modelleri, dosya adları ve beklenen özellik vektörü boyutları
language_configs = {
    # Decision Tree EN ve AR tek el (42 özellik) kullanacak
    'EN': {'model_file': 'decision_tree_EN.pkl', 'label_map_file': 'label_map_EN.pkl', 'feature_size': 42},
    'AR': {'model_file': 'decision_tree_AR.pkl', 'label_map_file': 'label_map_AR.pkl', 'feature_size': 42},
    # Decision Tree TR tek veya iki el (84 özellik) kullanacak
    'TR': {'model_file': 'decision_tree_TR.pkl', 'label_map_file': 'label_map_TR.pkl', 'feature_size': 84}, 
}

current_language = 'TR'
model = None
labels_dict = None

# Modeli ve etiket haritasını yükleyen fonksiyon
def load_resources(lang):
    """Belirtilen dile ait Decision Tree modelini ve etiket haritasını yükler."""
    global model, labels_dict

    config = language_configs[lang]
    model_path = os.path.join(MODELS_DIR, config['model_file'])
    label_map_path = os.path.join(MODELS_DIR, config['label_map_file'])

    if not os.path.exists(model_path) or not os.path.exists(label_map_path):
        print(f"Hata: {lang} diline ait Decision Tree modeli veya etiket dosyası bulunamadı.")
        print(f"Kontrol edilen yollar: \n Model: {model_path} \n Etiket: {label_map_path}")
        return False

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(label_map_path, 'rb') as f:
            labels_map = pickle.load(f)
            # Etiket haritası: Sayısal (int) -> Gerçek Harf (str - Latince etiket)
            labels_dict = labels_map

        print(f"[{lang}] dilindeki Decision Tree modeli ve etiketler başarıyla yüklendi. Beklenen boyut: {config['feature_size']}")
        return True

    except Exception as e:
        print(f"[{lang}] dilindeki modeli yüklerken bir hata oluştu: {e}")
        return False

def log_prediction_time(duration_ms, prediction):
    """Tahmin süresini ve sonucunu bir metin dosyasına kaydeder."""
    # Dosyaya ekleme ('a' modu) ile kaydetme
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"Zaman: {time.strftime('%Y-%m-%d %H:%M:%S')} | Tahmin: {prediction} | Süre: {duration_ms:.2f} ms\n")

# Mediapipe'ı başlat
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# İki el tespiti için max_num_hands=2 olarak ayarlandı.
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=2)

# UI ayarları (Tkinter)
root = tk.Tk()
root.title("Hand Gesture Recognition - Decision Tree")

char_var = tk.StringVar()
sentence_var = tk.StringVar()

# Font ayarı UI için kullanılıyor
char_label = tk.Label(root, text="Tahmin Edilen Karakter:", font=('Arial', 16))
char_entry = tk.Entry(root, textvariable=char_var, font=ui_font, state='readonly', width=5, justify='right') # Arapça için sağa yaslı
sentence_label = tk.Label(root, text="Oluşan Kelime:", font=('Arial', 16))
sentence_entry = tk.Entry(root, textvariable=sentence_var, font=ui_font, state='readonly', width=20, justify='right') # Arapça için sağa yaslı

char_label.grid(row=0, column=0, padx=10, pady=10)
char_entry.grid(row=0, column=1, padx=10, pady=10)
sentence_label.grid(row=1, column=0, padx=10, pady=10)
sentence_entry.grid(row=1, column=1, padx=10, pady=10)

hand_present = False
current_latin_label = "" # Modelden gelen Latince etiket (örn: 'sheen')
current_character = "" # Görüntülenecek karakter (örn: 'ش' veya 'A')
hand_start_time = None

def get_display_character(latin_label, lang):
    """
    Modelin verdiği Latince etiketi (örn. 'sheen') dil bazında doğru karaktere (örn. 'ش') çevirir.
    """
    if lang == 'AR':
        # Arapça için çeviri haritasını kullan
        return ARABIC_CHAR_MAP.get(latin_label, latin_label)
    # Diğer diller (EN, TR) için modelin etiketini aynen kullan
    return latin_label

def process_frame(frame):
    """
    Video karesini işler, el özelliklerini çıkarır, tahmin yapar ve görüntüyü günceller.
    """
    global hand_present, current_latin_label, current_character, hand_start_time

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # Varsayılan değerler
    predicted_latin_label = "" # Modelden gelen Latince etiket
    predicted_display_char = "" # Ekranda gösterilecek karakter (Arapça veya Latince)
    
    # Tüm ellerin koordinatlarını toplamak için listeler
    all_x_coords = []
    all_y_coords = []

    if results.multi_hand_landmarks:
        hand_present = True
        data_aux = [] 
        
        # Sadece ilk 2 eli (Mediapipe'ın bulduğu sırayla) işle
        for hand_landmarks in results.multi_hand_landmarks[:2]:
            
            # 1. Landmarks'ları çerçevede göster
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # 2. Bounding box ve tahmin için koordinatları topla
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            
            all_x_coords.extend(x_coords)
            all_y_coords.extend(y_coords)
            
            # 3. Özellikleri normalleştir (Göreceli Konum)
            min_x, min_y = min(x_coords), min(y_coords)
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                
                # Göreli konumu kaydet
                data_aux.append(x - min_x)
                data_aux.append(y - min_y)

        # 4. Özellik Vektörünü Modelin Beklediği Boyuta Tamamla (Padding/Truncation)
        expected_size = language_configs[current_language]['feature_size']
        
        # Eğer model tek el (42) bekliyorsa ve iki el algılanmışsa (len(data_aux) > 42), sadece ilk eli al.
        if expected_size == 42 and len(data_aux) > 42:
            final_data_vector = data_aux[:42]
        else:
            final_data_vector = data_aux[:expected_size]
            
            # Eğer beklenen boyuttan kısaysa, sıfırlarla doldur (Padding)
            if len(final_data_vector) < expected_size:
                padding_needed = expected_size - len(final_data_vector)
                final_data_vector.extend([0.0] * padding_needed) 

        # 5. Tahmin Yap ve Zamanı Ölç
        if len(final_data_vector) == expected_size:
            
            start_time = time.time() # Ölçüm başlangıcı
            try:
                prediction = model.predict([np.asarray(final_data_vector)])
                end_time = time.time() # Ölçüm sonu
                
                duration_ms = (end_time - start_time) * 1000 # Tahmin süresi (ms)
                
                # Modelden gelen Latince etiketi al (örn: 'sheen')
                predicted_latin_label = labels_dict.get(int(prediction[0]), 'UNKNOWN')
                
                # Ekranda gösterilecek karakteri al (örn: 'ش')
                predicted_display_char = get_display_character(predicted_latin_label, current_language)
                
                current_latin_label = predicted_latin_label
                current_character = predicted_display_char
                
                # Tahmin süresini metin dosyasına logla (logda Latince etiket kullanmak daha iyi)
                log_prediction_time(duration_ms, predicted_latin_label) 

                # 6. Sınırlayıcı Kutu (Bounding Box) ve Metin Çizimi
                x1 = int(min(all_x_coords) * W) - 20 
                y1 = int(min(all_y_coords) * H) - 20
                x2 = int(max(all_x_coords) * W) + 20
                y2 = int(max(all_y_coords) * H) + 20

                # Koordinatları çerçeve sınırları içinde tut
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W, x2)
                y2 = min(H, y2)

                # Bounding box çiz
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

                # Tahmin edilen harfi (Arapça/Latince) görüntüye yaz
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)
                
                text_position = (x1, max(0, y1 - 80)) 
                
                # PIL, Arapça metin için sağdan sola desteği sağlamak adına bidi=True kullanır.
                text_size = draw.textbbox(text_position, predicted_display_char, font=font, direction="rtl")
                
                # Arka plan kutusu çiz (Daha iyi okunurluk için)
                draw.rectangle((text_size[0], text_size[1], text_size[2], text_size[3]), fill=(255, 255, 255))
                
                # Metni kutunun üstüne, biraz yukarıya konumlandır (Arapça için yön önemlidir)
                # direction="rtl" Arapça metinleri doğru yönde yazdırır.
                draw.text(text_position, predicted_display_char, font=font, fill=(0, 0, 0), direction="rtl")
                
                # İşlem süresini çerçeveye yaz
                time_text = f"DT: {duration_ms:.2f} ms"
                cv2.putText(frame, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) # Mavi renk
                
                frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            except Exception as e:
                print(f"Tahmin/Çizim Hatası: {e}")
                
        # 7. Kelime Oluşturma Mantığı (3 saniye kuralı)
        current_time = time.time()
        if hand_start_time is None:
            hand_start_time = current_time
        elif current_time - hand_start_time >= 3:
            # Sadece geçerli bir karakter varsa ekle
            if current_character not in ["", "?", "UNKNOWN"]:
                # Arapça'da karakterleri tersten eklemek gerekir (sağdan sola)
                new_sentence = current_character + sentence_var.get()
                sentence_var.set(new_sentence)
            
            # Tkinter'da gösterilecek karakteri ayarla
            char_var.set(current_character)
            
            current_character = ""
            hand_start_time = None
            
    else:
        # El yoksa zamanlayıcıyı sıfırla
        hand_present = False
        hand_start_time = None
        char_var.set("") # El yoksa tahmin edilen karakteri temizle


    return frame

# UI ve diğer fonksiyonlar (Aynı Bırakıldı)

def update_video_feed():
    """Kamera akışını güncelleyen ana döngü."""
    ret, frame = cap.read()
    if not ret:
        video_label.after(10, update_video_feed)
        return

    # Frame boyutlandırma ve çevirme
    frame = cv2.resize(frame, (800, 600))
    frame = cv2.flip(frame, 1)

    processed_frame = process_frame(frame)
    
    # Tkinter'a uygun görüntü formatına dönüştür
    photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)))
    video_label.config(image=photo)
    video_label.photo = photo
    video_label.after(10, update_video_feed)

def change_language_and_load(lang):
    """Dil değiştirir ve yeni modeli yükler."""
    global current_language
    current_language = lang
    if load_resources(current_language):
        # Arapça'ya geçişte entry kutularının yönünü sağa yasla
        if lang == 'AR':
            char_entry.config(justify='right')
            sentence_entry.config(justify='right')
        else:
            # Diğer dillerde (TR, EN) sola yaslı kalabilir
            char_entry.config(justify='left')
            sentence_entry.config(justify='left')
        clear_characters()

def clear_characters():
    """Metin kutularını ve zamanlayıcıyı sıfırlar."""
    sentence_var.set("")
    char_var.set("")
    global current_latin_label, current_character, hand_start_time
    current_latin_label = ""
    current_character = ""
    hand_start_time = None

def delete_last_character():
    """Son karakteri siler."""
    current_sentence = sentence_var.get()
    
    if current_sentence:
        # Arapça sağdan sola yazıldığı için en baştaki karakter silinir
        if current_language == 'AR':
            sentence_var.set(current_sentence[1:]) 
        else:
            # Diğer dillerde en sondaki karakter silinir
            sentence_var.set(current_sentence[:-1])
    
    char_var.set("")
    global current_latin_label, current_character, hand_start_time
    current_latin_label = ""
    current_character = ""
    hand_start_time = None

# Butonlar 
delete_button = tk.Button(root, text="Sil", command=delete_last_character)
delete_button.grid(row=3, column=4, padx=10, pady=10)
clear_button = tk.Button(root, text="Temizle", command=clear_characters)
clear_button.grid(row=3, column=3, padx=10, pady=10)
en_button = tk.Button(root, text="EN (DT - 1H)", command=lambda: change_language_and_load('EN'))
en_button.grid(row=3, column=1, padx=10, pady=10)
ar_button = tk.Button(root, text="AR (DT - 1H)", command=lambda: change_language_and_load('AR'))
ar_button.grid(row=3, column=0, padx=10, pady=10)
tr_button = tk.Button(root, text="TR (DT - 1/2H)", command=lambda: change_language_and_load('TR'))
tr_button.grid(row=3, column=2, padx=10, pady=10)

if __name__ == '__main__':
    # Başlamadan önce log dosyasını temizle
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        print(f"{LOG_FILE} temizlendi.")
        
    if load_resources(current_language):
        cap = cv2.VideoCapture(0)
        # Kamera başlatılamazsa hata ver
        if not cap.isOpened():
             print("Hata: Kamera başlatılamadı.")
             exit()

        video_label = tk.Label(root)
        video_label.grid(row=2, column=0, columnspan=5, padx=10, pady=10)
        update_video_feed()
        root.mainloop()
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Başlatma başarısız oldu. Lütfen Decision Tree model dosyalarının doğru yolda ve doğru formatta olduğundan emin olun.")
