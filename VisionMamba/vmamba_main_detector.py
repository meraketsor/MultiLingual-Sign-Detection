import os
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageFont, ImageDraw
from collections import deque
import time
import torch
import torch.nn as nn
import timm
import json
from scipy.stats import mode

# =========================================================================
# YAPILANDIRMA VE SABİTLER (VMamba'ya özel agresif ayarlar)
# =========================================================================

MODEL_DIR = './vmamba_models'
MODEL_EPOCH = 3
INITIAL_LANGUAGE = 'AR' # Başlangıç dili AR olarak ayarlı

# TAHMİN STABİLİZASYON PARAMETRELERİ (AGRESİF AYARLAR)
PREDICTION_HISTORY_SIZE = 20 # Tahmin geçmişinde tutulacak kare sayısı (Artırıldı)
STABILITY_THRESHOLD = 15 # Bir tahminin geçerli sayılması için tamponda olması gereken minimum sayı (Artırıldı)
CONFIDENCE_THRESHOLD = 0.90 # Modelin tahmin güvenirliği eşiği (Artırıldı)
SENTENCE_ADD_DELAY = 1.0

# Font yükleme (Yerel sisteme bağlıdır)
try:
    font = ImageFont.truetype("arial.ttf", 75, encoding="utf-8")
    font_small = ImageFont.truetype("arial.ttf", 30, encoding="utf-8")
except IOError:
    # Bu hatayı alıyorsanız, fontları yükleyemediğinizden kaynaklanır.
    print("Uyarı: 'arial.ttf' font dosyası bulunamadı. Varsayılan PIL fontu kullanılıyor.")
    font = ImageFont.load_default()
    font_small = ImageFont.load_default()

# Dil bilgileri (Eğitimde kullanılan klasör isimleriyle BİREBİR aynı olmalıdır)
LANGUAGE_INFO = {
    'EN': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
    'TR': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z'],
    'AR': ['ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad', 'fa', 'gaaf', 'ghain', 'ha', 'haa', 'jeem', 'kaaf', 'khaa', 'la', 'laam', 'meem', 'nun', 'ra', 'saad', 'seen', 'sheen', 'ta', 'taa', 'thaa', 'thal', 'toot', 'waw', 'ya', 'yaa', 'zay']
}

# =========================================================================
# Adım 1: VMamba Model Sınıfının Tanımlanması
# =========================================================================
class CustomMambaClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(CustomMambaClassifier, self).__init__()
        # VMamba için varsayılan olarak ConvNeXt kullanıldı
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

# =========================================================================
# Adım 2: Model ve Kaynak Yönetimi
# =========================================================================
model = None
class_map = {}
class_names = []
current_language = INITIAL_LANGUAGE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prediction_history = deque(maxlen=PREDICTION_HISTORY_SIZE)
last_added_character = ""
last_add_time = time.time()

def load_resources(lang):
    """Belirtilen dil için model ağırlıklarını ve sınıf haritasını yükler."""
    global model, class_map, class_names, current_language, prediction_history
    
    model_path = os.path.join(MODEL_DIR, f'vmamba_{lang}_epoch_{MODEL_EPOCH}.pth')
    class_map_path = os.path.join(MODEL_DIR, f'class_map_{lang}.json')
    
    print(f"\n[{lang}] Yükleniyor... Model: {model_path}, Harita: {class_map_path}")

    # 1. Sınıf Haritasını Yükle
    if not os.path.exists(class_map_path):
        messagebox.showerror("HATA", f"Sınıf haritası dosyası '{class_map_path}' bulunamadı. Lütfen kontrol edin.")
        return False
    
    try:
        with open(class_map_path, 'r', encoding='utf-8') as f:
            class_map = json.load(f)
            
        num_classes = len(class_map)
        class_names = sorted(class_map, key=class_map.get)
        print(f"[{lang}] Sınıf sayısı: {num_classes}")
    except Exception as e:
        messagebox.showerror("HATA", f"Sınıf haritası yüklenirken sorun oluştu: {e}")
        return False

    # 2. Modeli Yükle
    if not os.path.exists(model_path):
        messagebox.showerror("HATA", f"Model dosyası '{model_path}' bulunamadı. Bu dil için eğitimi tamamladığınızdan emin olun.")
        model = None
        return False
    
    try:
        new_model = CustomMambaClassifier(num_classes=num_classes, pretrained=False).to(device)
        new_model.load_state_dict(torch.load(model_path, map_location=device))
        new_model.eval()
        
        model = new_model
        current_language = lang
        prediction_history.clear()
        print(f"PyTorch VMamba modeli başarıyla yüklendi. (Dil: {lang})")
        return True
        
    except Exception as e:
        messagebox.showerror("HATA", f"Model yüklenirken bir sorun oluştu: {e}")
        model = None
        return False

# =========================================================================
# Adım 3: Tahmin Stabilizasyonu ve Görüntü İşleme
# =========================================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

def get_stable_prediction(current_pred_idx):
    """Tahmin geçmişini kullanarak stabil bir tahmin döndürür."""
    global prediction_history
    
    if current_pred_idx is not None:
        prediction_history.append(current_pred_idx)
    
    if not prediction_history:
        return None, 0 
    
    # En sık çıkan tahmini bul (Majority Vote)
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
    """Görüntüyü işler, eli tespit eder ve modelle tahmin yapar."""
    global last_added_character, last_add_time, model
    
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    predicted_character = None
    max_confidence = 0.0
    
    stable_char = None 
    stable_count = 0
    
    current_pred_idx = None
    
    if results.multi_hand_landmarks and model is not None:
        # El Tespit Edildi
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # El izlerini çiz
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Kırpma kutusu hesapla
        x_list = [lm.x for lm in hand_landmarks.landmark]
        y_list = [lm.y for lm in hand_landmarks.landmark]
        
        padding = 50
        x1 = int(min(x_list) * W)
        y1 = int(min(y_list) * H)
        x2 = int(max(x_list) * W)
        y2 = int(max(y_list) * H)
        
        box_w = x2 - x1
        box_h = y2 - y1
        max_dim = max(box_w, box_h)
        
        square_side = max_dim + 2 * padding
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

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
            
            # Görüntü Ön İşleme ve Normalizasyon
            image_tensor = torch.from_numpy(resized_hand).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            normalized_hand = (image_tensor - mean) / std
            
            normalized_hand = normalized_hand.to(device)

            # Model Tahmini
            with torch.no_grad():
                outputs = model(normalized_hand.unsqueeze(0))
                probabilities = model.softmax(outputs)
            
            max_confidence, predicted_index = torch.max(probabilities, 1)
            max_confidence = max_confidence.item()
            predicted_index = predicted_index.item()
            
            current_character = class_names[predicted_index]
            current_pred_idx = predicted_index

            # Stabilizasyon Kontrolü
            stable_char, stable_count = get_stable_prediction(current_pred_idx)
            
            # --- Cümle Ekleme Lojik ---
            current_time = time.time()
            is_stable = stable_count >= STABILITY_THRESHOLD
            
            if is_stable and max_confidence >= CONFIDENCE_THRESHOLD:
                if stable_char is not None:
                    if stable_char != last_added_character:
                        if current_time - last_add_time >= SENTENCE_ADD_DELAY:
                            sentence_var.set(sentence_var.get() + stable_char)
                            last_added_character = stable_char
                            last_add_time = current_time
                    else:
                        last_add_time = current_time

            
            # --- Arayüz Görüntü Güncellemeleri ---
            
            # Kırpma kutusunu çiz
            color = (0, 255, 0) if is_stable and max_confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), color, 4)
            
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            # Tahmini karakteri yaz
            display_char = stable_char if stable_char and is_stable and stable_count >= STABILITY_THRESHOLD else (current_character if 'current_character' in locals() else "...")
            draw.text((crop_x1, crop_y1 - 80), display_char, font=font, fill=color)
            
            # Stabilite ve Güven bilgisini yaz
            info_text = f"Güven: {max_confidence*100:.1f}% | Stabilite: {stable_count}/{PREDICTION_HISTORY_SIZE}"
            draw.text((crop_x1, crop_y1 - 30), info_text, font=font_small, fill=(255, 255, 255))

            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    else:
        # El yoksa
        prediction_history.clear()
        char_var.set("El Bekleniyor...")
        stable_char = "El Bekleniyor..."
        
    char_var.set(stable_char if stable_char and stable_count >= STABILITY_THRESHOLD else (current_character if 'current_character' in locals() else "Bekleniyor..."))

    return frame

# =========================================================================
# Adım 4: Kullanıcı Arayüzü Fonksiyonları (CALLBACKS)
# Hata oluşmaması için buraya taşındı!
# =========================================================================

def clear_characters():
    """Cümleyi, anlık tahmini ve geçmişi temizler."""
    sentence_var.set("")
    char_var.set("Bekleniyor...")
    global last_added_character, last_add_time
    last_added_character = ""
    last_add_time = time.time()
    prediction_history.clear()

def delete_last_character():
    """Oluşturulan cümleden son karakteri siler."""
    current_sentence = sentence_var.get()
    if current_sentence:
        sentence_var.set(current_sentence[:-1])
    clear_characters()

def change_language_and_load(lang):
    """Dil değiştirme butonu tarafından çağrılır."""
    clear_characters()
    success = load_resources(lang)
    
    # Aktif butonu görsel olarak vurgula
    # Artık butonlar lang_frame içinde olduğu için root.grid_slaves'ı kullanamayız.
    # Bunun yerine Frame içindeki butonları bulmalıyız.
    for child in lang_frame.winfo_children():
        if isinstance(child, tk.Button):
            if child.cget('text').startswith(lang):
                child.config(relief=tk.SUNKEN, bd=3)
            else:
                child.config(relief=tk.RAISED, bd=1)
    
    if success:
        messagebox.showinfo("Başarılı", f"Model: {lang} ({MODEL_EPOCH}. epoch) başarıyla yüklendi.")
    else:
        messagebox.showerror("Hata", f"{lang} modeli veya sınıf haritası yüklenemedi. Konsolu kontrol edin.")


def update_video_feed():
    """Kamera akışını güncelleyen döngü."""
    ret, frame = cap.read()
    if ret:
        # Görüntüyü 4:3 oranında tutmak için 600x450 yapalım. (800x600 çok büyük)
        frame = cv2.resize(frame, (600, 450)) 
        frame = cv2.flip(frame, 1) 
        
        try:
            processed_frame = process_frame(frame)
        except Exception as e:
            print(f"process_frame sırasında hata: {e}")
            processed_frame = frame
        
        photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)))
        video_label.config(image=photo)
        video_label.photo = photo
    
    root.after(10, update_video_feed)


# =========================================================================
# Adım 5: Kullanıcı Arayüzü Başlatma (Widget Tanımlamaları)
# =========================================================================
root = tk.Tk()
root.title("Dil Seçimli İşaret Dili Tanıma")

# Değişkenler
char_var = tk.StringVar(value="Yükleniyor...")
sentence_var = tk.StringVar()

# --- Arayüz Öğeleri ---

# Tahmin Edilen Karakter
tk.Label(root, text="Tahmin Edilen Karakter:", font=('Arial', 16)).grid(row=0, column=0, padx=10, pady=10, sticky='w')
# Tahmin Entry'sinin 3 sütuna yayılmasını sağlayalım
tk.Entry(root, textvariable=char_var, font=('Arial', 30, 'bold'), state='readonly', width=5, justify='center', fg='green', relief=tk.SUNKEN).grid(row=0, column=1, columnspan=3, padx=10, pady=10, sticky='w')

# Oluşan Cümle
tk.Label(root, text="Oluşan Cümle:", font=('Arial', 16)).grid(row=1, column=0, padx=10, pady=10, sticky='w')
# Cümle entry'sinin 4 sütuna yayılmasını sağlayalım
tk.Entry(root, textvariable=sentence_var, font=('Arial', 18), state='readonly', width=40, relief=tk.FLAT, bd=2, bg='lightgray').grid(row=1, column=1, columnspan=4, padx=10, pady=10, sticky='ew')

# Video Geri Beslemesi (Şimdi 4 sütun kaplayacak: Col 0'dan Col 3'e)
video_label = tk.Label(root, bd=2, relief=tk.SUNKEN)
video_label.grid(row=2, column=0, columnspan=4, padx=10, pady=10, sticky='nsew')


# DİL SEÇİM BUTONLARINI İÇERECEK YENİ FRAME (Kameranın Sağına)
# 4. Sütun (Col 4) dil butonlarına ayrıldı.
lang_frame = tk.Frame(root, bd=2, relief=tk.RIDGE, bg='#F0F0F0')
lang_frame.grid(row=2, column=4, rowspan=2, padx=10, pady=10, sticky='nsew')
tk.Label(lang_frame, text="DİL SEÇİMİ", font=('Arial', 12, 'bold'), bg='#F0F0F0', anchor='center').grid(row=0, column=0, padx=5, pady=5, sticky='ew')

# Dil Seçimi Butonları (Yeni Frame içine taşındı)
tk.Button(lang_frame, text="AR (Arapça)", command=lambda: change_language_and_load('AR'), font=('Arial', 12, 'bold'), bg='#ADD8E6').grid(row=1, column=0, padx=5, pady=5, sticky='ew')
tk.Button(lang_frame, text="EN (İngilizce)", command=lambda: change_language_and_load('EN'), font=('Arial', 12), bg='#90EE90').grid(row=2, column=0, padx=5, pady=5, sticky='ew')
tk.Button(lang_frame, text="TR (Türkçe)", command=lambda: change_language_and_load('TR'), font=('Arial', 12), bg='#F08080').grid(row=3, column=0, padx=5, pady=5, sticky='ew')
tk.Button(lang_frame, text="Yeniden Başlat", command=clear_characters, font=('Arial', 12)).grid(row=4, column=0, padx=5, pady=5, sticky='ew')
# Dil Frame'i içindeki sütunun genişlemesini sağla
lang_frame.grid_columnconfigure(0, weight=1)


# Satır 3: İşlem Butonları (Video'nun hemen altında)
tk.Label(root, text="İşlemler:", font=('Arial', 12, 'bold')).grid(row=3, column=0, padx=10, pady=10, sticky='w')
# Cümleyi Temizle (Col 2 ve 3'ü kullanacak şekilde ayarlandı)
tk.Button(root, text="Cümleyi Temizle", command=lambda: sentence_var.set(""), font=('Arial', 12), bg='#FF6347', fg='white').grid(row=3, column=2, padx=5, pady=10, sticky='ew')
# Son Karakteri Sil (Col 4'ü kullanacak şekilde ayarlandı)
tk.Button(root, text="Son Karakteri Sil", command=delete_last_character, font=('Arial', 12), bg='#FFA500', fg='white').grid(row=3, column=3, padx=5, pady=10, sticky='ew')

# Sütun ağırlıklarını ayarlayalım
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)
root.grid_columnconfigure(3, weight=1)
root.grid_columnconfigure(4, weight=0) # Yan panelin genişlemesini engelle

# =========================================================================
# Uygulamayı Başlat
# =========================================================================
if __name__ == '__main__':
    # Başlangıçta ilk dili yükle
    load_resources(INITIAL_LANGUAGE)
    
    # Başlangıç butonunu görsel olarak vurgula
    # Artık butonlar lang_frame içinde olduğu için yeni yöntemi kullanmalıyız.
    try:
        if INITIAL_LANGUAGE == 'AR':
            lang_frame.grid_slaves(row=1)[0].config(relief=tk.SUNKEN, bd=3)
        elif INITIAL_LANGUAGE == 'EN':
            lang_frame.grid_slaves(row=2)[0].config(relief=tk.SUNKEN, bd=3)
        elif INITIAL_LANGUAGE == 'TR':
            lang_frame.grid_slaves(row=3)[0].config(relief=tk.SUNKEN, bd=3)
    except IndexError:
        pass

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Kamera Hatası", "Kamera açılamadı. Lütfen kamera bağlantısını kontrol edin.")
        exit()
        
    # Kamera çözünürlüğünü daha küçük bir değere ayarlayarak başlatma
    # 600x450'lik bir alanda daha iyi görünürlük sağlar.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    update_video_feed()
    root.mainloop()
    
    cap.release()
    cv2.destroyAllWindows()
