import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from pathlib import Path
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# --- YAPILANDIRMA ---

# Veri setinizin ana dizini
DATASET_DIR = Path(r'C:\Users\ridva\.conda\envs\vmamba_env\vmamba_proje\data_v1')

# Model ve çıktıların kaydedileceği dizin
MODEL_DIR = Path(r'C:\Users\ridva\.conda\envs\vmamba_env\vmamba_proje\decision_tree_v1')
MODEL_DIR.mkdir(parents=True, exist_ok=True) # Klasör yoksa oluştur

# Değerlendirme raporunun kaydedileceği dosya yolu
EVAL_FILE = MODEL_DIR / 'decision_tree_evaluation.txt'

# İşlenecek diller ve beklenen özellik boyutları
LANGUAGES = ['AR', 'EN', 'TR']

# YENİ: Her dil için beklenen özellik boyutu
# EN/AR: Tek el (42), TR: İki el (84)
FEATURE_SIZES = {
    'EN': 42,
    'AR': 42,
    'TR': 84, 
}

# Mediapipe'ı başlat
mp_hands = mp.solutions.hands
# İki el işaretlerini algılamak için max_num_hands=2
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=2)

# --- YARDIMCI FONKSİYONLAR ---

def format_duration(seconds):
    """Saniyeyi saat, dakika ve saniye formatına dönüştürür."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h} saat, {m} dakika, {s:.2f} saniye"

# --- ÖZELLİK ÇIKARMA ---

def extract_features_from_images(language_code):
    """
    Belirtilen dil klasöründeki tüm resimleri işler, el landmarklarını çıkarır
    ve normalleştirilmiş veriyi **84 boyutuna tamamlayarak** döndürür.
    
    Not: Tüm diller için 84 boyutlu veri toplanır. TR hariç diller için 
    eğitimden hemen önce 42 boyuta düşürülecektir.
    """
    print(f"\n[{language_code}] Resimlerden özellik çıkarma işlemi başlıyor...")
    
    data = []
    labels_list = []
    
    language_dir = DATASET_DIR / language_code
    
    if not language_dir.is_dir():
        print(f"Hata: {language_dir} klasörü bulunamadı. Atlanıyor.")
        return None, None
        
    class_folders = sorted([d for d in language_dir.iterdir() if d.is_dir()])
    
    if not class_folders:
        print(f"Hata: {language_dir} içinde sınıf (harf) klasörleri bulunamadı. Atlanıyor.")
        return None, None
    
    for class_folder in tqdm(class_folders, desc=f"-> {language_code} Sınıfları"):
        class_name = class_folder.name
        
        image_files = [f for f in class_folder.glob('*.jpg')]
        
        for image_path in image_files:
            try:
                frame = cv2.imread(str(image_path))
                if frame is None:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    data_aux = [] 
                    
                    # MediaPipe'ın bulduğu elleri (max 2) işle
                    for hand_landmarks in results.multi_hand_landmarks:
                        x_coords = [lm.x for lm in hand_landmarks.landmark]
                        y_coords = [lm.y for lm in hand_landmarks.landmark]
                        
                        # Normalleştirme: En küçük x ve y değerlerini bul
                        min_x, min_y = min(x_coords), min(y_coords)
                        
                        # Bu elin 42 özelliğini (21 landmark * 2 koordinat) data_aux'a ekle
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            
                            # Göreli konumu kaydet
                            data_aux.append(x - min_x)
                            data_aux.append(y - min_y)
                            
                    # Vektörün ilk 84 elementini al (2 elin verisini)
                    final_data_vector = data_aux[:84] 
                    
                    # Eğer özellik sayısı 84'ten az ise (yani 1 el bulunduysa: 42 özellik)
                    if len(final_data_vector) < 84:
                        padding_needed = 84 - len(final_data_vector)
                        final_data_vector.extend([0.0] * padding_needed) # Sıfırlarla doldur
                    
                    # Sadece 84 elemanlı vektörleri ekle
                    data.append(final_data_vector)
                    labels_list.append(class_name) 
                    
            except Exception as e:
                print(f"Hata: {image_path} dosyası işlenemedi. Sebep: {e}")

    data_np = np.asarray(data)
    labels_np = np.asarray(labels_list)
    
    print(f"[{language_code}] Veri toplama tamamlandı. Toplam {len(data_np)} örnek toplandı.")
    return data_np, labels_np

# --- EĞİTİM VE DEĞERLENDİRME ---

def train_and_evaluate_decision_tree(language_code, data, labels):
    """
    Decision Tree modelini eğitir, değerlendirir ve sonuçları kaydeder.
    """
    if data is None or labels is None or len(data) == 0:
        print(f"[{language_code}] Eğitim için yeterli veri bulunamadı. Atlanıyor.")
        return
        
    print(f"[{language_code}] Decision Tree modeli eğitiliyor...")
    start_time = time.time() # Eğitim başlangıç zamanı

    # 1. Özellik Boyutu Ayarlama (Kullanıcı Kuralına Göre)
    expected_size = FEATURE_SIZES.get(language_code, 84) # Varsayılan 84
    
    # Eğer beklenen boyut 84'den küçükse (EN/AR), veriyi 42 boyuta kes
    if expected_size < 84:
        # data, 84 boyutlu numpy array'dir. Sadece ilk 'expected_size' kadar sütunu al.
        data_prepared = data[:, :expected_size]
        print(f"[{language_code}] Tek el modeli için veri boyutu {data.shape[1]}'den {expected_size}'e düşürüldü.")
    else:
        # TR için 84 boyutlu veri olduğu gibi korunur.
        data_prepared = data
        print(f"[{language_code}] İki el modeli için veri boyutu {expected_size} olarak korundu.")
        
    # 2. Etiketleri sayısal değerlere dönüştür
    le = LabelEncoder()
    numeric_labels = le.fit_transform(labels)
    
    # Label Map (Sayısal Etiket -> Gerçek Harf) oluştur ve kaydet
    label_map = {int(i): label for i, label in enumerate(le.classes_)}
    with open(MODEL_DIR / f'label_map_{language_code}.pkl', 'wb') as f:
        pickle.dump(label_map, f)
    
    # 3. Veriyi eğitim ve test setlerine ayır
    x_train, x_test, y_train, y_test = train_test_split(
        data_prepared, # Hazırlanmış (kesilmiş veya tam) veri kullanılıyor
        numeric_labels, 
        test_size=0.2, 
        shuffle=True, 
        stratify=numeric_labels,
        random_state=42
    )
    
    # 4. Decision Tree modelini oluştur ve eğit
    model = DecisionTreeClassifier(random_state=42)
    model.fit(x_train, y_train)
    
    end_time = time.time() # Eğitim bitiş zamanı
    duration_seconds = end_time - start_time
    formatted_duration = format_duration(duration_seconds)

    # 5. Tahmin yap
    y_predict = model.predict(x_test)
    
    # 6. Doğruluk ve Sınıflandırma Raporu hesapla
    score = accuracy_score(y_test, y_predict)
    
    report = classification_report(
        y_test, 
        y_predict, 
        target_names=le.classes_, 
        output_dict=False,
        zero_division=0
    )
    
    # Sonuçları ekrana yazdır
    print(f"[{language_code}] Model Doğruluk Skoru: {score*100:.2f}% (Özellik Boyutu: {data_prepared.shape[1]})")
    print(f"[{language_code}] Eğitim Süresi: {formatted_duration}")
    print("-" * 50)
    print(report)
    print("-" * 50)
    
    # 7. Sonuçları dosyaya yaz
    evaluation_content = f"--- {language_code} İşaret Dili Modeli Değerlendirme Raporu ---\n"
    evaluation_content += f"Model Dosyası: decision_tree_{language_code}.pkl\n"
    evaluation_content += f"Özellik Boyutu: {data_prepared.shape[1]}\n" # Kullanılan boyutu rapora ekle
    evaluation_content += f"Doğruluk (Accuracy): {score:.4f}\n"
    evaluation_content += f"Eğitim Süresi: {formatted_duration}\n"
    evaluation_content += f"Eğitim Örneği Sayısı: {len(x_train)}\n"
    evaluation_content += f"Test Örneği Sayısı: {len(x_test)}\n"
    evaluation_content += "\n"
    evaluation_content += report
    evaluation_content += "\n" * 2
    
    # Mevcut rapora ekle (append modu)
    with open(EVAL_FILE, 'a', encoding='utf-8') as f:
        f.write(evaluation_content)

    # 8. Modeli .pkl dosyası olarak kaydet
    model_filename = MODEL_DIR / f'decision_tree_{language_code}.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"[{language_code}] Model başarıyla kaydedildi: {model_filename}")
    print(f"[{language_code}] Rapor başarıyla '{EVAL_FILE}' dosyasına eklendi.")
    
    return duration_seconds


# --- KODU ÇALIŞTIRMA ---
if __name__ == '__main__':
    # Rapor dosyasını temizle (her çalıştırmada yeni rapor başlasın)
    if EVAL_FILE.exists():
        os.remove(EVAL_FILE)
        print(f"{EVAL_FILE.name} dosyası temizlendi.")
        
    # Mediapipe Hands nesnesini başlat
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=2) 
    
    total_training_duration = 0
    total_start_time = time.time() # Toplam işlem başlangıç zamanı

    for lang in LANGUAGES:
        # 1. Özellik Çıkarma
        data, labels = extract_features_from_images(lang)
        
        # 2. Eğitim ve Değerlendirme (Bu kısım dil kuralını uyguluyor)
        duration = train_and_evaluate_decision_tree(lang, data, labels)
        if duration is not None:
            total_training_duration += duration

    # Toplam Süre Hesaplama
    total_end_time = time.time()
    total_script_duration = total_end_time - total_start_time # Özellik çıkarma dahil toplam süre
    
    formatted_total_training = format_duration(total_training_duration)
    formatted_total_script = format_duration(total_script_duration)
    
    # Toplam Süre Sonuçlarını Ekrana ve Dosyaya Yazma
    
    total_summary_content = f"\n\n--- GENEL EĞİTİM ÖZETİ ---\n"
    total_summary_content += f"Sadece Model Eğitim Süreleri Toplamı: {formatted_total_training}\n"
    total_summary_content += f"Script Toplam Çalışma Süresi (Özellik Çıkarma Dahil): {formatted_total_script}\n"
    total_summary_content += "---------------------------\n"
    
    print(total_summary_content)
    
    with open(EVAL_FILE, 'a', encoding='utf-8') as f:
        f.write(total_summary_content)

    # Sonuç
    print("\n\n--- TÜM İŞARET DİLİ MODELLERİNİN (DECISION TREE) EĞİTİMİ TAMAMLANDI ---")
