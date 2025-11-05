import os
import time
import pickle
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

# --- YAPILANDIRMA ---

# !!! KRİTİK GÜNCELLEME !!!
# Resimlerin bulunduğu ANA KLASÖR ve .pkl dosyalarının kaydedileceği ÇIKTI KLASÖRÜ
# Sizin verdiğiniz bilgiye göre, hem resimler hem de .pkl'lar BU KLASÖRDE olacak:
MAIN_DIR = Path(r'C:\Users\ridva\.conda\envs\vmamba_env\vmamba_proje\data_v1') 
SOURCE_DIR = MAIN_DIR  # Resimlerin kaynağı (data_v1/EN, data_v1/AR, data_v1/TR)
OUTPUT_DIR = MAIN_DIR  # .pkl dosyalarının hedefi

# İşlenecek diller ve beklenen özellik boyutu (Bu, cnn_trainer.py ile aynı olmalıdır)
language_configs = {
    'EN': {'feature_size': 42, 'max_hands': 1}, # 21 landmark * 2 koordinat
    'AR': {'feature_size': 42, 'max_hands': 1}, # 21 landmark * 2 koordinat
    'TR': {'feature_size': 84, 'max_hands': 2}, # 42 landmark * 2 koordinat (İki el)
}

# Kabul edilecek resim uzantıları
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')

# MediaPipe Hands kurulumu
mp_hands = mp.solutions.hands

# --- FONKSİYONLAR ---

def extract_landmarks(image_path: Path, hands_processor: mp_hands.Hands, max_hands: int, feature_size: int) -> list:
    """
    Belirtilen resimden el landmark'larını çıkarır.
    feature_size'a göre tek veya çift el verisi döndürür.
    """
    # OpenCV ile resmi yükle
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    # Resmin renk uzayını BGR'dan RGB'ye çevir (MediaPipe beklentisi)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Landmark tespiti
    results = hands_processor.process(image_rgb)
    
    landmarks_flat = []
    
    if results.multi_hand_landmarks:
        
        detected_hands = results.multi_hand_landmarks[:max_hands]
        
        for hand_landmarks in detected_hands:
            for landmark in hand_landmarks.landmark:
                # Koordinatları normalize edilmiş (0-1) değerler olarak ekle
                landmarks_flat.append(landmark.x)
                landmarks_flat.append(landmark.y)
        
        # Özellik listesini feature_size'a uyacak şekilde doldur (pad)
        if len(landmarks_flat) > feature_size:
            landmarks_flat = landmarks_flat[:feature_size]
        elif len(landmarks_flat) < feature_size:
            # Eksik veriyi sıfırlarla doldur (padding)
            landmarks_flat.extend([0.0] * (feature_size - len(landmarks_flat)))

    else:
        # El tespit edilemezse tüm özellikler için sıfır (padding)
        landmarks_flat.extend([0.0] * feature_size)

    return landmarks_flat

def process_and_save_dataset():
    """
    Tüm diller için resimleri işler, landmarkları çıkarır ve .pkl dosyalarını kaydeder.
    """
    if not SOURCE_DIR.exists():
        # Kaynak klasör bulunamazsa daha agresif bir hata mesajı
        print("\n" + "="*80)
        print(f"KRİTİK HATA: Ana Resim Klasörü ({SOURCE_DIR.resolve()}) BULUNAMADI!")
        print("Lütfen SOURCE_DIR değişkenindeki **TAM YOLU** kontrol edin ve klasörün var olduğundan emin olun.")
        print("="*80 + "\n")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Özellikler Buraya Kaydedilecek: {OUTPUT_DIR.resolve()}")
    print("-" * 70)

    for lang, config in language_configs.items():
        # Resim klasörü artık doğrudan SOURCE_DIR/EN şeklinde
        lang_source_dir = SOURCE_DIR / lang
        feature_size = config['feature_size']
        max_hands = config['max_hands']
        
        print(f"--- {lang} Dili İçin Özellik Çıkarılıyor (Max El: {max_hands}, Feature Boyutu: {feature_size}) ---")

        if not lang_source_dir.exists():
            # Alt klasör bulunamazsa uyarı, ancak işlem devam eder
            print(f"UYARI: {lang} diline ait resim klasörü ({lang_source_dir.resolve()}) bulunamadı. Atlanıyor.")
            print(f"Lütfen '{SOURCE_DIR.name}' klasörü içinde **'{lang}'** alt klasörünün ve resimlerinin olduğundan emin olun.")
            continue

        # MediaPipe işlemcisini oluştur (Her dil için farklı max_hands ayarı olabilir)
        with mp_hands.Hands(
            static_image_mode=True, 
            max_num_hands=max_hands,
            min_detection_confidence=0.5
        ) as hands:
            
            X_data = [] # Özellik vektörleri
            y_data = [] # Etiketler (tamsayı)
            labels_map = {} # Etiket ismi -> Etiket tamsayısı eşleştirmesi
            current_label_id = 0
            
            # Alt klasörleri (harf/işaret isimlerini) tara (Örn: data_v1/EN/A)
            class_labels = sorted([d.name for d in lang_source_dir.iterdir() if d.is_dir()])
            
            total_images = sum([len(list((lang_source_dir / label).glob(f'**/*{ext}'))) 
                                for label in class_labels for ext in IMAGE_EXTENSIONS])
            
            if total_images == 0:
                print(f"UYARI: {lang} diline ait klasörde (Örn: {lang_source_dir.name}/A) hiç resim bulunamadı. Atlanıyor.")
                continue

            with tqdm(total=total_images, desc=f"Processing {lang}") as pbar:
                for class_name in class_labels:
                    class_path = lang_source_dir / class_name
                    
                    if class_name not in labels_map:
                        labels_map[class_name] = current_label_id
                        current_label_id += 1
                        
                    label_id = labels_map[class_name]

                    # Tüm resimleri al
                    for ext in IMAGE_EXTENSIONS:
                        for image_path in class_path.glob(f'**/*{ext}'):
                            landmarks = extract_landmarks(image_path, hands, max_hands, feature_size)
                            
                            if landmarks and len(landmarks) == feature_size:
                                X_data.append(landmarks)
                                y_data.append(label_id)
                            
                            pbar.update(1)

            X_data = np.array(X_data, dtype=np.float32)
            y_data = np.array(y_data, dtype=np.int32)

            # Veri kaydı (data_v1 klasörüne)
            X_path = OUTPUT_DIR / f'X_{lang}.pkl'
            y_path = OUTPUT_DIR / f'y_{lang}.pkl'
            label_map_path = OUTPUT_DIR / f'label_map_{lang}.pkl'

            with open(X_path, 'wb') as f:
                pickle.dump(X_data, f)
            with open(y_path, 'wb') as f:
                pickle.dump(y_data, f)
            with open(label_map_path, 'wb') as f:
                pickle.dump(labels_map, f)
                
            print(f"\nBaşarılı: {lang} verisi kaydedildi.")
            print(f"   -> X_{lang}.pkl: {X_data.shape} (Özellik Vektörleri)")
            print(f"   -> y_{lang}.pkl: {y_data.shape} (Etiketler)")
            print("-" * 70)


if __name__ == '__main__':
    # İşleme başlamadan önce gerekli kütüphanelerin kontrolü
    try:
        import mediapipe as mp
    except ImportError:
        print("HATA: MediaPipe kütüphanesi yüklü değil.")
        print("Lütfen 'pip install mediapipe' komutunu çalıştırın.")
        exit()
        
    process_and_save_dataset()
