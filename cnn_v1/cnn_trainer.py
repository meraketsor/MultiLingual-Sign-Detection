import os
import time
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# TensorFlow ve Keras Modülleri
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- YAPILANDIRMA ---

# !!! KULLANICI TARAFINDAN GÜNCELLENMİŞ YOLLAR !!!
# Veri dosyalarınızın (X_EN.pkl, y_EN.pkl, label_map_EN.pkl vb.) bulunduğu ana dizin.
DATA_DIR = r'C:\Users\ridva\.conda\envs\vmamba_env\vmamba_proje\data_v1' 
# Eğitilmiş CNN modelinin ve etiket haritalarının kaydedileceği dizin.
MODEL_SAVE_DIR = r'C:\Users\ridva\.conda\envs\vmamba_env\vmamba_proje\cnn_v1' 

# Dil yapılandırmaları
language_configs = {
    'EN': {'feature_size': 42}, # Tek el için 42 özellik (21 landmark * 2 koordinat)
    'AR': {'feature_size': 42}, # Tek el için 42 özellik
    'TR': {'feature_size': 84}, # İki el için 84 özellik
}

# Eğitim parametreleri
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 50           
BATCH_SIZE = 64
VERBOSE = 1
DROPOUT_RATE = 0.3

# --- FONKSİYONLAR ---

def load_data(lang, feature_size):
    """
    Belirtilen dil için özellik vektörlerini ve etiketleri yükler,
    CNN için veriyi hazırlar ve eğitim/test setlerine böler.
    """
    print(f"\n--- {lang} Dili İçin Veri Yükleniyor ---")
    
    X_path = os.path.join(DATA_DIR, f'X_{lang}.pkl')
    y_path = os.path.join(DATA_DIR, f'y_{lang}.pkl')
    label_map_path = os.path.join(DATA_DIR, f'label_map_{lang}.pkl')
    
    if not os.path.exists(X_path) or not os.path.exists(y_path) or not os.path.exists(label_map_path):
        print(f"HATA: {lang} verileri veya etiket haritası bulunamadı. Yolu kontrol edin: {DATA_DIR}")
        return None, None, None, None, None, None, None

    try:
        # Veri Yükleme
        with open(X_path, 'rb') as f:
            X = pickle.load(f)
        with open(y_path, 'rb') as f:
            y = pickle.load(f)
        with open(label_map_path, 'rb') as f:
            labels_map = pickle.load(f)
            
        print(f"Yüklendi: X boyutu {X.shape}, y boyutu {y.shape}")
        
        # Özellik Boyut Kontrolü
        if X.shape[1] != feature_size:
            print(f"UYARI: Beklenen özellik boyutu {feature_size}, ancak yüklenen veri boyutu {X.shape[1]}. Devam ediliyor...")
            
        X = np.asarray(X, dtype=np.float32)
        
        # Eğitim ve Test olarak Ayırma
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        # CNN İçin Veriyi Yeniden Şekillendirme (Reshape)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Etiketleri Tek Sıcak (One-Hot) Kodlama
        num_classes = len(np.unique(y))
        y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
        y_test_one_hot = to_categorical(y_test, num_classes=num_classes)
        
        print(f"Hazır: Eğitim verisi {X_train.shape}, Test verisi {X_test.shape}")
        print(f"Toplam sınıf sayısı: {num_classes}")
        
        # Orijinal tamsayı etiketleri de rapor için gereklidir
        return X_train, X_test, y_train_one_hot, y_test_one_hot, y_test, num_classes, labels_map

    except Exception as e:
        print(f"Veri yüklenirken veya hazırlanırken hata: {e}")
        return None, None, None, None, None, None, None

def create_cnn_model(input_shape, num_classes):
    """
    1D Evrişimli Sinir Ağı (CNN) modelini tanımlar.
    """
    model = Sequential()
    
    # 1. Evrişim Katmanı
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(Dropout(DROPOUT_RATE))
    
    # 2. Evrişim Katmanı
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(DROPOUT_RATE))
    
    # 3. Evrişim Katmanı
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    # Düzleştirme (Flattening)
    model.add(Flatten())
    
    # Yoğun Katmanlar (Dense)
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    
    # Çıkış Katmanı
    model.add(Dense(num_classes, activation='softmax'))
    
    # Modeli derleme
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def save_model_and_labels(lang, model, labels_map):
    """Eğitilmiş modeli ve etiket haritasını kaydeder."""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    model_path = os.path.join(MODEL_SAVE_DIR, f'cnn_{lang}.h5')
    label_map_path = os.path.join(MODEL_SAVE_DIR, f'label_map_{lang}.pkl')
    
    # Modeli H5 formatında kaydetme
    model.save(model_path)
    
    # Etiket haritasını kaydetme
    with open(label_map_path, 'wb') as f:
        pickle.dump(labels_map, f)
        
    print(f"\nModel ve Etiketler başarıyla kaydedildi: {model_path}")

def train_and_evaluate_cnn():
    """Tüm diller için CNN modellerini eğitir ve değerlendirir."""
    
    # GPU/CPU ayarları
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU kullanılıyor.")
    else:
        print("CPU kullanılıyor.")
    
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    for lang, config in language_configs.items():
        feature_size = config['feature_size']
        
        # 1. Veriyi Yükle ve Hazırla
        X_train, X_test, y_train_one_hot, y_test_one_hot, y_test_labels_int, num_classes, labels_map = load_data(lang, feature_size)
        
        if X_train is None:
            continue
            
        # 2. Modeli Oluştur
        input_shape = (X_train.shape[1], 1)
        model = create_cnn_model(input_shape, num_classes)
        
        print(f"\n--- {lang} CNN Modeli Özeti ---")
        model.summary()

        # 3. Modeli Eğit ve Eğitim Süresini Ölç
        print(f"\n--- {lang} Modeli Eğitiliyor (Epochs: {EPOCHS}, Batch: {BATCH_SIZE}) ---")
        start_time = time.time()
        
        model.fit(
            X_train, y_train_one_hot,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test_one_hot),
            verbose=VERBOSE
        )
        
        end_time = time.time()
        training_duration = end_time - start_time
        print(f"\n*** Eğitim Tamamlandı. Süre: {training_duration:.2f} saniye ***")

        # 4. Modeli Kaydet
        save_model_and_labels(lang, model, labels_map)

        # 5. Değerlendirme
        print(f"\n--- {lang} Modeli Değerlendirme ve Raporlama ---")
        
        # Tahminleri al ve tamsayı etiketlerine çevir
        y_pred_one_hot = model.predict(X_test, verbose=0)
        y_pred_labels = np.argmax(y_pred_one_hot, axis=1)
        
        # Doğruluk
        accuracy = accuracy_score(y_test_labels_int, y_pred_labels)
        print(f"Test Doğruluğu (Accuracy): {accuracy:.4f}")
        
        # Sınıflandırma Raporu için etiket isimlerini hazırla
        id_to_label = {v: k for k, v in labels_map.items()}
        # target_names'in sıralı olmasını sağlamak için etiket ID'lerini sırala
        target_names = [id_to_label[i] for i in sorted(labels_map.values())]
        
        report = classification_report(
            y_test_labels_int, 
            y_pred_labels, 
            target_names=target_names, 
            zero_division=0, 
            digits=4 # Raporun daha hassas olması için
        )
        
        print("\nSınıflandırma Raporu (Classification Report):\n", report)
        
        # Raporu TXT dosyasına kaydetme
        report_file_path = os.path.join(MODEL_SAVE_DIR, f'cnn_report_{lang}.txt')
        with open(report_file_path, 'w', encoding='utf-8') as f:
            f.write(f"--- {lang} CNN Modeli Değerlendirme Raporu ---\n")
            f.write(f"Eğitim Süresi: {training_duration:.2f} saniye\n")
            f.write(f"Test Doğruluğu (Accuracy): {accuracy:.4f}\n")
            f.write(f"Test Örnek Sayısı: {len(y_test_labels_int)}\n\n")
            f.write(report)
            
        print(f"*** {lang} diline ait eğitim süresi ve sınıflandırma raporu {report_file_path} dosyasına kaydedildi. ***")
        print("-" * 70)


if __name__ == '__main__':
    train_and_evaluate_cnn()
