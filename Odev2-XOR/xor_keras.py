import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 1. XOR Veriseti (Girdi ve Çıktılar)
# X: Giriş özellikleri (0 ve 1 kombinasyonları)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
# y: Beklenen çıktılar (XOR mantığı: Girdiler farklıysa 1, aynıysa 0)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# 2. Çok Katmanlı Sinir Ağı (MLP) Modeli Oluşturma
model = Sequential()

# Gizli Katman (Hidden Layer): 4 nöron ve ReLU aktivasyon fonksiyonu kullanıyoruz.
# ReLU (veya Sigmoid/Tanh) doğrusal olmayan problemleri (XOR gibi) çözmeyi sağlayan temel unsurdur.
# Tek katmanlı (sadece giriş ve çıkış) bir ağ bu problemi çözemezdi.
model.add(Dense(4, input_dim=2, activation='relu', name='Gizli_Katman'))

# Çıktı Katmanı (Output Layer): İkili sınıflandırma (0 veya 1) olduğu için 1 nöron ve Sigmoid kullanıyoruz.
# Sigmoid çıktıyı 0 ile 1 arasına sıkıştırarak olasılık benzeri bir değer üretir.
model.add(Dense(1, activation='sigmoid', name='Cikti_Katmani'))

# 3. Modeli Derleme (Compile)
# İkili (binary) bir problem olduğu için binary_crossentropy kayıp fonksiyonu (loss) kullanılır.
# Adam optimizasyon algoritması, ağırlıkları güncellemek için en popüler ve etkili yöntemlerden biridir.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modelin mimari yapısını ekrana yazdıralım
print("\n--- Modelin Mimari Yapısı ---")
model.summary()

# 4. Modeli Eğitme (Training)
print("\nModel eğitiliyor, lütfen bekleyin...")
# Ağın veri setini defalarca (epochs) görmesi ve ağırlıkları güncellemesi sağlanır.
# XOR basit bir problem olsa da ağın öğrenmesi için genellikle fazla epoch gerekir.
history = model.fit(X, y, epochs=2000, verbose=0) 

# 5. Sonuçları Değerlendirme ve Tahminler
print("\n--- Eğitim Tamamlandı. Tahmin Sonuçları ---")
predictions = model.predict(X, verbose=0)
for i in range(len(X)):
    gercek_deger = y[i][0]
    tahmin_edilen = predictions[i][0]
    yuvarlanmis_tahmin = round(tahmin_edilen)
    print(f"Girdi: {X[i]} -> Beklenen Çıktı: {gercek_deger} | Ağın Tahmini: {tahmin_edilen:.4f} (Yuvarlanmış: {yuvarlanmis_tahmin})")

# 6. Eğitim Sürecini Görselleştirme (Kayıp ve Başarı)
plt.figure(figsize=(12, 5))

# Kayıp (Loss) Grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Kayıp (Loss)', color='red')
plt.title('Eğitim Sürecinde Kayıp (Loss)')
plt.xlabel('Epoch (Eğitim Adımı)')
plt.ylabel('Kayıp Değeri')
plt.grid(True)
plt.legend()

# Doğruluk (Accuracy) Grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Doğruluk (Accuracy)', color='blue')
plt.title('Eğitim Sürecinde Doğruluk (Accuracy)')
plt.xlabel('Epoch (Eğitim Adımı)')
plt.ylabel('Doğruluk Oranı')
plt.grid(True)
plt.legend()

plt.tight_layout()
grafik_dosyasi = 'xor_egitim_grafikleri.png'
plt.savefig(grafik_dosyasi)
print(f"\nÖğrenme sürecinin grafikleri '{grafik_dosyasi}' adıyla bulunduğunuz dizine kaydedildi.")
