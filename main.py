import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():

    data_path = "data.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Hata: {data_path} bulunamadı.")
        return
    
    # Bazı Breast Cancer veri setlerinde sonda virgül olduğu için "Unnamed: 32" isimli boş bir sütun oluşabiliyor, onu temizleyelim
    empty_cols = [col for col in df.columns if 'Unnamed' in col]
    if empty_cols:
        df = df.drop(empty_cols, axis=1)

    # Özellikler (X) ve hedef değişkeni (y) ayır
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    # 'M' (Malignant/Kötü Huylu) sınıfını 1, 'B' (Benign/İyi Huylu) sınıfını 0 olarak kodla (Binary classification)
    y = y.map({'M': 1, 'B': 0})

    # Veriyi Eğitim (%80) ve Test (%20) olacak şekilde böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Özellikleri ölçeklendir (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modeli oluştur ve eğit (Random Forest Classifier kullanıldı)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Test verisi üzerinde tahmin yap
    y_pred = model.predict(X_test_scaled)

    # Performans metriklerini hesapla
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("-" * 50)
    print("Makine Öğrenmesi Sınıflandırma Sonuçları (Random Forest)")
    print("-" * 50)
    print(f"Accuracy (Doğruluk) : {accuracy:.4f}")
    print(f"Precision (Kesinlik): {precision:.4f}")
    print(f"Recall (Duyarlılık) : {recall:.4f}")
    print(f"F1-Score            : {f1:.4f}")
    print("-" * 50)
    print("Karmaşıklık Matrisi (Confusion Matrix):")
    print(conf_matrix)
    print("  [TN, FP]")
    print("  [FN, TP]")
    print("-" * 50)

if __name__ == "__main__":
    main()
