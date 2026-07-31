import os
import cv2
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_extraction import extract_features

def load_data(dataset_path):
    X, y = [], []
    for label in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, label)
        # Skip if it's not a directory
        if not os.path.isdir(class_path):
            continue
            
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            
            # Ensure the image was loaded correctly
            if img is not None:
                features = extract_features(img)
                X.append(features)
                y.append(label)
                
    return np.array(X), np.array(y)

def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True))
    ])

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Akurasi Model: {accuracy:.4f}")
    
    return model

def save_model(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model berhasil disimpan di: {save_path}")

if __name__ == "__main__":
    DATASET_PATH = r"c:\Users\LENOVO\OneDrive\Documents\rice-leaf-disease-web\dataset"
    MODEL_SAVE_PATH = "model/svm_model.pkl"

    print("Memuat dataset dan mengekstrak fitur, mohon tunggu...")
    X, y = load_data(DATASET_PATH)
    
    if len(X) == 0:
        print("Error: Tidak ada data gambar yang ditemukan atau berhasil diproses!")
    else:
        print(f"Total data diproses: {len(X)} gambar.")
        print("Sedang melatih model SVM...")
        trained_model = train_and_evaluate_model(X, y)
        
        save_model(trained_model, MODEL_SAVE_PATH)
