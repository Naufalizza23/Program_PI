import os
import cv2
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_extraction import extract_features

DATASET_PATH = "dataset/Citra_Daun_Padi"

X, y = [], []

for label in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, label)
    for img_name in os.listdir(class_path):
        img = cv2.imread(os.path.join(class_path, img_name))
        X.append(extract_features(img))
        y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", probability=True))
])

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Akurasi Model:", accuracy)

os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/svm_model.pkl", "wb"))
