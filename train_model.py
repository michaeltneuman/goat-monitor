import os
import cv2
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import random
DATA_DIRS = {"goats": "goat_monitor/images/goats","no_goats": "goat_monitor/images/nogoats"}

def load_images():
    X, y = [], []
    for label, folder in DATA_DIRS.items():
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE).flatten()
            if img is None:
                continue
            X.append(img)
            y.append(1 if label == "goats" else 0)
    return np.array(X), np.array(y)

X, y = load_images()
iterations = 250
test_size = 0.1
random_state = random.randint(1,999999)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
clf = LinearSVC(max_iter=i)
clf.fit(X_train, y_train)
with open("goat_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)