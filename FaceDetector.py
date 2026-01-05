import cv2
import pandas as pan
import numpy as np
import os

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from skimage.feature import local_binary_pattern

image_directories = {
    "train": "C:/Users/user/Downloads/BS/train",
    "test":  "C:/Users/user/Downloads/BS/test",
    "valid": "C:/Users/user/Downloads/BS/valid",
}

def preprocess_image(image_path):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(gray, (256, 256))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(resized)

def extract_lbp_histogram(image_path, num_points=8, radius=1):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    lbp = local_binary_pattern(image, num_points, radius, method='uniform')
    
    n_bins = num_points * (num_points - 1) + 3
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_hog_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    win_size = (256, 256)
    cell_size = (32, 32)
    block_size = (64, 64)
    block_stride = (32, 32)
    num_bins = 9
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    features = hog.compute(image)
    return features.flatten()

def all_images_preprocess(images_type, no_of_pictures):
    for i in range(1, no_of_pictures):
        try:
            img = preprocess_image(f"{image_directories[images_type]}/{i}.jpg")
            cv2.imwrite(f"{image_directories[images_type]}_processed/{i}.jpg", img)
        except:
            continue

def load_features_from_csv(csv_path):
    df = pan.read_csv(csv_path)
    X, y = [], []
    
    for _, row in df.iterrows():
        npy_path = row["image_path"]
        img_path = npy_path.replace("_feature_extracted/numpy_extracted", "_processed").replace(".npy", ".jpg")
        
        if not os.path.exists(img_path):
            continue
        
        try:
            lbp_hist = extract_lbp_histogram(img_path)
            hog_feat = extract_hog_features(img_path)
            features = np.concatenate([lbp_hist, hog_feat])
            
            X.append(features)
            y.append(row["label"])
        except:
            continue

    return np.array(X), np.array(y)

def train_svm(train_csv, test_csv):
    X_train, y_train = load_features_from_csv(train_csv)
    X_test, y_test = load_features_from_csv(test_csv)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm = SVC(
        kernel="rbf",
        C=100.0,
        gamma=0.001,
        probability=True,
        random_state=42,
        class_weight="balanced"
    )
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)

    print("\n ---REZULTATI---")
    print("Točnost:", accuracy_score(y_test, y_pred))
    return svm, scaler

def predict_from_image(img_path, svm, scaler):
    lbp_hist = extract_lbp_histogram(img_path)
    hog_feat = extract_hog_features(img_path)
    
    features = np.concatenate([lbp_hist, hog_feat])
    features = scaler.transform([features])

    pred = svm.predict(features)[0]
    #prob = svm.predict_proba(features)[0]

    #print("\n ---PREDIKCIJA---")
    #print(f"Rezultat - {img_path}:", "SPOOF" if pred == 1 else "REAL")
    #print("Vjerojatnosti [REAL, SPOOF]:", prob)
    return pred

def predict_all_validate_images(dir_path):
    svm, scaler = train_svm(
        f"{image_directories['train']}_feature_extracted/train_annotations.csv",
        f"{image_directories['test']}_feature_extracted/test_annotations.csv"
    )
    matches = []
    csv = pan.read_csv(f"{image_directories['valid']}_feature_extracted/valid_annotations.csv")
    for i in range(len(csv)):
        try:
            prediction = predict_from_image(f"{dir_path}/{i+1}.jpg", svm, scaler)    
            label = csv['label'].iloc[i]
            if(prediction == label):
                matches.append(1)
            else:
                matches.append(0)
        except Exception as e:
            print(f"Greška pri procesiranju slike {i+1}: {e}")
            continue
        
    return matches

def main():
    
    all_images_preprocess("train", 877)
    all_images_preprocess("test", 98)
    all_images_preprocess("valid", 135)

    matches = predict_all_validate_images(f"{image_directories['valid']}_processed")
    accuracy = sum(matches) / len(matches)
    print(f"\n Ukupna točnost na validaciji: {accuracy*100:.2f}%)")
    print(f"\n Točnih predviđanja: {sum(matches)}/{len(matches)}") 
    
if __name__ == "__main__":
    main()
