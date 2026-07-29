import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 1. Đường dẫn dataset
dataset_path = "source_code/Data"

if not os.path.exists(dataset_path):
    print("Không tìm thấy dataset:", dataset_path)
    exit()

# 2. Đọc dữ liệu ảnh
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path, target_size=(30, 30), batch_size=32,
    class_mode='sparse', subset='training', shuffle=False
)

valid_generator = datagen.flow_from_directory(
    dataset_path, target_size=(30, 30), batch_size=32,
    class_mode='sparse', subset='validation', shuffle=False
)

# 3. Hàm chuyển đổi dữ liệu từ Generator thành mảng phẳng (Flatten) cho SVM
def extract_features_and_labels(generator):
    features = []
    labels = []
    # Duyệt qua tất cả các batch trong generator
    for _ in range(len(generator)):
        X_batch, y_batch = next(generator)
        # Làm phẳng ảnh từ (30, 30, 3) thành (2700,)
        X_flatten = X_batch.reshape(X_batch.shape[0], -1)
        features.append(X_flatten)
        labels.append(y_batch)
    return np.vstack(features), np.concatenate(labels)

print("Đang trích xuất dữ liệu cho SVM...")
X_train, y_train = extract_features_and_labels(train_generator)
X_val, y_val = extract_features_and_labels(valid_generator)

# 4. Khởi tạo và huấn luyện mô hình SVM
print("Đang huấn luyện mô hình SVM ...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# 5. Đánh giá mô hình
y_pred = svm_model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"Độ chính xác trên tập Validation: {acc * 100:.2f}%")
print("\nBáo cáo chi tiết:")
print(classification_report(y_val, y_pred))

# 6. Lưu mô hình SVM (thay vì file .h5, SVM lưu bằng joblib hoặc pickle)
joblib.dump(svm_model, 'best_svm_model.pkl')
print("Đã lưu mô hình tại 'best_svm_model.pkl'")
