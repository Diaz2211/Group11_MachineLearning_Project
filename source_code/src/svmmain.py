import cv2
import numpy as np
import joblib  # Thư viện để load mô hình SVM

classes = { 0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)', 9:'Cam vuot', 10:'Cam do xe', 11:'Giao duong khong uu tien', 12:'Cam di thang',
            13:'Giao duong uu tien', 14:'Stop', 15:'Duong cam', 16:'Toi da 3,5t', 17:'Cam di nguoc chieu',
            18:'Nguy hiem khac', 19:'Toi da 1,5t', 20:'5km/h', 21:'Duong gap khuc', 22:'Vong xuyen',
            23:'Duong tron truot', 24:'Duong 2 chieu hep phai', 25:'Cong truong', 26:'Cam xe khach va tai',
            27:'Nguoi di bo', 28:'Truong hoc', 29:'Nguoi di xe dap', 30:'Dien cao the', 31:'Re phai',
            32:'Re trai', 33:'Di thang', 34:'Di thang hoac re phai', 35:'Di thang hoac re trai',
            36:'Di ben phai', 37:'Di ben trai'
}

svm_model = joblib.load('source_code/src/model/best_svm_model.pkl')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("Đang chạy Webcam với SVM... Nhấn 'q' để thoát.")
x_start, y_start, x_end, y_end = 220, 140, 420, 340

while True:
    success, img_original = cap.read()
    if not success: break

    cv2.rectangle(img_original, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
    img_roi = img_original[y_start:y_end, x_start:x_end]

    # Tiền xử lý ảnh giống lúc train SVM
    img_rgb = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (30, 30))
    img_normalized = img_resized.astype('float32') / 255
    
    # LÀM PHẲNG ẢNH
    img_input = img_normalized.flatten().reshape(1, -1)

    # Dự đoán bằng SVM
    class_index = svm_model.predict(img_input)[0]
    
    txt = f"{classes.get(class_index, 'Unknown')}"
    color = (0, 255, 0)

    cv2.putText(img_original, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Nhan dien bien bao (SVM)", img_original)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
