import cv2
import numpy as np
import joblib

# دالة استخراج السمات (نفس المستخدمة أثناء التدريب)
def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# 1. تحميل النموذج المدرب
model = joblib.load("cat_dog_model.pkl")

# 2. تحميل صورة جديدة للاختبار
# 
# test_image_path = "dataset/dogs/dog.4003.jpg"  # استبدل المسار بصورتك
# test_image_path = "dataset/cats/cat.4004.jpg"  # استبدل المسار بصورتك
test_image_path = "/home/abdullah/Documents/0aame/COURSE-AI/4-image processing books/1-PRACTICAL/Dr.1/dog2.jpg"  # استبدل المسار بصورتك

image = cv2.imread(test_image_path)

if image is None:
    print("لم يتم العثور على الصورة!")
else:
    # 3. معالجة الصورة بنفس خطى التدريب
    resized_img = cv2.resize(image, (128, 128))
    features = extract_color_histogram(resized_img).reshape(1, -1)

    # 4. التنبؤ
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    label_map = {0: "قطة (Cat)", 1: "كلب (Dog)"}
    result = label_map[prediction]
    confidence = probabilities[prediction] * 100

    # 5. عرض النتيجة على الصورة
    text = f"Prediction: {result} ({confidence:.1f}%)"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    print(f"النتيجة: {result} بنسبة ثقة {confidence:.2f}%")
    
    cv2.imshow("Test Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()