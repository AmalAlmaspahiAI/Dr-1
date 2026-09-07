import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# قراءة السمات من ملف Excel
df = pd.read_excel("extracted_features.xlsx")
X = df.drop(columns=["Image_Name", "Category", "Label"]).values
y = df["Label"].values

# تحسين SVM باستخدام PCA + GridSearch + Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("svm", SVC())
])

params = {
    "pca__n_components": [0.80, 0.90, 0.95, 0.99],
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.001, 0.01, 0.1],
    "svm__kernel": ["rbf", "linear"]
}

grid = GridSearchCV(
    pipeline,
    params,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X, y)

print("=== النموذج المحسن ===")
print("أفضل إعدادات:", grid.best_params_)
print(f"دقة Cross-Validation: {grid.best_score_:.2%}")

# اختبار مستقل بنفس تقسيم الدرس الأصلي للمقارنة العادلة
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

best_model = grid.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\n=== نتائج الاختبار المستقل ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.2%}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.2%}")
print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.2%}")
print(f"F1-Score : {f1_score(y_test, y_pred, zero_division=0):.2%}")

print("\nملاحظة: عدد الصور 28 فقط، لذلك دقة الاختبار المستقل تتغير كثيرًا.")
print("Cross-Validation يعطي تقييمًا أكثر استقرارًا من تقسيم واحد.")
