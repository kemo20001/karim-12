import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# تحميل البيانات
data = pd.read_csv('bank1.csv', sep=';')

# التحقق من القيم في العمود y
print("القيم في العمود 'y' قبل المعالجة:")
print(data['y'].value_counts(dropna=False))

# معالجة القيم المفقودة في y
data = data.dropna(subset=['y'])  # إزالة الصفوف التي تحتوي على NaN في y
print(f"عدد الصفوف بعد إزالة القيم المفقودة في 'y': {len(data)}")

# التحقق من أن y يحتوي فقط على 0 و1
if not data['y'].isin([0, 1]).all():
    raise ValueError("العمود 'y' يحتوي على قيم غير متوقعة (غير 0 أو 1)")

# معالجة البيانات
# تحويل الأعمدة الفئوية إلى أرقام باستخدام One-Hot Encoding
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                      'contact', 'month', 'day_of_week', 'poutcome']
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# فصل الميزات والهدف
X = data_encoded.drop('y', axis=1)
y = data_encoded['y'].astype(int)  # التأكد من أن y من النوع int

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# تحجيم البيانات
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# معالجة عدم التوازن باستخدام SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# إعداد النموذج وضبط المعلمات
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200]
}
model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_balanced, y_train_balanced)

# الحصول على أفضل نموذج
best_model = grid_search.best_estimator_

# تقييم النموذج
y_pred = best_model.predict(X_test_scaled)
print("تقرير التصنيف:")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1]))
print("أفضل المعلمات:", grid_search.best_params_)

# حفظ النموذج والمحول وأسماء الميزات
joblib.dump(best_model, 'bank_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'model_features.pkl')