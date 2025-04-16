from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# تحميل النموذج والمحول وأسماء الميزات
model = joblib.load('bank_model.pkl')
scaler = joblib.load('scaler.pkl')
model_features = joblib.load('model_features.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # جمع البيانات من النموذج
        input_data = {
            'age': int(request.form['age']),
            'job': request.form['job'],
            'marital': request.form['marital'],
            'education': request.form['education'],
            'default': request.form['default'],
            'housing': request.form['housing'],
            'loan': request.form['loan'],
            'contact': request.form['contact'],
            'month': request.form['month'],
            'day_of_week': request.form['day_of_week'],
            'duration': int(request.form['duration']),
            'campaign': int(request.form['campaign']),
            'pdays': int(request.form['pdays']),
            'previous': int(request.form['previous']),
            'poutcome': request.form['poutcome'],
            'emp.var.rate': float(request.form['emp.var.rate']),
            'cons.price.idx': float(request.form['cons.price.idx']),
            'cons.conf.idx': float(request.form['cons.conf.idx']),
            'euribor3m': float(request.form['euribor3m']),
            'nr.employed': float(request.form['nr.employed'])
        }

        # التحقق من صحة المدخلات
        if input_data['age'] < 18 or input_data['age'] > 100:
            return render_template('index.html', prediction_text='العمر يجب أن يكون بين 18 و100 سنة')
        if input_data['duration'] < 0:
            return render_template('index.html', prediction_text='مدة المكالمة يجب أن تكون موجبة')
        if input_data['campaign'] < 1:
            return render_template('index.html', prediction_text='عدد المكالمات يجب أن يكون 1 أو أكثر')

        # تحويل البيانات إلى DataFrame
        input_df = pd.DataFrame([input_data])

        # تطبيق One-Hot Encoding
        categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 
                              'loan', 'contact', 'month', 'day_of_week', 'poutcome']
        input_encoded = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)

        # إضافة الأعمدة الناقصة وترتيبها
        for col in model_features:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_features]

        # تحجيم البيانات
        input_scaled = scaler.transform(input_encoded)

        # التنبؤ
        prediction = model.predict(input_scaled)[0]
        prediction_text = 'نعم' if prediction == 1 else 'لا'

        return render_template('index.html', prediction_text=f'هل سيشترك العميل؟ {prediction_text}')

    except ValueError:
        return render_template('index.html', prediction_text='خطأ: يرجى إدخال قيم صحيحة (أرقام في الحقول العددية)')
    except Exception as e:
        return render_template('index.html', prediction_text=f'خطأ غير متوقع: {str(e)}')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)