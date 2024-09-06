import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Завантаження всіх об'єктів із файлу
aussie_rain = joblib.load('models/aussie_rain.joblib')

model = aussie_rain['model']
imputer = aussie_rain['imputer']
scaler = aussie_rain['scaler']
encoder = aussie_rain['encoder']
input_cols = aussie_rain['input_cols']
target_col = aussie_rain['target_col']
numeric_cols = aussie_rain['numeric_cols']
categorical_cols = aussie_rain['categorical_cols']
encoded_cols = aussie_rain['encoded_cols']

# Завантаження даних для отримання можливих значень
data = pd.read_csv('data/weatherAUS.csv')  # Замість цього шляху вкажіть правильний шлях до вашого набору даних

# Отримання унікальних значень для категоріальних змінних
locations = data['Location'].unique()
wind_gust_dirs = data['WindGustDir'].unique()
wind_dirs_9am = data['WindDir9am'].unique()
wind_dirs_3pm = data['WindDir3pm'].unique()
rain_today_options = data['RainToday'].unique()

# Функція для прогнозування
def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    
    # Застосування препроцесингу
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    
    X_input = input_df[numeric_cols + list(encoded_cols)]
    
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    
    return pred, prob

# Заголовок застосунку
st.title('Прогнозування погоди: Чи буде дощ завтра?')
st.markdown('Цей додаток використовує модель машинного навчання для прогнозування ймовірності дощу на основі метеорологічних даних.')

# Введення характеристик для прогнозування
st.header("Введіть метеорологічні дані:")

col1, col2 = st.columns(2)

# Введення характеристик
with col1:
    date = st.date_input('Дата')
    location = st.selectbox('Локація', locations)
    min_temp = st.slider('Мінімальна температура (°C)', float(data['MinTemp'].min()), float(data['MinTemp'].max()), 15.0)
    max_temp = st.slider('Максимальна температура (°C)', float(data['MaxTemp'].min()), float(data['MaxTemp'].max()), 25.0)
    rainfall = st.slider('Кількість опадів (мм)', float(data['Rainfall'].min()), float(data['Rainfall'].max()), 10.0)
    evaporation = st.slider('Випаровування (мм)', float(data['Evaporation'].min()), float(data['Evaporation'].max()), 5.0)
    sunshine = st.slider('Сонячне світло (години)', float(data['Sunshine'].min()), float(data['Sunshine'].max()), 7.0)
    wind_gust_dir = st.selectbox('Напрям вітру при поривах', wind_gust_dirs)
    wind_gust_speed = st.slider('Швидкість пориву вітру (км/год)', float(data['WindGustSpeed'].min()), float(data['WindGustSpeed'].max()), 50.0)
    wind_dir_9am = st.selectbox('Напрям вітру о 9:00', wind_dirs_9am)
    wind_dir_3pm = st.selectbox('Напрям вітру о 15:00', wind_dirs_3pm)
    wind_speed_9am = st.slider('Швидкість вітру о 9:00 (км/год)', float(data['WindSpeed9am'].min()), float(data['WindSpeed9am'].max()), 15.0)
    wind_speed_3pm = st.slider('Швидкість вітру о 15:00 (км/год)', float(data['WindSpeed3pm'].min()), float(data['WindSpeed3pm'].max()), 20.0)

with col2:
    humidity_9am = st.slider('Вологість о 9:00 (%)', float(data['Humidity9am'].min()), float(data['Humidity9am'].max()), 80.0)
    humidity_3pm = st.slider('Вологість о 15:00 (%)', float(data['Humidity3pm'].min()), float(data['Humidity3pm'].max()), 50.0)
    pressure_9am = st.slider('Тиск о 9:00 (гПа)', float(data['Pressure9am'].min()), float(data['Pressure9am'].max()), 1010.0)
    pressure_3pm = st.slider('Тиск о 15:00 (гПа)', float(data['Pressure3pm'].min()), float(data['Pressure3pm'].max()), 1005.0)
    cloud_9am = st.slider('Хмарність о 9:00 (октави)', float(data['Cloud9am'].min()), float(data['Cloud9am'].max()), 4.0)
    cloud_3pm = st.slider('Хмарність о 15:00 (октави)', float(data['Cloud3pm'].min()), float(data['Cloud3pm'].max()), 4.0)
    temp_9am = st.slider('Температура о 9:00 (°C)', float(data['Temp9am'].min()), float(data['Temp9am'].max()), 20.0)
    temp_3pm = st.slider('Температура о 15:00 (°C)', float(data['Temp3pm'].min()), float(data['Temp3pm'].max()), 30.0)
    rain_today = st.selectbox('Чи був дощ сьогодні?', rain_today_options)

# Кнопка для прогнозування
if st.button("Прогнозувати"):
    new_input = {
        'Date': date.strftime('%Y-%m-%d'),
        'Location': location,
        'MinTemp': min_temp,
        'MaxTemp': max_temp,
        'Rainfall': rainfall,
        'Evaporation': evaporation,
        'Sunshine': sunshine,
        'WindGustDir': wind_gust_dir,
        'WindGustSpeed': wind_gust_speed,
        'WindDir9am': wind_dir_9am,
        'WindDir3pm': wind_dir_3pm,
        'WindSpeed9am': wind_speed_9am,
        'WindSpeed3pm': wind_speed_3pm,
        'Humidity9am': humidity_9am,
        'Humidity3pm': humidity_3pm,
        'Pressure9am': pressure_9am,
        'Pressure3pm': pressure_3pm,
        'Cloud9am': cloud_9am,
        'Cloud3pm': cloud_3pm,
        'Temp9am': temp_9am,
        'Temp3pm': temp_3pm,
        'RainToday': rain_today
    }
    
    pred, prob = predict_input(new_input)
    
    # Відображення результатів
    st.write(f"Прогноз: {'Так, буде дощ' if pred == 'Yes' else 'Ні, не буде дощу'}")
    st.write(f"Ймовірність: {prob:.2f}")
