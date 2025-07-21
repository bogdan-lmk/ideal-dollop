#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from RSIPredictor import transform_rsi_features
from RSIPredictionTest import load_rsi_model

def show_prediction_dates():
    """
    Показывает для каких дат делаются прогнозы RSI
    """
    print("🔍 АНАЛИЗ ДАТ ПРОГНОЗОВ RSI")
    print("=" * 50)
    
    # 1. Загружаем исходные данные
    print("📊 1. Загружаем исходный dataset...")
    raw_df = pd.read_csv("accumulatedData_2024.csv", parse_dates=["open_time"])
    
    print(f"   • Всего записей в файле: {len(raw_df)}")
    print(f"   • Период данных: {raw_df['open_time'].min()} - {raw_df['open_time'].max()}")
    
    # 2. Обрабатываем данные для ML
    print("\n🔧 2. Обрабатываем данные для машинного обучения...")
    processed_df = transform_rsi_features(raw_df)
    
    print(f"   • Записей после обработки: {len(processed_df)}")
    if len(processed_df) > 0:
        # Получаем индексы обработанных данных
        processed_indices = processed_df.index
        processed_dates = raw_df.iloc[processed_indices]['open_time']
        
        print(f"   • Период обработанных данных: {processed_dates.min()} - {processed_dates.max()}")
    
    # 3. Показываем логику прогнозирования
    print("\n🎯 3. Логика прогнозирования:")
    print("   • Для даты X мы предсказываем RSI на дату X+1")
    print("   • Например: данные за 15.05.2022 → прогноз RSI на 16.05.2022")
    
    # 4. Загружаем модель и делаем прогнозы
    print("\n🤖 4. Делаем прогнозы...")
    model, feature_info = load_rsi_model()
    
    if model is None:
        print("   ❌ Модель не найдена. Сначала запустите: python RSIPredictor.py")
        return
    
    # Берем последние 10 записей для примера
    if len(processed_df) >= 10:
        latest_data = processed_df.tail(10)
        
        print(f"\n📅 5. ПОСЛЕДНИЕ 10 ПРОГНОЗОВ:")
        print("-" * 60)
        print("Дата данных       | Текущий RSI | Прогноз RSI (след. день)")
        print("-" * 60)
        
        # Подготавливаем признаки
        features = latest_data[feature_info['important_features']]
        features.columns = [f"f{i}" for i in range(len(feature_info['important_features']))]
        predictions = model.predict(features.astype('float32'))
        
        for i, (idx, row) in enumerate(latest_data.iterrows()):
            data_date_str = str(raw_df.iloc[idx]['open_time'])[:10]  # Получаем строку даты
            current_rsi = row['rsi_current']
            predicted_rsi = predictions[i]
            
            # Парсим дату и добавляем день
            try:
                data_date = pd.to_datetime(data_date_str)
                next_date = data_date + pd.Timedelta(days=1)
                next_date_str = next_date.strftime('%d.%m.%Y')
            except:
                next_date_str = "след. день"
            
            print(f"{data_date_str.replace('-', '.')}      | {current_rsi:11.2f} | {predicted_rsi:11.2f} (на {next_date_str})")
    
    # 6. Объясняем временные рамки
    print(f"\n⏰ 6. ВРЕМЕННЫЕ РАМКИ:")
    print(f"   • Исходные данные: {len(raw_df)} дней")
    print(f"   • Данные для ML: {len(processed_df)} дней (потери из-за технических индикаторов)")
    print(f"   • Потери данных происходят из-за:")
    print(f"     - Расчета скользящих средних (нужна история)")
    print(f"     - RSI расчета (нужно минимум 14 дней)")
    print(f"     - MACD расчета (нужно минимум 26 дней)")
    print(f"     - Удаления последней строки (нет цели для прогноза)")
    
    print(f"\n✅ ИТОГО: Модель готова предсказывать RSI на СЛЕДУЮЩИЙ ДЕНЬ")
    print(f"    на основе данных предыдущих дней!")

def predict_specific_date(target_date):
    """
    Показывает прогноз для конкретной даты
    """
    print(f"\n🎯 ПРОГНОЗ RSI НА {target_date}")
    print("=" * 40)
    
    # Загружаем данные
    raw_df = pd.read_csv("accumulatedData_2024.csv", parse_dates=["open_time"])
    processed_df = transform_rsi_features(raw_df)
    
    # Ищем данные за день до целевой даты
    target_date_parsed = pd.to_datetime(target_date)
    prev_date = target_date_parsed - pd.Timedelta(days=1)
    
    # Находим строку с данными за предыдущий день
    matching_rows = processed_df[processed_df.index.map(lambda x: raw_df.iloc[x]['open_time'].date()) == prev_date.date()]
    
    if len(matching_rows) == 0:
        print(f"❌ Нет данных за {prev_date.strftime('%d.%m.%Y')} для прогноза на {target_date}")
        return
    
    # Делаем прогноз
    model, feature_info = load_rsi_model()
    if model is None:
        print("❌ Модель не найдена")
        return
    
    row_data = matching_rows.iloc[0]
    features = row_data[feature_info['important_features']].values.reshape(1, -1)
    features_df = pd.DataFrame(features, columns=[f"f{i}" for i in range(len(feature_info['important_features']))])
    
    prediction = model.predict(features_df.astype('float32'))[0]
    
    print(f"📊 Данные за: {prev_date.strftime('%d.%m.%Y')}")
    print(f"🎯 Прогноз RSI на: {target_date}")
    print(f"📈 Текущий RSI: {row_data['rsi_current']:.2f}")
    print(f"🔮 Прогнозируемый RSI: {prediction:.2f}")

if __name__ == "__main__":
    show_prediction_dates()
    
    # Пример прогноза для конкретной даты
    print("\n" + "="*60)
    predict_specific_date("2022-05-18")  # Пример: прогноз на 18 мая 2022