#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from RSIPredictor import transform_rsi_features
from RSIPredictionTest import load_rsi_model

def predict_for_date(target_date="2025-06-30"):
    """
    Делаем прогноз RSI на указанную дату
    """
    print(f"🎯 ПРОГНОЗ RSI НА {target_date}")
    print("=" * 50)
    
    # Загружаем данные
    print("📊 1. Загружаем данные...")
    raw_df = pd.read_csv("accumulatedData_2024.csv", parse_dates=["open_time"])
    
    # Проверяем, есть ли данные за предыдущий день
    target_date_parsed = pd.to_datetime(target_date)
    prev_date = target_date_parsed - pd.Timedelta(days=1)
    
    print(f"   • Ищем данные за {prev_date.strftime('%d.%m.%Y')} для прогноза на {target_date}")
    
    # Проверяем последние даты в данных
    print(f"   • Последние 3 даты в данных:")
    for i in range(3):
        date_str = str(raw_df.iloc[-(i+1)]['open_time'])[:10]
        print(f"     - {date_str}")
    
    # Обрабатываем данные
    print("\n🔧 2. Обрабатываем данные...")
    processed_df = transform_rsi_features(raw_df)
    
    if len(processed_df) == 0:
        print("❌ Нет обработанных данных")
        return
    
    # Берем самую последнюю строку (она будет за 29.06.2025)
    last_row = processed_df.iloc[-1]
    last_date_idx = processed_df.index[-1]
    actual_date = str(raw_df.iloc[last_date_idx]['open_time'])[:10]
    
    print(f"   • Последние обработанные данные за: {actual_date}")
    print(f"   • Текущий RSI: {last_row['rsi_current']:.2f}")
    
    # Загружаем модель
    print("\n🤖 3. Загружаем модель...")
    model, feature_info = load_rsi_model()
    
    if model is None:
        print("❌ Модель не найдена. Сначала запустите: python RSIPredictor.py")
        return
    
    # Подготавливаем данные для прогноза
    print("\n🔮 4. Делаем прогноз...")
    features = last_row[feature_info['important_features']].values.reshape(1, -1)
    features_df = pd.DataFrame(features, columns=[f"f{i}" for i in range(len(feature_info['important_features']))])
    
    # Делаем предикт
    prediction = model.predict(features_df.astype('float32'))[0]
    
    # Результат
    print(f"\n📈 РЕЗУЛЬТАТ:")
    print(f"   🗓  Данные за: {actual_date}")
    print(f"   📊 Текущий RSI: {last_row['rsi_current']:.2f}")
    print(f"   🎯 ПРОГНОЗ RSI на {target_date}: {prediction:.2f}")
    
    # Анализ прогноза
    print(f"\n💡 АНАЛИЗ:")
    current_rsi = last_row['rsi_current']
    change = prediction - current_rsi
    
    if prediction < 30:
        signal = "🟢 СИЛЬНАЯ ПОКУПКА - Сильная перепроданность"
    elif prediction < 40:
        signal = "🔵 ПОКУПКА - Перепроданность"
    elif prediction > 70:
        signal = "🔴 СИЛЬНАЯ ПРОДАЖА - Сильная перекупленность"
    elif prediction > 60:
        signal = "🟠 ПРОДАЖА - Перекупленность"
    else:
        signal = "⚪ НЕЙТРАЛЬНО - Нормальный диапазон"
    
    print(f"   📈 Изменение RSI: {change:+.2f} пунктов")
    print(f"   🎯 Торговый сигнал: {signal}")
    
    if abs(change) > 5:
        print(f"   ⚡ ВНИМАНИЕ: Прогнозируется значительное изменение RSI!")
    
    return prediction, current_rsi, change

def get_latest_predictions(num_days=5):
    """
    Получить прогнозы на несколько последних дней
    """
    print(f"\n📅 ПОСЛЕДНИЕ {num_days} ПРОГНОЗОВ:")
    print("=" * 60)
    print("Дата данных  | Текущий RSI | Прогноз RSI | Изменение | Сигнал")
    print("-" * 60)
    
    # Загружаем данные
    raw_df = pd.read_csv("accumulatedData_2024.csv", parse_dates=["open_time"])
    processed_df = transform_rsi_features(raw_df)
    model, feature_info = load_rsi_model()
    
    if model is None or len(processed_df) < num_days:
        print("❌ Недостаточно данных или модель не найдена")
        return
    
    # Берем последние дни
    last_rows = processed_df.tail(num_days)
    
    for i, (idx, row) in enumerate(last_rows.iterrows()):
        # Дата
        date_str = str(raw_df.iloc[idx]['open_time'])[:10]
        
        # Текущий RSI
        current_rsi = row['rsi_current']
        
        # Прогноз
        features = row[feature_info['important_features']].values.reshape(1, -1)
        features_df = pd.DataFrame(features, columns=[f"f{i}" for i in range(len(feature_info['important_features']))])
        prediction = model.predict(features_df.astype('float32'))[0]
        
        # Изменение
        change = prediction - current_rsi
        
        # Сигнал
        if prediction < 30:
            signal = "🟢 ПОКУПКА"
        elif prediction > 70:
            signal = "🔴 ПРОДАЖА"
        else:
            signal = "⚪ НЕЙТРАЛ"
        
        print(f"{date_str} | {current_rsi:10.2f} | {prediction:10.2f} | {change:8.2f} | {signal}")

if __name__ == "__main__":
    print("🚀 ПРОГНОЗ RSI НА СЛЕДУЮЩИЙ ДЕНЬ")
    print("=" * 60)
    
    # Прогноз на 30.06.2025
    try:
        prediction, current, change = predict_for_date("2025-06-30")
        
        # Показываем последние прогнозы для контекста
        get_latest_predictions(5)
        
        print(f"\n✅ ИТОГО: Прогноз RSI на 30.06.2025 = {prediction:.2f}")
        print(f"   (изменение: {change:+.2f} от текущего {current:.2f})")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("\nПроверьте:")
        print("1. Запущена ли модель: python RSIPredictor.py")
        print("2. Есть ли данные в accumulatedData_2024.csv")