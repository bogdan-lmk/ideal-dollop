#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import talib
from RSIPredictionTest import load_rsi_model

def get_latest_rsi_prediction():
    """
    Получить РЕАЛЬНЫЙ прогноз RSI на 30.06.2025 на основе данных за 29.06.2025
    """
    print("🎯 ПОЛУЧЕНИЕ РЕАЛЬНОГО ПРОГНОЗА RSI НА 30.06.2025")
    print("=" * 60)
    
    # Загружаем полный датасет
    print("📊 1. Загружаем полные данные...")
    df = pd.read_csv("accumulatedData_2024.csv")
    
    # Конвертируем данные в нужный формат
    print("🔧 2. Конвертируем данные...")
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'atr', 'atr_stop', 
                   'atr_to_price_ratio', 'fast_ema', 'slow_ema', 'ema_fast_deviation',
                   'pchange', 'avpchange', 'gma', 'positionBetweenBands', 'choppiness_index',
                   'volatility_percent', 'rsi_volatility', 'adx', 'rsi_delta', 'linear_regression']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '.', regex=False),
                errors='coerce'
            ).fillna(0)
    
    # Вычисляем RSI и MACD для всех данных
    print("📈 3. Вычисляем технические индикаторы...")
    df['rsi_current'] = talib.RSI(df['close'].values, timeperiod=14)
    
    macd, macd_signal, macd_hist = talib.MACD(df['close'].values, 
                                              fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_histogram'] = macd_hist
    
    # Дополнительные признаки
    df['rsi_sma_5'] = df['rsi_current'].rolling(5).mean()
    df['rsi_sma_10'] = df['rsi_current'].rolling(10).mean()
    df['rsi_momentum'] = df['rsi_current'].diff()
    df['rsi_rate_of_change'] = df['rsi_current'].pct_change()
    
    df['close_diff'] = df['close'].diff()
    df['normalized_price_change'] = df['close_diff'] / df['close'].shift(1)
    df['rolling_mean_close'] = df['close'].rolling(20).mean()
    df['rolling_std_close'] = df['close'].rolling(20).std()
    
    df['volume_sma'] = df['volume'].rolling(10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    df['macd_signal_diff'] = df['macd'] - df['macd_signal']
    df['macd_momentum'] = df['macd'].diff()
    
    # Находим последние доступные данные
    print("🔍 4. Ищем данные за 29.06.2025...")
    
    # Берем последнюю строку с полными данными
    last_row_idx = len(df) - 1
    while last_row_idx >= 0 and pd.isna(df.iloc[last_row_idx]['rsi_current']):
        last_row_idx -= 1
    
    if last_row_idx < 0:
        print("❌ Не найдены данные с RSI")
        return None
    
    last_row = df.iloc[last_row_idx]
    date_str = str(last_row['open_time'])[:10]
    
    print(f"   • Найдены данные за: {date_str}")
    print(f"   • RSI на эту дату: {last_row['rsi_current']:.2f}")
    print(f"   • Цена закрытия: {last_row['close']:.2f}")
    
    # Загружаем обученную модель
    print("\n🤖 5. Загружаем модель...")
    model, feature_info = load_rsi_model()
    
    if model is None:
        print("❌ Модель не найдена. Запустите: python RSIPredictor.py")
        return None
    
    # Подготавливаем признаки для предсказания
    print("🔮 6. Подготавливаем данные для прогноза...")
    
    # Список всех признаков, которые использует модель
    required_features = feature_info['important_features']
    
    # Создаем вектор признаков
    feature_values = []
    missing_features = []
    
    for feature in required_features:
        if feature in last_row and not pd.isna(last_row[feature]):
            feature_values.append(last_row[feature])
        else:
            feature_values.append(0.0)  # Заполняем нулями отсутствующие признаки
            missing_features.append(feature)
    
    if missing_features:
        print(f"   ⚠️  Отсутствующие признаки заполнены нулями: {len(missing_features)} из {len(required_features)}")
    
    # Делаем прогноз
    print("🎯 7. Делаем прогноз...")
    features_array = np.array(feature_values).reshape(1, -1).astype('float32')
    prediction = model.predict(features_array)[0]
    
    # Результаты
    print("\n" + "="*60)
    print("📈 ПРОГНОЗ RSI НА 30.06.2025")
    print("="*60)
    print(f"📊 Данные за: {date_str}")
    print(f"💰 Цена закрытия: {last_row['close']:.2f}")
    print(f"📉 Текущий RSI: {last_row['rsi_current']:.2f}")
    print(f"🎯 ПРОГНОЗ RSI на 30.06.2025: {prediction:.2f}")
    
    # Анализ изменения
    change = prediction - last_row['rsi_current']
    print(f"📈 Изменение RSI: {change:+.2f} пунктов")
    
    # Торговые сигналы
    print("\n💡 ТОРГОВЫЙ АНАЛИЗ:")
    
    if prediction < 30:
        signal = "🟢 СИЛЬНАЯ ПОКУПКА"
        advice = "RSI в зоне перепроданности - возможен отскок"
    elif prediction < 40:
        signal = "🔵 ПОКУПКА"
        advice = "RSI приближается к перепроданности"
    elif prediction > 70:
        signal = "🔴 СИЛЬНАЯ ПРОДАЖА"
        advice = "RSI в зоне перекупленности - возможна коррекция"
    elif prediction > 60:
        signal = "🟠 ПРОДАЖА"
        advice = "RSI приближается к перекупленности"
    else:
        signal = "⚪ НЕЙТРАЛЬНО"
        advice = "RSI в нормальном диапазоне"
    
    print(f"🎯 Сигнал: {signal}")
    print(f"💭 Совет: {advice}")
    
    if abs(change) > 5:
        print(f"⚡ ВАЖНО: Прогнозируется значительное изменение RSI на {abs(change):.1f} пунктов!")
    
    # Дополнительная информация
    print(f"\n📊 ДОПОЛНИТЕЛЬНО:")
    print(f"   • MACD: {last_row.get('macd', 'N/A')}")
    print(f"   • ADX: {last_row.get('adx', 'N/A')}")
    print(f"   • Volatility: {last_row.get('volatility_percent', 'N/A')}%")
    
    return {
        'date': date_str,
        'current_rsi': last_row['rsi_current'],
        'predicted_rsi': prediction,
        'change': change,
        'price': last_row['close'],
        'signal': signal
    }

if __name__ == "__main__":
    result = get_latest_rsi_prediction()
    
    if result:
        print(f"\n✅ ИТОГ: На 30.06.2025 RSI прогнозируется {result['predicted_rsi']:.2f}")
        print(f"   (изменение {result['change']:+.2f} от текущего {result['current_rsi']:.2f})")
        print(f"   Торговый сигнал: {result['signal']}")
    else:
        print("❌ Не удалось получить прогноз")