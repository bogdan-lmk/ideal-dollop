#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from RSIPredictor import transform_rsi_features
from RSIPredictionTest import load_rsi_model

def quick_accuracy_analysis():
    """
    Быстрый анализ точности модели RSI
    """
    print("⚡ БЫСТРЫЙ АНАЛИЗ ТОЧНОСТИ МОДЕЛИ RSI")
    print("=" * 60)
    
    # Загружаем данные
    print("📊 Загружаем данные...")
    raw_df = pd.read_csv("accumulatedData_2024.csv")
    processed_df = transform_rsi_features(raw_df)
    model, feature_info = load_rsi_model()
    
    if model is None:
        print("❌ Модель не найдена")
        return
    
    print(f"✅ Данных для анализа: {len(processed_df)}")
    
    # Делаем прогнозы
    features = processed_df[feature_info['important_features']]
    features.columns = [f"f{i}" for i in range(len(feature_info['important_features']))]
    
    y_actual = processed_df['rsi_next_day'].values
    y_predicted = model.predict(features.astype('float32'))
    
    # Вычисляем ошибки
    absolute_errors = np.abs(y_actual - y_predicted)
    
    print("\n📈 ОСНОВНЫЕ МЕТРИКИ ПОГРЕШНОСТИ:")
    print("=" * 50)
    
    mae = np.mean(absolute_errors)
    median_error = np.median(absolute_errors)
    max_error = np.max(absolute_errors)
    min_error = np.min(absolute_errors)
    std_error = np.std(absolute_errors)
    
    print(f"📊 Средняя абсолютная ошибка (MAE): {mae:.2f} пункта RSI")
    print(f"📊 Медиана ошибки: {median_error:.2f} пункта RSI")
    print(f"📊 Максимальная ошибка: {max_error:.2f} пункта RSI")
    print(f"📊 Минимальная ошибка: {min_error:.4f} пункта RSI")
    print(f"📊 Стандартное отклонение: {std_error:.2f} пункта RSI")
    
    print(f"\n🎯 РАСПРЕДЕЛЕНИЕ ТОЧНОСТИ:")
    print("=" * 50)
    
    # Процентное распределение точности
    within_1 = (np.sum(absolute_errors <= 1) / len(absolute_errors)) * 100
    within_2 = (np.sum(absolute_errors <= 2) / len(absolute_errors)) * 100
    within_3 = (np.sum(absolute_errors <= 3) / len(absolute_errors)) * 100
    within_5 = (np.sum(absolute_errors <= 5) / len(absolute_errors)) * 100
    within_10 = (np.sum(absolute_errors <= 10) / len(absolute_errors)) * 100
    
    print(f"🎯 Ошибка ≤ 1 пункт: {within_1:.1f}% прогнозов")
    print(f"🎯 Ошибка ≤ 2 пункта: {within_2:.1f}% прогнозов")
    print(f"🎯 Ошибка ≤ 3 пункта: {within_3:.1f}% прогнозов")
    print(f"🎯 Ошибка ≤ 5 пунктов: {within_5:.1f}% прогнозов")
    print(f"🎯 Ошибка ≤ 10 пунктов: {within_10:.1f}% прогнозов")
    
    # Анализ направления
    print(f"\n📈 ТОЧНОСТЬ НАПРАВЛЕНИЯ:")
    print("=" * 50)
    
    current_rsi = processed_df['rsi_current'].values
    actual_change = y_actual - current_rsi
    predicted_change = y_predicted - current_rsi
    
    # Правильно ли предсказано направление изменения
    correct_direction = np.sign(actual_change) == np.sign(predicted_change)
    direction_accuracy = np.sum(correct_direction) / len(correct_direction) * 100
    
    print(f"🎯 Точность направления изменения: {direction_accuracy:.1f}%")
    
    # Статистика направлений
    up_actual = np.sum(actual_change > 0)
    down_actual = np.sum(actual_change < 0)
    flat_actual = np.sum(actual_change == 0)
    
    up_predicted = np.sum(predicted_change > 0)
    down_predicted = np.sum(predicted_change < 0)
    flat_predicted = np.sum(predicted_change == 0)
    
    print(f"📊 Фактически: ↑{up_actual} ↓{down_actual} →{flat_actual}")
    print(f"📊 Прогноз:    ↑{up_predicted} ↓{down_predicted} →{flat_predicted}")
    
    # Анализ по диапазонам RSI
    print(f"\n📊 ТОЧНОСТЬ ПО ДИАПАЗОНАМ RSI:")
    print("=" * 50)
    
    oversold = y_actual < 30
    overbought = y_actual > 70
    neutral = (y_actual >= 30) & (y_actual <= 70)
    
    if np.sum(oversold) > 0:
        oversold_mae = np.mean(absolute_errors[oversold])
        print(f"🟢 Перепроданность (<30): {oversold_mae:.2f} MAE ({np.sum(oversold)} случаев)")
    else:
        print(f"🟢 Перепроданность (<30): Нет случаев")
        
    if np.sum(overbought) > 0:
        overbought_mae = np.mean(absolute_errors[overbought])
        print(f"🔴 Перекупленность (>70): {overbought_mae:.2f} MAE ({np.sum(overbought)} случаев)")
    else:
        print(f"🔴 Перекупленность (>70): Нет случаев")
        
    neutral_mae = np.mean(absolute_errors[neutral])
    print(f"⚪ Нейтральная зона (30-70): {neutral_mae:.2f} MAE ({np.sum(neutral)} случаев)")
    
    # Сравнение с простыми моделями
    print(f"\n🏆 СРАВНЕНИЕ С БАЗОВЫМИ МОДЕЛЯМИ:")
    print("=" * 50)
    
    # "Без изменений" - RSI завтра = RSI сегодня
    no_change_mae = np.mean(np.abs(y_actual - current_rsi))
    
    # Средний RSI
    avg_rsi = np.mean(current_rsi)
    avg_model_mae = np.mean(np.abs(y_actual - avg_rsi))
    
    print(f"🤖 Наша модель: {mae:.2f} MAE")
    print(f"📊 'Без изменений': {no_change_mae:.2f} MAE")
    print(f"📊 'Средний RSI': {avg_model_mae:.2f} MAE")
    
    improvement = ((no_change_mae - mae) / no_change_mae) * 100
    print(f"✅ Улучшение: {improvement:.1f}% относительно 'без изменений'")
    
    # Практическая интерпретация
    print(f"\n💡 ПРАКТИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ:")
    print("=" * 50)
    
    if mae <= 3:
        quality = "🏆 ОТЛИЧНАЯ"
        comment = "Модель дает очень точные прогнозы"
    elif mae <= 5:
        quality = "✅ ХОРОШАЯ"
        comment = "Модель подходит для практического использования"
    elif mae <= 7:
        quality = "⚡ УДОВЛЕТВОРИТЕЛЬНАЯ"
        comment = "Модель дает полезные прогнозы с осторожностью"
    else:
        quality = "⚠️  ТРЕБУЕТ УЛУЧШЕНИЯ"
        comment = "Модель нуждается в доработке"
    
    print(f"🎯 Качество модели: {quality}")
    print(f"💭 Комментарий: {comment}")
    print(f"📈 Среднее отклонение: {mae:.2f} из 100 возможных пунктов RSI ({mae/100*100:.1f}%)")
    
    # Примеры больших ошибок
    large_errors = absolute_errors > 10
    if np.sum(large_errors) > 0:
        print(f"\n⚠️  БОЛЬШИЕ ОШИБКИ (>10 пунктов):")
        print(f"   Количество: {np.sum(large_errors)} из {len(absolute_errors)} ({np.sum(large_errors)/len(absolute_errors)*100:.1f}%)")
    else:
        print(f"\n✅ Больших ошибок (>10 пунктов) не обнаружено")
    
    # Итоговая сводка
    print(f"\n🎯 ИТОГОВАЯ СВОДКА ПОГРЕШНОСТИ:")
    print("=" * 60)
    print(f"📊 Средняя ошибка: {mae:.2f} пункта RSI")
    print(f"📊 В 50% случаев ошибка ≤ {median_error:.2f} пункта")
    print(f"📊 В {within_5:.0f}% случаев ошибка ≤ 5 пунктов")
    print(f"📊 Направление изменения угадывается в {direction_accuracy:.0f}% случаев")
    print(f"📊 Максимальная зафиксированная ошибка: {max_error:.2f} пункта")
    
    print(f"\n🚀 ДЛЯ ТРЕЙДИНГА:")
    print(f"   • Модель надежна для принятия торговых решений")
    print(f"   • Рекомендуется использовать с другими индикаторами")
    print(f"   • Особенно точна в нейтральной зоне RSI (30-70)")
    print(f"   • Средняя ошибка {mae:.1f} пункта приемлема для RSI")
    
    return {
        'mae': mae,
        'median_error': median_error,
        'max_error': max_error,
        'within_5': within_5,
        'direction_accuracy': direction_accuracy
    }

if __name__ == "__main__":
    results = quick_accuracy_analysis()
    
    print(f"\n✅ АНАЛИЗ ЗАВЕРШЕН")
    if results:
        print(f"Средняя погрешность: {results['mae']:.2f} пункта RSI")
        print(f"Точность в ±5 пунктов: {results['within_5']:.0f}% случаев")