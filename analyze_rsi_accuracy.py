#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from RSIPredictor import transform_rsi_features
from RSIPredictionTest import load_rsi_model

def analyze_prediction_accuracy():
    """
    Детальный анализ точности прогнозов RSI
    """
    print("📊 АНАЛИЗ ПОГРЕШНОСТИ ПРОГНОЗОВ RSI")
    print("=" * 60)
    
    # Загружаем данные и модель
    print("🔄 1. Загружаем данные и модель...")
    raw_df = pd.read_csv("accumulatedData_2024.csv", parse_dates=["open_time"])
    processed_df = transform_rsi_features(raw_df)
    model, feature_info = load_rsi_model()
    
    if model is None or len(processed_df) == 0:
        print("❌ Ошибка загрузки данных или модели")
        return
    
    print(f"   ✅ Загружено {len(processed_df)} обработанных записей")
    
    # Подготавливаем данные для анализа
    print("🔧 2. Подготавливаем данные для анализа...")
    
    features = processed_df[feature_info['important_features']]
    features.columns = [f"f{i}" for i in range(len(feature_info['important_features']))]
    
    # Фактические и прогнозируемые значения
    y_actual = processed_df['rsi_next_day'].values
    y_predicted = model.predict(features.astype('float32'))
    
    # Основные метрики ошибок
    print("📈 3. ОСНОВНЫЕ МЕТРИКИ ТОЧНОСТИ:")
    print("-" * 50)
    
    # Абсолютные ошибки
    absolute_errors = np.abs(y_actual - y_predicted)
    
    mae = np.mean(absolute_errors)  # Средняя абсолютная ошибка
    median_ae = np.median(absolute_errors)  # Медиана абсолютной ошибки
    max_error = np.max(absolute_errors)  # Максимальная ошибка
    min_error = np.min(absolute_errors)  # Минимальная ошибка
    std_error = np.std(absolute_errors)  # Стандартное отклонение ошибки
    
    print(f"📊 Средняя абсолютная ошибка (MAE): {mae:.2f} пунктов RSI")
    print(f"📊 Медиана абсолютной ошибки: {median_ae:.2f} пунктов RSI")
    print(f"📊 Максимальная ошибка: {max_error:.2f} пунктов RSI")
    print(f"📊 Минимальная ошибка: {min_error:.2f} пунктов RSI")
    print(f"📊 Стандартное отклонение: {std_error:.2f} пунктов RSI")
    
    # Процентная точность
    print(f"\n🎯 4. ПРОЦЕНТНАЯ ТОЧНОСТЬ:")
    print("-" * 50)
    
    # Точность в разных диапазонах ошибок
    within_1 = np.sum(absolute_errors <= 1) / len(absolute_errors) * 100
    within_2 = np.sum(absolute_errors <= 2) / len(absolute_errors) * 100
    within_3 = np.sum(absolute_errors <= 3) / len(absolute_errors) * 100
    within_5 = np.sum(absolute_errors <= 5) / len(absolute_errors) * 100
    within_10 = np.sum(absolute_errors <= 10) / len(absolute_errors) * 100
    
    print(f"✅ Ошибка ≤ 1 пункт RSI: {within_1:.1f}% прогнозов")
    print(f"✅ Ошибка ≤ 2 пункта RSI: {within_2:.1f}% прогнозов")
    print(f"✅ Ошибка ≤ 3 пункта RSI: {within_3:.1f}% прогнозов")
    print(f"✅ Ошибка ≤ 5 пунктов RSI: {within_5:.1f}% прогнозов")
    print(f"✅ Ошибка ≤ 10 пунктов RSI: {within_10:.1f}% прогнозов")
    
    # Анализ по диапазонам RSI
    print(f"\n📊 5. ТОЧНОСТЬ ПО ДИАПАЗОНАМ RSI:")
    print("-" * 50)
    
    # Разделяем на диапазоны RSI
    oversold_mask = y_actual < 30  # Перепроданность
    overbought_mask = y_actual > 70  # Перекупленность  
    neutral_mask = (y_actual >= 30) & (y_actual <= 70)  # Нейтральная зона
    
    if np.sum(oversold_mask) > 0:
        oversold_mae = np.mean(absolute_errors[oversold_mask])
        print(f"🟢 Перепроданность (RSI < 30): {oversold_mae:.2f} MAE ({np.sum(oversold_mask)} прогнозов)")
    else:
        print(f"🟢 Перепроданность (RSI < 30): Нет данных")
    
    if np.sum(overbought_mask) > 0:
        overbought_mae = np.mean(absolute_errors[overbought_mask])
        print(f"🔴 Перекупленность (RSI > 70): {overbought_mae:.2f} MAE ({np.sum(overbought_mask)} прогнозов)")
    else:
        print(f"🔴 Перекупленность (RSI > 70): Нет данных")
    
    if np.sum(neutral_mask) > 0:
        neutral_mae = np.mean(absolute_errors[neutral_mask])
        print(f"⚪ Нейтральная зона (30-70): {neutral_mae:.2f} MAE ({np.sum(neutral_mask)} прогнозов)")
    
    # Анализ направления изменения
    print(f"\n🎯 6. ТОЧНОСТЬ НАПРАВЛЕНИЯ ИЗМЕНЕНИЯ:")
    print("-" * 50)
    
    # Вычисляем направления изменений
    actual_direction = np.sign(y_actual - processed_df['rsi_current'].values)
    predicted_direction = np.sign(y_predicted - processed_df['rsi_current'].values)
    
    # Точность направления
    direction_accuracy = np.sum(actual_direction == predicted_direction) / len(actual_direction) * 100
    
    print(f"📈 Точность направления изменения: {direction_accuracy:.1f}%")
    
    # Детализация по направлениям
    up_predictions = np.sum(predicted_direction == 1)
    down_predictions = np.sum(predicted_direction == -1)
    flat_predictions = np.sum(predicted_direction == 0)
    
    print(f"   • Прогнозы роста RSI: {up_predictions} ({up_predictions/len(predicted_direction)*100:.1f}%)")
    print(f"   • Прогнозы падения RSI: {down_predictions} ({down_predictions/len(predicted_direction)*100:.1f}%)")
    print(f"   • Прогнозы без изменений: {flat_predictions} ({flat_predictions/len(predicted_direction)*100:.1f}%)")
    
    # Анализ больших ошибок
    print(f"\n⚠️  7. АНАЛИЗ БОЛЬШИХ ОШИБОК:")
    print("-" * 50)
    
    large_errors_mask = absolute_errors > 10
    large_errors_count = np.sum(large_errors_mask)
    
    if large_errors_count > 0:
        print(f"🚨 Ошибки > 10 пунктов: {large_errors_count} случаев ({large_errors_count/len(absolute_errors)*100:.1f}%)")
        
        # Показываем примеры больших ошибок
        large_error_indices = np.where(large_errors_mask)[0][:5]  # Первые 5 случаев
        
        print(f"   Примеры больших ошибок:")
        for idx in large_error_indices:
            actual_val = y_actual[idx]
            pred_val = y_predicted[idx]
            error = absolute_errors[idx]
            print(f"   • Факт: {actual_val:.1f}, Прогноз: {pred_val:.1f}, Ошибка: {error:.1f}")
    else:
        print(f"✅ Больших ошибок (>10 пунктов) не обнаружено")
    
    # Относительная точность
    print(f"\n📊 8. ОТНОСИТЕЛЬНАЯ ТОЧНОСТЬ:")
    print("-" * 50)
    
    # Относительная ошибка в процентах от значения RSI
    relative_errors = (absolute_errors / y_actual) * 100
    mean_relative_error = np.mean(relative_errors)
    median_relative_error = np.median(relative_errors)
    
    print(f"📊 Средняя относительная ошибка: {mean_relative_error:.1f}%")
    print(f"📊 Медиана относительной ошибки: {median_relative_error:.1f}%")
    
    # Сравнение с базовыми моделями
    print(f"\n🏆 9. СРАВНЕНИЕ С БАЗОВЫМИ МОДЕЛЯМИ:")
    print("-" * 50)
    
    # Модель "без изменений" (RSI завтра = RSI сегодня)
    no_change_errors = np.abs(y_actual - processed_df['rsi_current'].values)
    no_change_mae = np.mean(no_change_errors)
    
    # Модель средних значений
    mean_rsi = np.mean(processed_df['rsi_current'].values)
    mean_model_errors = np.abs(y_actual - mean_rsi)
    mean_model_mae = np.mean(mean_model_errors)
    
    print(f"🤖 Наша модель MAE: {mae:.2f}")
    print(f"📊 Модель 'без изменений' MAE: {no_change_mae:.2f}")
    print(f"📊 Модель 'среднее значение' MAE: {mean_model_mae:.2f}")
    
    improvement_vs_no_change = ((no_change_mae - mae) / no_change_mae) * 100
    improvement_vs_mean = ((mean_model_mae - mae) / mean_model_mae) * 100
    
    print(f"✅ Улучшение относительно 'без изменений': {improvement_vs_no_change:.1f}%")
    print(f"✅ Улучшение относительно 'среднего': {improvement_vs_mean:.1f}%")
    
    # Создаем визуализацию
    create_accuracy_plots(y_actual, y_predicted, absolute_errors)
    
    # Итоговая оценка
    print(f"\n🎯 10. ИТОГОВАЯ ОЦЕНКА МОДЕЛИ:")
    print("=" * 60)
    
    if mae <= 3:
        grade = "🏆 ОТЛИЧНАЯ"
    elif mae <= 5:
        grade = "✅ ХОРОШАЯ"
    elif mae <= 7:
        grade = "⚡ УДОВЛЕТВОРИТЕЛЬНАЯ"
    else:
        grade = "⚠️  ТРЕБУЕТ УЛУЧШЕНИЯ"
    
    print(f"📊 Средняя ошибка: {mae:.2f} пунктов RSI")
    print(f"🎯 Оценка модели: {grade}")
    print(f"📈 Точность направления: {direction_accuracy:.1f}%")
    print(f"✅ Прогнозов в пределах 5 пунктов: {within_5:.1f}%")
    
    print(f"\n💡 ПРАКТИЧЕСКИЕ ВЫВОДЫ:")
    print(f"   • Модель дает точные прогнозы в {within_5:.0f}% случаев (±5 пунктов)")
    print(f"   • Средняя ошибка {mae:.1f} пунктов приемлема для RSI (0-100)")
    print(f"   • Направление изменения предсказывается в {direction_accuracy:.0f}% случаев")
    
    return {
        'mae': mae,
        'max_error': max_error,
        'direction_accuracy': direction_accuracy,
        'within_5': within_5,
        'within_3': within_3,
        'within_1': within_1
    }

def create_accuracy_plots(y_actual, y_predicted, absolute_errors):
    """
    Создает графики для анализа точности
    """
    plt.figure(figsize=(15, 10))
    
    # График 1: Фактические vs Прогнозируемые
    plt.subplot(2, 3, 1)
    plt.scatter(y_actual, y_predicted, alpha=0.6, s=20)
    plt.plot([0, 100], [0, 100], 'r--', lw=2)
    plt.xlabel('Фактический RSI')
    plt.ylabel('Прогнозируемый RSI')
    plt.title('Фактический vs Прогнозируемый RSI')
    plt.grid(True, alpha=0.3)
    
    # График 2: Распределение ошибок
    plt.subplot(2, 3, 2)
    plt.hist(absolute_errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Абсолютная ошибка')
    plt.ylabel('Частота')
    plt.title('Распределение абсолютных ошибок')
    plt.axvline(np.mean(absolute_errors), color='red', linestyle='--', 
                label=f'Среднее: {np.mean(absolute_errors):.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 3: Ошибки во времени
    plt.subplot(2, 3, 3)
    plt.plot(absolute_errors, alpha=0.7)
    plt.xlabel('Номер прогноза')
    plt.ylabel('Абсолютная ошибка')
    plt.title('Ошибки во времени')
    plt.axhline(np.mean(absolute_errors), color='red', linestyle='--', 
                label=f'Среднее: {np.mean(absolute_errors):.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 4: Ошибки по диапазонам RSI
    plt.subplot(2, 3, 4)
    rsi_bins = np.arange(0, 101, 10)
    bin_errors = []
    bin_centers = []
    
    for i in range(len(rsi_bins)-1):
        mask = (y_actual >= rsi_bins[i]) & (y_actual < rsi_bins[i+1])
        if np.sum(mask) > 0:
            bin_errors.append(np.mean(absolute_errors[mask]))
            bin_centers.append((rsi_bins[i] + rsi_bins[i+1]) / 2)
    
    if bin_errors:
        plt.bar(bin_centers, bin_errors, width=8, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Диапазон RSI')
        plt.ylabel('Средняя абсолютная ошибка')
        plt.title('Ошибки по диапазонам RSI')
        plt.grid(True, alpha=0.3)
    
    # График 5: Накопительное распределение ошибок
    plt.subplot(2, 3, 5)
    sorted_errors = np.sort(absolute_errors)
    cumulative = np.arange(1, len(sorted_errors)+1) / len(sorted_errors) * 100
    plt.plot(sorted_errors, cumulative, linewidth=2)
    plt.xlabel('Абсолютная ошибка')
    plt.ylabel('Процент прогнозов (%)')
    plt.title('Накопительное распределение ошибок')
    plt.grid(True, alpha=0.3)
    
    # Добавляем линии для ключевых значений
    for error_threshold in [1, 3, 5, 10]:
        pct = np.sum(absolute_errors <= error_threshold) / len(absolute_errors) * 100
        plt.axvline(error_threshold, linestyle='--', alpha=0.7, 
                   label=f'{error_threshold}: {pct:.1f}%')
    plt.legend()
    
    # График 6: Остатки (residuals)
    plt.subplot(2, 3, 6)
    residuals = y_predicted - y_actual
    plt.scatter(y_predicted, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Прогнозируемый RSI')
    plt.ylabel('Остатки (Прогноз - Факт)')
    plt.title('График остатков')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rsi_accuracy_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Графики сохранены как 'rsi_accuracy_analysis.png'")
    plt.show()

if __name__ == "__main__":
    results = analyze_prediction_accuracy()
    
    if results:
        print(f"\n🎯 КРАТКИЕ РЕЗУЛЬТАТЫ:")
        print(f"   MAE: {results['mae']:.2f} пунктов")
        print(f"   Максимальная ошибка: {results['max_error']:.2f} пунктов")
        print(f"   Точность направления: {results['direction_accuracy']:.1f}%")
        print(f"   Прогнозов в ±5 пунктов: {results['within_5']:.1f}%")