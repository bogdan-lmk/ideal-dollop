#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("🚀 ПРОСТОЕ ОБЪЯСНЕНИЕ RSI ПРОГНОЗОВ")
print("=" * 50)

print("📅 1. КАК РАБОТАЕТ ПРОГНОЗИРОВАНИЕ:")
print("   • У нас есть данные с 10.01.2022 по 29.06.2025")
print("   • Модель изучает данные за СЕГОДНЯ")
print("   • И предсказывает RSI на ЗАВТРА")
print()

print("📊 2. ПРИМЕР:")
print("   Данные за 17.05.2022:")
print("   • Цена закрытия: 43,100")
print("   • RSI сегодня: 51.47")
print("   • ПРОГНОЗ RSI на 18.05.2022: 51.34")
print()

print("🎯 3. КАК ЗАПУСТИТЬ:")
print("   1) Обучить модель:")
print("      cd '/Users/buyer7/Downloads/Archive 2'")
print("      source rsi_env/bin/activate")
print("      python RSIPredictor.py")
print()
print("   2) Протестировать:")
print("      python RSIPredictionTest.py")
print()
print("   3) Посмотреть примеры:")
print("      python RSI_Usage_Example.py")
print()

print("📈 4. ЧТО ПОКАЗЫВАЮТ РЕЗУЛЬТАТЫ:")
print("   Date: 17.05.2022")
print("   Current RSI: 51.47        <- RSI на 17.05.2022")
print("   Predicted Next Day RSI: 51.34  <- Прогноз на 18.05.2022")
print("   ----------------------------------------")
print()

print("🔍 5. ТОЧНОСТЬ МОДЕЛИ:")
print("   • Средняя ошибка: 4.84 пункта RSI")
print("   • Максимальная ошибка: 15.41 пункта")
print("   • RSI меняется от 0 до 100, так что 4.84 - это хорошо!")
print()

print("⚡ 6. ДЛЯ ЧЕГО ИСПОЛЬЗОВАТЬ:")
print("   • RSI < 30 = перепроданность (возможен рост)")
print("   • RSI > 70 = перекупленность (возможно падение)")
print("   • Если модель предсказывает RSI изменится с 25 на 35,")
print("     это сигнал к покупке!")
print()

print("✅ ГОТОВО! Теперь запускайте скрипты по порядку!")

# Показываем, какие файлы у нас есть
import os
print("\n📁 ФАЙЛЫ В ПРОЕКТЕ:")
files = [f for f in os.listdir('.') if f.endswith('.py') and 'rsi' in f.lower()]
for f in files:
    print(f"   • {f}")

print("\n🎉 УСПЕХОВ В ТРЕЙДИНГЕ!")