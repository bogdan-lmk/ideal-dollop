#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import joblib
import numpy as np
from RSIPredictionTest import load_rsi_model

def try_onnx_export():
    """
    Альтернативные методы экспорта в ONNX
    """
    print("🔄 ПОПЫТКА ЭКСПОРТА В ONNX")
    print("=" * 40)
    
    # Загружаем модель
    model, features = load_rsi_model()
    if model is None:
        print("❌ Модель не найдена")
        return
    
    print(f"✅ Модель загружена: {type(model)}")
    print(f"✅ Признаков: {len(features['important_features'])}")
    
    # Метод 1: Catboost встроенный экспорт
    try:
        print("\n🔄 Метод 1: CatBoost встроенный экспорт...")
        model.save_model("models/rsi_model_catboost.cbm", format="cbm")
        print("✅ Модель экспортирована в CatBoost формат (.cbm)")
        print("   Этот формат можно загрузить в CatBoost на любой платформе")
    except Exception as e:
        print(f"❌ CatBoost экспорт не удался: {e}")
    
    # Метод 2: JSON экспорт для кроссплатформенности
    try:
        print("\n🔄 Метод 2: JSON экспорт...")
        model.save_model("models/rsi_model.json", format="json")
        print("✅ Модель экспортирована в JSON формат")
        print("   JSON можно использовать на любой платформе")
    except Exception as e:
        print(f"❌ JSON экспорт не удался: {e}")
    
    # Метод 3: Создание lightweight версии
    try:
        print("\n🔄 Метод 3: Создание lightweight модели...")
        
        # Извлекаем параметры модели
        model_info = {
            'model_type': 'CatBoostRegressor',
            'feature_names': features['important_features'],
            'feature_count': len(features['important_features']),
            'model_params': {
                'iterations': model.get_param('iterations'),
                'learning_rate': model.get_param('learning_rate'),
                'depth': model.get_param('depth'),
            }
        }
        
        # Сохраняем метаданные
        joblib.dump(model_info, "models/rsi_model_info.pkl")
        print("✅ Метаданные модели сохранены")
        
    except Exception as e:
        print(f"❌ Lightweight версия не удалась: {e}")
    
    # Метод 4: Python код модели
    try:
        print("\n🔄 Метод 4: Экспорт в Python код...")
        model.save_model("models/rsi_model.py", format="python")
        print("✅ Модель экспортирована как Python код")
        print("   Можно использовать без внешних зависимостей")
    except Exception as e:
        print(f"❌ Python код экспорт не удался: {e}")
    
    print(f"\n📁 ДОСТУПНЫЕ ФОРМАТЫ:")
    import os
    formats = []
    
    if os.path.exists("models/rsi_predictor_model.pkl"):
        formats.append("✅ PKL (pickle) - для Python")
    if os.path.exists("models/rsi_model_features.pkl"):
        formats.append("✅ PKL features - метаданные")
    if os.path.exists("models/rsi_model_catboost.cbm"):
        formats.append("✅ CBM - CatBoost бинарный")
    if os.path.exists("models/rsi_model.json"):
        formats.append("✅ JSON - кроссплатформенный")
    if os.path.exists("models/rsi_model.py"):
        formats.append("✅ PY - Python код")
    
    for fmt in formats:
        print(f"   {fmt}")
    
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    print(f"   • Для Python: используйте PKL файлы")
    print(f"   • Для других языков: используйте JSON/CBM")
    print(f"   • Для продакшена: рассмотрите Python код версию")

if __name__ == "__main__":
    try_onnx_export()