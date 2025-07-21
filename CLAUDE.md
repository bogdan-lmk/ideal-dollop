# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a financial trading ML platform for predicting cryptocurrency/stock movements using technical indicators. The codebase implements both classification (trading signals) and regression (price prediction) models with production-ready ONNX export capabilities.

## Development Commands

### Model Training
```bash
# Binary classification for trading signals
python ML.py

# Price prediction with feature engineering
python Predictor.py  

# Position closing decisions
python CloseDeal.py

# Position closure prediction
python CheckClose.py
```

### Model Testing
```bash
# Test trained models
python PredictionTest.py
```

### Data Processing
Models expect OHLCV data with technical indicators. The `transform()` function in `Predictor.py` handles feature engineering for regression tasks.

## Architecture

### Core ML Pipeline
```
Raw OHLCV Data → Feature Engineering → Model Training → Export (PKL/ONNX) → Production
```

### Key Components
- **ML.py**: XGBoost binary classification with SMOTE balancing
- **Predictor.py**: CatBoost regression with Optuna hyperparameter optimization  
- **CloseDeal.py**: Random Forest for position closing decisions
- **CheckClose.py**: XGBoost for position closure prediction

### Data Flow
1. Load data from `accumulatedData_*.csv` files
2. Apply feature engineering (technical indicators, rolling stats)
3. Train models with TimeSeriesSplit cross-validation
4. Export models as both PKL and ONNX formats

## Technical Patterns

### Feature Engineering
- Price differences and normalized changes
- Rolling statistics (mean, std with configurable windows)
- Technical indicators: ATR, EMA, RSI, Bollinger Bands, ADX
- Target engineering for return prediction

### Model Training Patterns
- **Time Series Validation**: Always use `TimeSeriesSplit` for temporal data
- **Class Imbalance**: Use SMOTE, `scale_pos_weight`, or `class_weight='balanced'`
- **Hyperparameter Optimization**: Optuna integration for automated tuning
- **Feature Selection**: Importance-based filtering (threshold > 0.01)

### Model Export
All models save in two formats:
- **PKL**: For Python environments (`joblib.dump/load`)
- **ONNX**: For cross-platform deployment (`convert_sklearn`, `convert_xgboost`)

## Key Libraries
- **ML Frameworks**: XGBoost, CatBoost, LightGBM, scikit-learn
- **Data Processing**: pandas, numpy
- **Optimization**: optuna, imbalanced-learn (SMOTE)
- **Export**: onnx, onnxmltools, skl2onnx, joblib

## Model Artifacts
- `trained_model.pkl/onnx` - Binary classification model
- `xgb_model.pkl` - Price prediction model  
- `short_model.pkl/onnx` - Position closing model
- `close_position_model.pkl/onnx` - Position closure model

## Data Requirements
Input CSV files must contain:
- OHLCV columns: `open_time`, `open`, `high`, `low`, `close`, `volume`
- Technical indicators: `atr`, `fast_ema`, `slow_ema`, `rsi_*`, `adx`, etc.
- Use comma decimal separator for numeric values

## Development Workflow
1. **Data Preparation**: Ensure OHLCV data has required technical indicators
2. **Feature Engineering**: Use/modify `transform()` function in `Predictor.py`
3. **Model Training**: Run appropriate script based on task (classification/regression)
4. **Hyperparameter Tuning**: Uncomment Optuna sections for optimization
5. **Evaluation**: Models output financial metrics (MAE for prices, ROC AUC for signals)
6. **Export**: Models automatically saved in both PKL and ONNX formats

## Adding New Models
1. Follow existing patterns for data loading and preprocessing
2. Use `TimeSeriesSplit` for cross-validation
3. Include both accuracy and financial performance metrics
4. Export in both PKL and ONNX formats for deployment flexibility
5. Add feature importance analysis for interpretability