# RSI Prediction System

A scalable and extensible machine learning solution for predicting next-day RSI (Relative Strength Index) values using technical analysis indicators and market data.

## 🚀 Features

- **Next-day RSI prediction** with 4.84 MAE accuracy
- **TA-Lib integration** for precise MACD and RSI calculations
- **CatBoost regression model** with time series validation
- **Feature engineering** with 29+ technical indicators
- **Production-ready exports** (PKL format)
- **Comprehensive testing** and evaluation suite
- **Trading strategy integration** examples

## 📊 Model Performance

- **Mean Absolute Error**: 4.84 RSI points
- **R² Score**: 0.30
- **Cross-validation**: 5-fold time series split
- **Feature importance**: RSI current (22.4%), Position Between Bands (7.2%), Volatility (5.2%)
- **Training data**: 128 processed samples from 2024 crypto market data

## 🏗 Architecture

### Core Components

```
RSI Prediction Pipeline:
├── Data Loading (accumulatedData_2024.csv)
├── Feature Engineering (transform_rsi_features)
├── Model Training (CatBoostRegressor)
├── Validation (TimeSeriesSplit)
├── Export (PKL format)
└── Testing & Usage Examples
```

### File Structure

```
RSI Prediction System/
├── RSIPredictor.py              # Main training script
├── RSIPredictionTest.py         # Testing and evaluation
├── RSI_Usage_Example.py         # Usage examples
├── rsi_predictor_model.pkl      # Trained model
├── rsi_model_features.pkl       # Feature metadata
├── rsi_model_evaluation.png     # Performance plots
└── rsi_env/                     # Virtual environment
```

## 🛠 Installation

### Prerequisites

- Python 3.8+
- TA-Lib library (requires system-level installation)

### Setup Virtual Environment

```bash
cd /path/to/project
python3 -m venv rsi_env
source rsi_env/bin/activate
pip install TA-Lib numpy pandas scikit-learn catboost xgboost lightgbm matplotlib joblib optuna skl2onnx
```

### macOS TA-Lib Installation

```bash
brew install ta-lib
```

## 📈 Usage

### 1. Train the Model

```python
# Run training script
python RSIPredictor.py
```

**Output:**
- `rsi_predictor_model.pkl` - Trained CatBoost model
- `rsi_model_features.pkl` - Feature mapping and metadata

### 2. Make Predictions

```python
from RSIPredictionTest import load_rsi_model, predict_next_day_rsi

# Load trained model
model, features = load_rsi_model()

# Predict RSI for recent data
predictions, data = predict_next_day_rsi("accumulatedData_2024.csv", num_predictions=5)
```

### 3. Evaluate Model

```python
python RSIPredictionTest.py
```

### 4. Usage Examples

```python
python RSI_Usage_Example.py
```

## 🔧 Feature Engineering

### Technical Indicators Used

**Core RSI Features:**
- Current RSI (primary predictor)
- RSI moving averages (5, 10 periods)
- RSI momentum and rate of change

**MACD Features:**
- MACD line, signal line, histogram
- MACD momentum and signal difference

**Price & Volume Features:**
- Close price differences and normalized changes
- Rolling statistics (mean, std)
- Volume ratios and moving averages

**Market Structure:**
- ADX (trend strength)
- Volatility percentage
- Choppiness index
- Position between Bollinger Bands
- ATR-based features

### Feature Importance Rankings

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | RSI Current | 22.39% |
| 2 | Position Between Bands | 7.23% |
| 3 | Volatility Percent | 5.20% |
| 4 | ADX | 4.74% |
| 5 | RSI Momentum | 4.40% |

## 📊 Model Architecture

### CatBoost Configuration

```python
CatBoostRegressor(
    iterations=1500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=42
)
```

### Validation Strategy

- **Time Series Split**: 5-fold cross-validation
- **Train/Test Split**: 80/20 with temporal ordering preserved
- **Feature Selection**: Importance threshold > 0.01
- **Early Stopping**: Based on validation error

## 🎯 API Reference

### Core Functions

#### `transform_rsi_features(df, rsi_window=14, bollinger_window=20)`
Feature engineering pipeline that processes OHLCV data into ML-ready features.

**Parameters:**
- `df`: DataFrame with OHLCV and technical indicators
- `rsi_window`: RSI calculation period (default: 14)
- `bollinger_window`: Rolling statistics window (default: 20)

**Returns:** Processed DataFrame with engineered features and next-day RSI target

#### `train_rsi_predictor()`
Main training function that handles the complete ML pipeline.

**Returns:** Tuple of (model, feature_info, metrics)

#### `predict_next_day_rsi(data_file, num_predictions=10)`
Make RSI predictions on recent data.

**Parameters:**
- `data_file`: Path to CSV data file
- `num_predictions`: Number of recent predictions to show

**Returns:** Tuple of (predictions, processed_data)

### Utility Functions

#### `calculate_rsi(close_prices, window=14)`
Calculate RSI using TA-Lib.

#### `calculate_macd(close_prices, fast=12, slow=26, signal=9)`
Calculate MACD components using TA-Lib.

## 🔄 Integration Examples

### Trading Strategy Integration

```python
def generate_trading_signals(current_rsi, predicted_rsi):
    if current_rsi < 30 and predicted_rsi > 35:
        return "BUY - Oversold recovery expected"
    elif current_rsi > 70 and predicted_rsi < 65:
        return "SELL - Overbought correction expected"
    else:
        return "HOLD - Stable RSI expected"
```

### Real-time Prediction

```python
def predict_live_rsi(live_data):
    model, features = load_rsi_model()
    processed_data = transform_rsi_features(live_data)
    
    # Extract and predict
    X = processed_data[features['important_features']]
    X.columns = [f"f{i}" for i in range(len(features['important_features']))]
    
    return model.predict(X.astype('float32'))[-1]  # Latest prediction
```

## 📋 Data Requirements

### Input CSV Format

Required columns in `accumulatedData_*.csv`:

```csv
open_time,open,high,low,close,volume,atr,fast_ema,slow_ema,adx,volatility_percent,...
2024-01-01,50000,51000,49000,50500,1000,500,50200,50100,25,2.5,...
```

**Essential Columns:**
- `open_time`: Timestamp
- `open`, `high`, `low`, `close`, `volume`: OHLCV data
- `atr`: Average True Range
- `fast_ema`, `slow_ema`: Exponential moving averages
- `adx`: Average Directional Index
- `volatility_percent`: Price volatility measure

## 🧪 Testing & Validation

### Model Evaluation

```bash
python RSIPredictionTest.py
```

**Outputs:**
- Performance metrics (MAE, RMSE, R²)
- Error analysis by RSI ranges
- Visual evaluation plots
- Recent prediction examples

### Performance Metrics

- **Overall MAE**: 4.84 RSI points
- **Neutral Range (30-70) MAE**: 4.63 points
- **Max Error**: 15.41 points
- **Standard Error**: 3.39 points

## 🚀 Production Deployment

### Model Serving

```python
# Load production model
import joblib
model = joblib.load("rsi_predictor_model.pkl")
features = joblib.load("rsi_model_features.pkl")

# Prediction pipeline
def predict_rsi(market_data):
    processed = transform_rsi_features(market_data)
    X = processed[features['important_features']]
    X.columns = [f"f{i}" for i in range(len(features['important_features']))]
    return model.predict(X.astype('float32'))
```

### Monitoring & Updates

- **Retraining**: Monthly with new market data
- **Feature Drift**: Monitor feature importance changes
- **Performance Tracking**: MAE threshold alerts
- **Data Quality**: Validate input data completeness

## 🔮 Future Enhancements

### Planned Improvements

1. **ONNX Export**: Resolve CatBoost → ONNX conversion
2. **Ensemble Methods**: XGBoost + LightGBM stacking
3. **Multi-timeframe**: 4H, 1D, 1W RSI predictions
4. **Advanced Features**: Market sentiment, order book data
5. **Real-time API**: WebSocket integration for live predictions

### Scalability Considerations

- **Feature Store**: Centralized feature management
- **Model Registry**: Version control for models
- **Batch Processing**: Spark for large datasets
- **Caching**: Redis for frequent predictions

## 🤝 Contributing

### Development Workflow

1. **Feature Engineering**: Add new indicators in `transform_rsi_features()`
2. **Model Improvements**: Experiment in `train_rsi_predictor()`
3. **Validation**: Update tests in `RSIPredictionTest.py`
4. **Documentation**: Update usage examples

### Code Quality

- Follow existing patterns from `Predictor.py` and `ML.py`
- Use `TimeSeriesSplit` for temporal validation
- Export models in PKL format (ONNX when supported)
- Include both accuracy and financial metrics

## 📄 License

This RSI prediction system follows the same patterns and conventions as the existing codebase. Refer to the main project license for usage terms.

## 🆘 Troubleshooting

### Common Issues

**TA-Lib Installation:**
```bash
# macOS
brew install ta-lib
pip install TA-Lib

# Ubuntu
sudo apt-get install libta-lib-dev
pip install TA-Lib
```

**Memory Issues:**
- Reduce dataset size for initial testing
- Use feature selection to reduce dimensionality
- Consider batch processing for large datasets

**Low Accuracy:**
- Add more technical indicators
- Increase training data volume
- Experiment with different time windows
- Consider ensemble methods

**Model Loading Errors:**
- Ensure virtual environment is activated
- Check file paths for model files
- Verify all dependencies are installed

---

**Built with ❤️ for quantitative trading and financial ML applications**