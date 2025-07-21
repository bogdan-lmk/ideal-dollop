import numpy as np
import pandas as pd
import talib
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.ensemble import StackingRegressor
import matplotlib.pyplot as plt
import joblib
import optuna
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import warnings
warnings.filterwarnings('ignore')

def calculate_rsi(close_prices, window=14):
    """
    Calculate RSI using talib for consistency
    """
    return talib.RSI(close_prices.values, timeperiod=window)

def calculate_macd(close_prices, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD using talib
    Returns: macd_line, macd_signal, macd_histogram
    """
    macd, macd_signal, macd_hist = talib.MACD(
        close_prices.values, 
        fastperiod=fast_period, 
        slowperiod=slow_period, 
        signalperiod=signal_period
    )
    return macd, macd_signal, macd_hist

def transform_rsi_features(df, rsi_window=14, bollinger_window=20):
    """
    Feature engineering for RSI prediction following project patterns.
    Calculates MACD, RSI, and other technical indicators for next-day RSI prediction.
    """
    # Convert comma decimals to numeric for existing features
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
    
    # Calculate current RSI (target for next day)
    df['rsi_current'] = calculate_rsi(df['close'], window=rsi_window)
    
    # Calculate MACD components
    macd, macd_signal, macd_hist = calculate_macd(df['close'])
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_histogram'] = macd_hist
    
    # Additional RSI-related features
    df['rsi_sma_5'] = df['rsi_current'].rolling(5).mean()
    df['rsi_sma_10'] = df['rsi_current'].rolling(10).mean()
    df['rsi_momentum'] = df['rsi_current'].diff()
    df['rsi_rate_of_change'] = df['rsi_current'].pct_change()
    
    # Price-based features following existing patterns
    df['close_diff'] = df['close'].diff()
    df['normalized_price_change'] = df['close_diff'] / df['close'].shift(1)
    df['rolling_mean_close'] = df['close'].rolling(bollinger_window).mean()
    df['rolling_std_close'] = df['close'].rolling(bollinger_window).std()
    
    # Volume features
    df['volume_sma'] = df['volume'].rolling(10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # MACD-based features
    df['macd_signal_diff'] = df['macd'] - df['macd_signal']
    df['macd_momentum'] = df['macd'].diff()
    
    # Create target: next day RSI
    df['rsi_next_day'] = df['rsi_current'].shift(-1)
    
    # Remove last row (no target) and handle NaN values
    df = df.iloc[:-1].copy()
    df = df.dropna()
    
    return df.reset_index(drop=True)

def objective(trial, X_train, y_train):
    """
    Optuna hyperparameter optimization for CatBoost
    """
    params = {
        "iterations": trial.suggest_int("iterations", 500, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "random_seed": 42,
        "verbose": False
    }
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    maes = []
    
    for train_idx, valid_idx in tscv.split(X_train):
        X_fold_train, X_fold_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_fold_train, y_fold_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        
        model = CatBoostRegressor(**params)
        model.fit(X_fold_train, y_fold_train)
        
        y_pred = model.predict(X_fold_valid)
        mae = mean_absolute_error(y_fold_valid, y_pred)
        maes.append(mae)
    
    return np.mean(maes)

def train_rsi_predictor():
    """
    Main training function for RSI prediction
    """
    print("🔄 Loading and processing data...")
    
    # Load data
    raw_df = pd.read_csv("accumulatedData_2024.csv", parse_dates=["open_time"])
    print(f"📊 Loaded {len(raw_df)} rows of data")
    
    # Feature engineering
    processed_df = transform_rsi_features(raw_df)
    print(f"✅ Processed {len(processed_df)} rows after feature engineering")
    
    # Select features for RSI prediction
    feature_cols = [
        # Price features
        'close', 'high', 'low', 'open', 'close_diff', 'normalized_price_change',
        'rolling_mean_close', 'rolling_std_close',
        
        # Technical indicators
        'atr', 'fast_ema', 'slow_ema', 'ema_fast_deviation', 'adx',
        'rsi_current', 'rsi_sma_5', 'rsi_sma_10', 'rsi_momentum', 'rsi_rate_of_change',
        
        # MACD features
        'macd', 'macd_signal', 'macd_histogram', 'macd_signal_diff', 'macd_momentum',
        
        # Volume features
        'volume', 'volume_sma', 'volume_ratio',
        
        # Other indicators
        'volatility_percent', 'choppiness_index', 'positionBetweenBands'
    ]
    
    # Filter available features
    available_features = [col for col in feature_cols if col in processed_df.columns]
    print(f"📋 Using {len(available_features)} features for prediction")
    
    X = processed_df[available_features]
    y = processed_df['rsi_next_day']
    
    # Convert to float32 for consistency
    X = X.astype('float32')
    y = y.astype('float32')
    
    print(f"🎯 Target RSI statistics:")
    print(f"   Mean: {y.mean():.2f}")
    print(f"   Std: {y.std():.2f}")
    print(f"   Min: {y.min():.2f}")
    print(f"   Max: {y.max():.2f}")
    
    # Time series validation
    print("\n🔍 Performing time series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train CatBoost model
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0
        )
        
        model.fit(X_train_fold, y_train_fold, eval_set=(X_test_fold, y_test_fold), use_best_model=True)
        y_pred_fold = model.predict(X_test_fold)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_fold, y_pred_fold)
        mse = mean_squared_error(y_test_fold, y_pred_fold)
        r2 = r2_score(y_test_fold, y_pred_fold)
        
        fold_scores.append(mae)
        print(f"   Fold {fold+1}: MAE={mae:.4f}, MSE={mse:.4f}, R²={r2:.4f}")
    
    print(f"📊 Average CV MAE: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    # Final model training
    print("\n🚀 Training final model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train main model
    best_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=100
    )
    
    best_model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
    
    # Feature importance analysis
    feature_importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n📈 Top 10 Most Important Features:")
    for i, (feature, importance) in enumerate(feature_importance.head(10).items(), 1):
        print(f"   {i:2d}. {feature}: {importance:.4f}")
    
    # Filter important features (threshold > 0.01)
    important_features = feature_importance[feature_importance > 0.01].index.tolist()
    print(f"\n🎯 Selected {len(important_features)} important features (threshold > 0.01)")
    
    X_filtered = X[important_features].copy()
    
    # Rename columns for ONNX compatibility
    feature_mapping = {old_name: f"f{i}" for i, old_name in enumerate(important_features)}
    X_filtered.columns = [f"f{i}" for i in range(len(important_features))]
    
    # Retrain on filtered features
    X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(
        X_filtered, y, test_size=0.2, shuffle=False
    )
    
    final_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=100
    )
    
    final_model.fit(X_train_filtered, y_train_filtered, 
                   eval_set=(X_test_filtered, y_test_filtered), use_best_model=True)
    
    # Final evaluation
    y_pred_final = final_model.predict(X_test_filtered)
    
    mae_final = mean_absolute_error(y_test_filtered, y_pred_final)
    mse_final = mean_squared_error(y_test_filtered, y_pred_final)
    r2_final = r2_score(y_test_filtered, y_pred_final)
    
    print(f"\n🎯 Final Model Performance:")
    print(f"   MAE: {mae_final:.4f}")
    print(f"   MSE: {mse_final:.4f}")
    print(f"   R²: {r2_final:.4f}")
    
    # RSI-specific metrics
    rsi_errors = np.abs(y_test_filtered - y_pred_final)
    print(f"   Max RSI Error: {np.max(rsi_errors):.2f}")
    print(f"   Mean RSI Error: {np.mean(rsi_errors):.2f}")
    
    # Save feature mapping for future use
    feature_info = {
        'feature_mapping': feature_mapping,
        'important_features': important_features,
        'model_features': list(X_filtered.columns)
    }
    
    # Save model and metadata
    joblib.dump(final_model, "models/rsi_predictor_model.pkl")
    joblib.dump(feature_info, "models/rsi_model_features.pkl")
    
    print("\n💾 Model saved as 'models/rsi_predictor_model.pkl'")
    print("💾 Feature info saved as 'models/rsi_model_features.pkl'")
    
    # Export to ONNX
    try:
        initial_type = [('float_input', FloatTensorType([None, X_filtered.shape[1]]))]
        onnx_model = convert_sklearn(final_model, initial_types=initial_type)
        
        with open("models/rsi_predictor_model.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        print("💾 Model exported to 'models/rsi_predictor_model.onnx'")
    except Exception as e:
        print(f"⚠️  ONNX export failed: {e}")
        print("   Model saved in PKL format only")
    
    return final_model, feature_info, (mae_final, mse_final, r2_final)

if __name__ == "__main__":
    print("🚀 Starting RSI Prediction Model Training")
    print("=" * 50)
    
    try:
        model, features, metrics = train_rsi_predictor()
        print("\n✅ RSI predictor training completed successfully!")
        print(f"📊 Final MAE: {metrics[0]:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise