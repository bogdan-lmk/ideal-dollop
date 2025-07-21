"""
RSI Prediction Usage Examples
Demonstrates how to use the trained RSI prediction model in different scenarios
"""

import pandas as pd
import numpy as np
import joblib
from RSIPredictor import transform_rsi_features, calculate_rsi, calculate_macd
from RSIPredictionTest import load_rsi_model, predict_rsi_for_new_data

def load_and_predict_example():
    """
    Example 1: Load existing data and make predictions
    """
    print("🔹 Example 1: Predicting RSI on existing dataset")
    print("=" * 50)
    
    # Load model
    model, feature_info = load_rsi_model()
    if model is None:
        print("❌ Please train the model first using RSIPredictor.py")
        return
    
    # Load data
    df = pd.read_csv("accumulatedData_2024.csv", parse_dates=["open_time"])
    processed_df = transform_rsi_features(df)
    
    # Get latest 5 data points
    latest_data = processed_df.tail(5)
    
    # Extract features and predict
    features = latest_data[feature_info['important_features']]
    features.columns = [f"f{i}" for i in range(len(feature_info['important_features']))]
    predictions = model.predict(features.astype('float32'))
    
    print(f"🎯 Recent RSI Predictions:")
    for i, (idx, row) in enumerate(latest_data.iterrows()):
        date_str = str(df.iloc[idx]['open_time'])[:10] if idx < len(df) else f"Row {idx}"
        current_rsi = row['rsi_current']
        predicted_rsi = predictions[i]
        
        print(f"   {date_str}: Current={current_rsi:.1f} → Predicted={predicted_rsi:.1f}")
    
    return predictions

def new_data_prediction_example():
    """
    Example 2: Predict RSI for new price data
    """
    print("\n🔹 Example 2: Predicting RSI for new price data")
    print("=" * 50)
    
    # Simulate new OHLCV data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    
    # Generate realistic crypto price data
    base_price = 50000
    prices = []
    current_price = base_price
    
    for _ in range(50):
        # Random walk with some mean reversion
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        current_price *= (1 + change)
        prices.append(current_price)
    
    # Create OHLCV data
    new_data = pd.DataFrame({
        'open_time': dates,
        'close': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'open': [prices[i-1] if i > 0 else prices[i] for i in range(len(prices))],
        'volume': [np.random.uniform(1000, 5000) for _ in prices]
    })
    
    # Add minimal technical indicators (simplified for example)
    new_data['atr'] = 0.02 * new_data['close']  # Simplified ATR
    new_data['fast_ema'] = new_data['close'].ewm(span=12).mean()
    new_data['slow_ema'] = new_data['close'].ewm(span=26).mean()
    new_data['ema_fast_deviation'] = new_data['fast_ema'] - new_data['slow_ema']
    new_data['adx'] = 25  # Simplified ADX
    new_data['volatility_percent'] = new_data['close'].rolling(10).std() / new_data['close'].rolling(10).mean() * 100
    new_data['choppiness_index'] = 50  # Simplified
    new_data['positionBetweenBands'] = 0.5  # Simplified
    
    # Fill other required features with default values
    additional_features = {
        'atr_stop': new_data['atr'] * 2,
        'atr_to_price_ratio': new_data['atr'] / new_data['close'],
        'pchange': new_data['close'].pct_change() * 100,
        'avpchange': new_data['close'].pct_change().rolling(5).mean() * 100,
        'gma': new_data['close'].rolling(20).mean(),
        'rsi_volatility': 10,
        'rsi_delta': 0,
        'linear_regression': new_data['close']
    }
    
    for col, values in additional_features.items():
        new_data[col] = values
    
    # Process features for RSI prediction
    try:
        processed_new_data = transform_rsi_features(new_data)
        
        if len(processed_new_data) > 0:
            # Make predictions using the utility function
            predictions = predict_rsi_for_new_data(
                new_data['close'], 
                processed_new_data
            )
            
            if predictions is not None:
                print(f"🎯 RSI Predictions for simulated data:")
                for i, pred in enumerate(predictions[-5:]):  # Show last 5
                    date = dates[-(len(predictions)-i)]
                    current_rsi = processed_new_data['rsi_current'].iloc[-(len(predictions)-i)]
                    print(f"   {date.strftime('%Y-%m-%d')}: Current={current_rsi:.1f} → Predicted={pred:.1f}")
            else:
                print("❌ Prediction failed - insufficient features")
        else:
            print("❌ No data available after processing")
            
    except Exception as e:
        print(f"❌ Error processing new data: {e}")
        print("   Note: New data requires all technical indicators to be pre-calculated")

def model_analysis_example():
    """
    Example 3: Analyze model features and importance
    """
    print("\n🔹 Example 3: Model Analysis")
    print("=" * 50)
    
    model, feature_info = load_rsi_model()
    if model is None:
        return
    
    # Feature importance analysis
    importances = pd.Series(model.feature_importances_, 
                           index=feature_info['important_features'])
    
    print("📊 Top 10 Most Important Features for RSI Prediction:")
    top_features = importances.nlargest(10)
    for i, (feature, importance) in enumerate(top_features.items(), 1):
        print(f"   {i:2d}. {feature:<20}: {importance:6.2f}%")
    
    # Model metadata
    print(f"\n📋 Model Information:")
    print(f"   Total Features: {len(feature_info['important_features'])}")
    print(f"   Model Type: CatBoostRegressor")
    print(f"   Training Features: {', '.join(feature_info['important_features'][:5])}...")
    
    return importances

def batch_prediction_example():
    """
    Example 4: Batch predictions for multiple time periods
    """
    print("\n🔹 Example 4: Batch Predictions")
    print("=" * 50)
    
    model, feature_info = load_rsi_model()
    if model is None:
        return
    
    # Load full dataset
    df = pd.read_csv("accumulatedData_2024.csv", parse_dates=["open_time"])
    processed_df = transform_rsi_features(df)
    
    if len(processed_df) == 0:
        print("❌ No processed data available")
        return
    
    # Make predictions for all available data
    features = processed_df[feature_info['important_features']]
    features.columns = [f"f{i}" for i in range(len(feature_info['important_features']))]
    all_predictions = model.predict(features.astype('float32'))
    
    # Calculate prediction accuracy statistics
    actual_next_rsi = processed_df['rsi_next_day'].values
    errors = np.abs(all_predictions - actual_next_rsi)
    
    print(f"📊 Batch Prediction Results:")
    print(f"   Total Predictions: {len(all_predictions)}")
    print(f"   Mean Absolute Error: {np.mean(errors):.2f}")
    print(f"   Max Error: {np.max(errors):.2f}")
    print(f"   Min Error: {np.min(errors):.2f}")
    print(f"   Std Error: {np.std(errors):.2f}")
    
    # Accuracy by RSI ranges
    overbought_mask = actual_next_rsi > 70
    oversold_mask = actual_next_rsi < 30
    neutral_mask = (actual_next_rsi >= 30) & (actual_next_rsi <= 70)
    
    if np.sum(overbought_mask) > 0:
        print(f"   Overbought (>70) MAE: {np.mean(errors[overbought_mask]):.2f}")
    if np.sum(oversold_mask) > 0:
        print(f"   Oversold (<30) MAE: {np.mean(errors[oversold_mask]):.2f}")
    if np.sum(neutral_mask) > 0:
        print(f"   Neutral (30-70) MAE: {np.mean(errors[neutral_mask]):.2f}")
    
    return all_predictions, actual_next_rsi

def integration_example():
    """
    Example 5: Integration with trading strategy
    """
    print("\n🔹 Example 5: Trading Strategy Integration")
    print("=" * 50)
    
    model, feature_info = load_rsi_model()
    if model is None:
        return
    
    # Load recent data
    df = pd.read_csv("accumulatedData_2024.csv", parse_dates=["open_time"])
    processed_df = transform_rsi_features(df)
    
    if len(processed_df) < 10:
        print("❌ Insufficient data for strategy example")
        return
    
    # Get recent predictions
    recent_data = processed_df.tail(10)
    features = recent_data[feature_info['important_features']]
    features.columns = [f"f{i}" for i in range(len(feature_info['important_features']))]
    predictions = model.predict(features.astype('float32'))
    
    # Simple trading signals based on RSI predictions
    signals = []
    for i, (idx, row) in enumerate(recent_data.iterrows()):
        current_rsi = row['rsi_current']
        predicted_rsi = predictions[i]
        
        # Trading logic
        if current_rsi < 30 and predicted_rsi > 35:
            signal = "🟢 BUY - Oversold recovery expected"
        elif current_rsi > 70 and predicted_rsi < 65:
            signal = "🔴 SELL - Overbought correction expected"
        elif abs(predicted_rsi - current_rsi) < 2:
            signal = "⚪ HOLD - Stable RSI expected"
        elif predicted_rsi > current_rsi:
            signal = "🔵 WEAK BUY - RSI trending up"
        else:
            signal = "🟠 WEAK SELL - RSI trending down"
        
        signals.append(signal)
        
        date_str = str(df.iloc[idx]['open_time'])[:10] if idx < len(df) else f"Row {idx}"
        print(f"   {date_str}: RSI {current_rsi:.1f} → {predicted_rsi:.1f} | {signal}")
    
    return signals

if __name__ == "__main__":
    print("🚀 RSI Prediction Usage Examples")
    print("=" * 60)
    
    # Run all examples
    try:
        load_and_predict_example()
        new_data_prediction_example()
        model_analysis_example()
        batch_prediction_example()
        integration_example()
        
        print("\n✅ All examples completed successfully!")
        print("\n📝 Next Steps:")
        print("   1. Integrate predictions into your trading system")
        print("   2. Fine-tune the model with more recent data")
        print("   3. Add additional technical indicators for better accuracy")
        print("   4. Implement ensemble methods for robust predictions")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")