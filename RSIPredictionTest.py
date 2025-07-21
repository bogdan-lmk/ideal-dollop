import pandas as pd
import numpy as np
import joblib
from RSIPredictor import transform_rsi_features, calculate_rsi
import matplotlib.pyplot as plt

def load_rsi_model():
    """
    Load trained RSI prediction model and feature mapping
    """
    try:
        model = joblib.load("models/rsi_predictor_model.pkl")
        feature_info = joblib.load("models/rsi_model_features.pkl")
        print("✅ RSI prediction model loaded successfully")
        return model, feature_info
    except FileNotFoundError:
        print("❌ Model files not found. Please train the model first using RSIPredictor.py")
        return None, None

def predict_next_day_rsi(data_file="accumulatedData_2024.csv", num_predictions=10):
    """
    Predict RSI for the next day using the trained model
    """
    model, feature_info = load_rsi_model()
    if model is None:
        return
    
    print(f"🔄 Loading data from {data_file}...")
    
    # Load and process data
    raw_df = pd.read_csv(data_file, parse_dates=["open_time"])
    processed_df = transform_rsi_features(raw_df)
    
    print(f"📊 Processed {len(processed_df)} data points")
    
    # Get the latest data points for prediction
    latest_data = processed_df.tail(num_predictions).copy()
    
    # Extract features using the same feature set as training
    important_features = feature_info['important_features']
    X_test = latest_data[important_features]
    
    # Rename columns to match training format
    X_test.columns = [f"f{i}" for i in range(len(important_features))]
    X_test = X_test.astype('float32')
    
    # Make predictions
    rsi_predictions = model.predict(X_test)
    
    # Display results
    print(f"\n🎯 RSI Predictions for Next Day:")
    print("=" * 60)
    
    for i, (idx, row) in enumerate(latest_data.iterrows()):
        actual_rsi = row['rsi_current']
        predicted_rsi = rsi_predictions[i]
        date = raw_df.iloc[idx]['open_time']
        
        print(f"Date: {date}")
        print(f"Current RSI: {actual_rsi:.2f}")
        print(f"Predicted Next Day RSI: {predicted_rsi:.2f}")
        print("-" * 40)
    
    return rsi_predictions, latest_data

def evaluate_rsi_model():
    """
    Evaluate the RSI prediction model on test data
    """
    model, feature_info = load_rsi_model()
    if model is None:
        return
    
    print("🔍 Evaluating RSI prediction model...")
    
    # Load and process data
    raw_df = pd.read_csv("accumulatedData_2024.csv", parse_dates=["open_time"])
    processed_df = transform_rsi_features(raw_df)
    
    # Use last 20% of data for evaluation
    test_size = int(len(processed_df) * 0.2)
    test_data = processed_df.tail(test_size).copy()
    
    # Prepare features
    important_features = feature_info['important_features']
    X_test = test_data[important_features]
    X_test.columns = [f"f{i}" for i in range(len(important_features))]
    X_test = X_test.astype('float32')
    
    # Get actual RSI values (next day)
    y_actual = test_data['rsi_next_day'].values
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = np.mean(np.abs(y_actual - y_pred))
    mse = np.mean((y_actual - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # RSI-specific metrics
    rsi_errors = np.abs(y_actual - y_pred)
    
    print(f"\n📊 Model Evaluation Results:")
    print("=" * 40)
    print(f"Test Data Points: {len(y_actual)}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Square Error: {rmse:.4f}")
    print(f"Max RSI Error: {np.max(rsi_errors):.2f}")
    print(f"Mean RSI Error: {np.mean(rsi_errors):.2f}")
    print(f"Std RSI Error: {np.std(rsi_errors):.2f}")
    
    # RSI range analysis
    overbought_errors = rsi_errors[y_actual > 70]
    oversold_errors = rsi_errors[y_actual < 30]
    neutral_errors = rsi_errors[(y_actual >= 30) & (y_actual <= 70)]
    
    print(f"\n📈 Error Analysis by RSI Range:")
    print(f"Overbought (>70): {len(overbought_errors)} points, Avg Error: {np.mean(overbought_errors) if len(overbought_errors) > 0 else 0:.2f}")
    print(f"Oversold (<30): {len(oversold_errors)} points, Avg Error: {np.mean(oversold_errors) if len(oversold_errors) > 0 else 0:.2f}")
    print(f"Neutral (30-70): {len(neutral_errors)} points, Avg Error: {np.mean(neutral_errors) if len(neutral_errors) > 0 else 0:.2f}")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Actual vs Predicted
    plt.subplot(2, 2, 1)
    plt.scatter(y_actual, y_pred, alpha=0.6, s=20)
    plt.plot([0, 100], [0, 100], 'r--', lw=2)
    plt.xlabel('Actual RSI')
    plt.ylabel('Predicted RSI')
    plt.title('Actual vs Predicted RSI')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Time series comparison
    plt.subplot(2, 2, 2)
    sample_size = min(200, len(y_actual))
    indices = range(sample_size)
    plt.plot(indices, y_actual[:sample_size], label='Actual', alpha=0.8)
    plt.plot(indices, y_pred[:sample_size], label='Predicted', alpha=0.8)
    plt.xlabel('Time')
    plt.ylabel('RSI')
    plt.title('RSI Prediction Time Series (Sample)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    plt.subplot(2, 2, 3)
    plt.hist(rsi_errors, bins=50, alpha=0.7, color='orange')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Feature importance
    plt.subplot(2, 2, 4)
    importances = pd.Series(model.feature_importances_, index=important_features)
    top_features = importances.nlargest(10)
    top_features.plot(kind='barh')
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importances')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rsi_model_evaluation.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Evaluation plots saved as 'rsi_model_evaluation.png'")
    plt.show()
    
    return mae, rmse, rsi_errors

def predict_rsi_for_new_data(close_prices, features_df=None):
    """
    Predict RSI for new data points
    
    Args:
        close_prices: Series or array of close prices
        features_df: Optional DataFrame with pre-calculated features
    """
    model, feature_info = load_rsi_model()
    if model is None:
        return None
    
    if features_df is None:
        # Calculate basic features if not provided
        df = pd.DataFrame({'close': close_prices})
        df['rsi_current'] = calculate_rsi(df['close'])
        # Add minimal required features (you may need to calculate more)
        df = df.dropna()
    else:
        df = features_df.copy()
    
    # Extract required features
    important_features = feature_info['important_features']
    available_features = [f for f in important_features if f in df.columns]
    
    if len(available_features) < len(important_features) * 0.8:  # At least 80% of features
        print(f"⚠️  Warning: Only {len(available_features)}/{len(important_features)} required features available")
        print("   Prediction quality may be reduced")
    
    X = df[available_features]
    
    # Pad missing features with zeros
    for missing_feature in set(important_features) - set(available_features):
        X[missing_feature] = 0
    
    # Reorder columns to match training
    X = X[important_features]
    X.columns = [f"f{i}" for i in range(len(important_features))]
    X = X.astype('float32')
    
    predictions = model.predict(X)
    return predictions

if __name__ == "__main__":
    print("🎯 RSI Prediction Testing Suite")
    print("=" * 50)
    
    # Test model loading
    model, features = load_rsi_model()
    if model is not None:
        print(f"📋 Model uses {len(features['important_features'])} features")
        
        # Evaluate model performance
        print("\n🔍 Running model evaluation...")
        evaluate_rsi_model()
        
        # Make predictions on recent data
        print("\n🎯 Making predictions on recent data...")
        predict_next_day_rsi()
        
    print("\n✅ RSI prediction testing completed!")