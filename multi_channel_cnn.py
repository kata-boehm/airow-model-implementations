import numpy as np
import pandas as pd
from dataclasses import dataclass
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

@dataclass
class MultiChannelConfig:
    """Configuration for multi-channel CNN."""
    random_seed: int = 123
    
    # Feature engineering (same as before)
    short_window: int = 30
    medium_window: int = 60
    long_window: int = 120
    
    # CNN specific
    segment_length: int = 300
    segment_overlap: int = 150
    
    # Post-processing
    peak_distance: int = 30
    peak_prominence: float = 0.1
    smooth_sigma: float = 5.0
    
    # Feature groups (based on XGBoost importance)
    feature_groups: dict = None
    
    def __post_init__(self):
        """Define semantic feature groups based on XGBoost insights."""
        self.feature_groups = {
            'raw_and_smooth': [
                'hr_raw',
                'hr_smooth_short',
                'hr_smooth_medium', 
                'hr_smooth_long'
            ],
            'trends': [
                'hr_increasing',
                'hr_decreasing',
                'hr_stable',
                'hr_diff_1s',
                'hr_diff_5s',
                'hr_diff_10s',
                'hr_diff_30s',
                'hr_accel'
            ],
            'baseline_deviations': [
                'hr_vs_short_baseline',
                'hr_vs_medium_baseline',
                'hr_vs_long_baseline'
            ],
            'rolling_stats': [
                'hr_std_short',
                'hr_min_short',
                'hr_max_short',
                'hr_range_short',
                'hr_std_medium',
                'hr_min_medium',
                'hr_max_medium'
            ],
            'temporal_context': [
                'hr_lag_30s',
                'hr_lag_60s',
                'hr_lag_90s',
                'hr_lead_30s',
                'hr_lead_60s',
                'hr_change_past_to_future'
            ]
        }


def create_hr_features_grouped(df: pd.DataFrame, config: MultiChannelConfig) -> tuple:
    """Create features and return them grouped by semantic meaning."""
    import numpy as np
    
    features = pd.DataFrame(index=df.index)
    hr = df['heart_rate'].values
    
    # Raw and smoothed HR
    features['hr_raw'] = hr
    features['hr_smooth_short'] = df['heart_rate'].rolling(window=config.short_window, center=True, min_periods=1).mean()
    features['hr_smooth_medium'] = df['heart_rate'].rolling(window=config.medium_window, center=True, min_periods=1).mean()
    features['hr_smooth_long'] = df['heart_rate'].rolling(window=config.long_window, center=True, min_periods=1).mean()
    
    # Derivatives
    features['hr_diff_1s'] = df['heart_rate'].diff(1)
    features['hr_diff_5s'] = df['heart_rate'].diff(5)
    features['hr_diff_10s'] = df['heart_rate'].diff(10)
    features['hr_diff_30s'] = df['heart_rate'].diff(30)
    features['hr_accel'] = features['hr_diff_1s'].diff(1)
    
    # Rolling statistics
    features['hr_std_short'] = df['heart_rate'].rolling(window=config.short_window, center=True, min_periods=1).std()
    features['hr_min_short'] = df['heart_rate'].rolling(window=config.short_window, center=True, min_periods=1).min()
    features['hr_max_short'] = df['heart_rate'].rolling(window=config.short_window, center=True, min_periods=1).max()
    features['hr_range_short'] = features['hr_max_short'] - features['hr_min_short']
    features['hr_std_medium'] = df['heart_rate'].rolling(window=config.medium_window, center=True, min_periods=1).std()
    features['hr_min_medium'] = df['heart_rate'].rolling(window=config.medium_window, center=True, min_periods=1).min()
    features['hr_max_medium'] = df['heart_rate'].rolling(window=config.medium_window, center=True, min_periods=1).max()
    
    # Relative features
    features['hr_vs_short_baseline'] = hr - features['hr_smooth_short']
    features['hr_vs_medium_baseline'] = hr - features['hr_smooth_medium']
    features['hr_vs_long_baseline'] = hr - features['hr_smooth_long']
    
    # Trends (CRITICAL per XGBoost)
    features['hr_increasing'] = (features['hr_diff_10s'] > 0).astype(int)
    features['hr_decreasing'] = (features['hr_diff_10s'] < 0).astype(int)
    features['hr_stable'] = (features['hr_diff_10s'].abs() < 1).astype(int)
    
    # Lagged and lead (temporal context)
    features['hr_lag_30s'] = df['heart_rate'].shift(30)
    features['hr_lag_60s'] = df['heart_rate'].shift(60)
    features['hr_lag_90s'] = df['heart_rate'].shift(90)
    features['hr_lead_30s'] = df['heart_rate'].shift(-30)
    features['hr_lead_60s'] = df['heart_rate'].shift(-60)
    features['hr_change_past_to_future'] = features['hr_lead_60s'] - features['hr_lag_60s']
    
    # Clean
    features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    for col in features.columns:
        if features[col].std() > 0:
            q99, q01 = features[col].quantile(0.99), features[col].quantile(0.01)
            features[col] = features[col].clip(q01, q99)
    
    # Group features
    grouped_features = {}
    for group_name, feature_names in config.feature_groups.items():
        grouped_features[group_name] = features[feature_names].values
    
    return features, grouped_features


def prepare_multichannel_segments(sessions, config: MultiChannelConfig):
    """Prepare segments with features organized by semantic groups."""
    seg_len = config.segment_length
    overlap = config.segment_overlap
    stride = seg_len - overlap
    
    # Dictionary to store segments for each feature group
    all_segments = {group: [] for group in config.feature_groups.keys()}
    all_y = []
    
    for session in sessions:
        # Get grouped features for this session
        _, grouped_features = create_hr_features_grouped(session['df'], config)
        y = session['y']
        
        # Create segments
        n_segments = (len(y) - seg_len) // stride + 1
        for i in range(n_segments):
            start, end = i * stride, i * stride + seg_len
            if end <= len(y):
                # Store segment for each feature group
                for group_name, features in grouped_features.items():
                    all_segments[group_name].append(features[start:end])
                all_y.append(y[start:end])
    
    # Convert to numpy arrays
    X_grouped = {
        group: np.array(segments) 
        for group, segments in all_segments.items()
    }
    y_array = np.array(all_y)
    
    return X_grouped, y_array


def build_multichannel_cnn(config: MultiChannelConfig, input_shapes: dict):
    """
    Build multi-channel CNN with parallel branches for each feature group.
    
    Architecture:
    - Separate Conv1D branch for each feature group
    - Each branch learns patterns specific to that feature type
    - Concatenate learned representations
    - Final layers for classification
    """
    
    # Input layers for each feature group
    inputs = {}
    branches = {}
    
    for group_name, shape in input_shapes.items():
        inputs[group_name] = layers.Input(shape=shape, name=f'input_{group_name}')
    
    # Branch 1: Raw and Smoothed HR (temporal evolution)
    x = inputs['raw_and_smooth']
    x = layers.Conv1D(32, 9, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, 21, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    branches['raw_and_smooth'] = x
    
    # Branch 2: Trends (direction and velocity)
    x = inputs['trends']
    x = layers.Conv1D(32, 9, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, 21, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    branches['trends'] = x
    
    # Branch 3: Baseline Deviations (intensity relative to context)
    x = inputs['baseline_deviations']
    x = layers.Conv1D(24, 9, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(24, 21, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    branches['baseline_deviations'] = x
    
    # Branch 4: Rolling Statistics (variability patterns)
    x = inputs['rolling_stats']
    x = layers.Conv1D(24, 9, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(24, 21, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    branches['rolling_stats'] = x
    
    # Branch 5: Temporal Context (past-future relationships)
    x = inputs['temporal_context']
    x = layers.Conv1D(32, 9, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, 21, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    branches['temporal_context'] = x
    
    # Concatenate all branches
    concatenated = layers.Concatenate()([branches[g] for g in config.feature_groups.keys()])
    
    # Fusion layers
    x = layers.Conv1D(128, 21, padding='same', activation='relu')(concatenated)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(64, 45, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(32, 1, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    output = layers.Conv1D(1, 1, activation='sigmoid', name='output')(x)
    
    # Create model
    model = models.Model(inputs=list(inputs.values()), outputs=output)
    
    return model


def train_multichannel_cnn(X_train_grouped, y_train, config: MultiChannelConfig):
    """Train the multi-channel CNN."""
    
    # Calculate class weights
    pos_weight = (y_train.size - y_train.sum()) / y_train.sum()
    print(f"Positive weight: {pos_weight:.2f}")
    
    # Define weighted loss
    def weighted_loss(y_true, y_pred):
        bce = keras.backend.binary_crossentropy(y_true, y_pred)
        weight_map = y_true * (pos_weight - 1) + 1
        return keras.backend.mean(bce * weight_map)
    
    # Get input shapes
    input_shapes = {
        group: (config.segment_length, X_train_grouped[group].shape[2])
        for group in config.feature_groups.keys()
    }
    
    print("\nInput shapes per group:")
    for group, shape in input_shapes.items():
        print(f"  {group:25s}: {shape}")
    
    # Build model
    model = build_multichannel_cnn(config, input_shapes)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=weighted_loss,
        metrics=['accuracy']
    )
    
    print(f"\n📊 Model summary:")
    print(f"Total parameters: {model.count_params():,}")
    
    # Prepare inputs as list (in correct order)
    X_train_list = [X_train_grouped[group] for group in config.feature_groups.keys()]
    
    # Train
    print("\n🏋️ Training multi-channel CNN...")
    history = model.fit(
        X_train_list, y_train,
        epochs=25,
        batch_size=32,
        validation_split=0.15,
        callbacks=[
            keras.callbacks.EarlyStopping('val_loss', patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ],
        verbose=1
    )
    
    return model, history


def predict_multichannel_cnn(model, session, config: MultiChannelConfig):
    """Predict using multi-channel CNN with sliding window."""
    seg_len = config.segment_length
    overlap = config.segment_overlap
    stride = seg_len - overlap
    
    # Get grouped features
    _, grouped_features = create_hr_features_grouped(session['df'], config)
    
    # Initialize prediction arrays
    pred_sum = np.zeros(len(session['y']))
    pred_cnt = np.zeros(len(session['y']))
    
    # Sliding window prediction
    n_segments = (len(session['y']) - seg_len) // stride + 1
    for i in range(n_segments):
        start, end = i * stride, i * stride + seg_len
        if end <= len(session['y']):
            # Prepare input for this segment (one sample per group)
            segment_inputs = [
                grouped_features[group][start:end].reshape(1, seg_len, -1)
                for group in config.feature_groups.keys()
            ]
            
            # Predict
            pred = model.predict(segment_inputs, verbose=0)[0, :, 0]
            
            # Accumulate
            pred_sum[start:end] += pred
            pred_cnt[start:end] += 1
    
    # Average overlapping predictions
    probabilities = np.divide(pred_sum, pred_cnt, where=pred_cnt > 0)
    
    return probabilities


# Example usage template
if __name__ == "__main__":
    print("Multi-channel CNN module loaded!")
    print("\nTo use this:")
    print("1. Replace create_hr_features() with create_hr_features_grouped()")
    print("2. Replace prepare_cnn_segments() with prepare_multichannel_segments()")
    print("3. Use train_multichannel_cnn() instead of model.fit()")
    print("4. Use predict_multichannel_cnn() instead of predict_cnn()")
