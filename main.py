import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# Import custom data preparation pipeline
from prepare import prepare_rossmann_data

# Load raw training and store data
print("Loading Data")
train_raw = pd.read_csv('data/train.csv', low_memory=False)
store_raw = pd.read_csv('data/store.csv')

# Clean and prepare the dataset for modeling
print('Running data preparation')
df = prepare_rossmann_data(train_raw, store_raw, is_train=True)

# Create a time-based train/validation split
def create_timeseries_split(df, days_to_predict=42):
    df = df.sort_values('Date').reset_index(drop=True)
    max_date = df['Date'].max()
    cutoff_date = max_date - pd.Timedelta(days=days_to_predict)

    train_set = df[df['Date'] <= cutoff_date].copy()
    val_set = df[df['Date'] > cutoff_date].copy()

    return train_set, val_set

train_df, val_df = create_timeseries_split(df)

# Separate features and target variable
print("Separating features and targets...")
cols_to_drop = ['Sales', 'Customers', 'Date']

X_train = train_df.drop(columns=cols_to_drop)
y_train = np.log1p(train_df['Sales'])

X_val = val_df.drop(columns=cols_to_drop)
y_val = np.log1p(val_df['Sales'])

# Encode categorical features using one-hot encoding
print("Encoding categorical variables...")

# Categorical columns requiring encoding
categorical_cols = ['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval']

# Convert categorical columns to strings for consistent encoding
X_train[categorical_cols] = X_train[categorical_cols].astype(str)
X_val[categorical_cols] = X_val[categorical_cols].astype(str)

# Initialize one-hot encoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit on training data and transform both datasets
encoded_train = encoder.fit_transform(X_train[categorical_cols])
encoded_val = encoder.transform(X_val[categorical_cols])

# Generate encoded feature names
encoded_cols = encoder.get_feature_names_out(categorical_cols)

# Convert encoded arrays back to DataFrames
encoded_train_df = pd.DataFrame(
    encoded_train,
    columns=encoded_cols,
    index=X_train.index
)

encoded_val_df = pd.DataFrame(
    encoded_val,
    columns=encoded_cols,
    index=X_val.index
)

# Replace original categorical columns with encoded features
X_train = pd.concat(
    [X_train.drop(columns=categorical_cols), encoded_train_df],
    axis=1
)

X_val = pd.concat(
    [X_val.drop(columns=categorical_cols), encoded_val_df],
    axis=1
)

# Train the XGBoost regression model
print(f"Features ready. Training on {X_train.shape[1]} columns.")
print("Initializing XGBRegressor...")

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=42,
    n_jobs=-1
)

print("Training the model... (This may take a few minutes)")
model.fit(X_train, y_train)

print("Training complete!")

# Evaluate model performance on the validation set
print("Making predictions on the validation set...")

# Generate predictions in log-transformed space
val_predictions_log = model.predict(X_val)

# Convert predictions back to the original sales scale
y_pred_actual = np.expm1(val_predictions_log)
y_val_actual = np.expm1(y_val)

# Root Mean Square Percentage Error (RMSPE)
def calculate_rmspe(y_true, y_pred):
    # Prevent division by zero
    y_true_safe = np.where(y_true == 0, 1e-9, y_true)

    percentage_error = (y_true_safe - y_pred) / y_true_safe
    rmspe = np.sqrt(np.mean(percentage_error ** 2))

    return rmspe

final_score = calculate_rmspe(y_val_actual, y_pred_actual)

print(f"====================================")
print(f"Validation RMSPE Score: {final_score:.4f}")
print(f"====================================")