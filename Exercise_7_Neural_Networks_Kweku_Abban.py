import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

#-------------------------------------------------data preprocessing------------------------------------------------------------------

# Load the dataset
url = 'https://raw.githubusercontent.com/zhenliangma/Applied-AI-in-Transportation/master/Exercise_7_Neural_networks/Exercise7data.csv'
df = pd.read_csv(url)

# Limit the DataFrame to the first 1000 rows
df = df.iloc[:1000]

# Drop unnecessary columns
df = df.drop(['Arrival_time', 'Stop_id', 'Bus_id', 'Line_id'], axis=1)

# Split into features and target variable
x = df.drop(['Arrival_delay'], axis=1)
y = df['Arrival_delay']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#-------------------------------------------------Function to build and train model---------------------------------------------------

def build_and_train_model(units1, units2, dropout1, dropout2, epochs, batch_size):
    # Create a Sequential model
    model = Sequential()
    model.add(Dense(units1, activation='relu', input_shape=(4,)))  # Use input_shape instead of input_dim
    model.add(Dropout(dropout1))
    model.add(Dense(units2, activation='relu'))
    model.add(Dropout(dropout2))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])

    # Callbacks
    early_stop = EarlyStopping(monitor='val_mae', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=3)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_mae', save_best_only=True, mode='min')

    # Train the model and capture the history
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stop, reduce_lr, checkpoint], verbose=0)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mae, mse, r2, model, history

#-------------------------------------------------Hyperparameter Tuning------------------------------------------------------------------

# Define hyperparameter search space
units1_options = [32, 64]
units2_options = [64, 128]
dropout1_options = [0.2, 0.3]
dropout2_options = [0.3, 0.4]
epochs_options = [50, 100, 150, 200]
batch_size_options = [16, 32]

# Store the best model and metrics
best_mae = np.inf
best_model = None
best_params = {}
best_history = None

# Search through all combinations of hyperparameters
for units1 in units1_options:
    for units2 in units2_options:
        for dropout1 in dropout1_options:
            for dropout2 in dropout2_options:
                for epochs in epochs_options:
                    for batch_size in batch_size_options:
                        mae, mse, r2, model, history = build_and_train_model(units1, units2, dropout1, dropout2, epochs, batch_size)
                        if mae < best_mae:
                            best_mae = mae
                            best_mse = mse
                            best_r2 = r2
                            best_model = model
                            best_history = history
                            best_params = {
                                'units1': units1,
                                'units2': units2,
                                'dropout1': dropout1,
                                'dropout2': dropout2,
                                'epochs': epochs,
                                'batch_size': batch_size
                            }

# Print the best hyperparameters and evaluation metrics
print("Best model parameters:")
print(best_params)
print(f"Mean Absolute Error (MAE): {best_mae}")
print(f"Mean Squared Error (MSE): {best_mse}")
print(f"R-squared (R²): {best_r2}")

#-------------------------------------------------Plot Training and Validation Loss-------------------------------------------------

# Assuming the best model was trained in this run, we can plot the training history
sns.set()
plt.plot(best_history.history['mae'], label='Training MAE')
plt.plot(best_history.history['val_mae'], label='Validation MAE')
plt.title('Training and Validation MAE for Best Model')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend(loc='upper right')
plt.show()
