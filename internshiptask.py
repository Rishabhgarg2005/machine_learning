<<<<<<< HEAD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error
data = pd.read_csv("task1data.csv")

print(data.head())
print(data.isnull().sum())
data.info()

numeric_cols = ["PRCP", "TMAX", "TMIN", "TAVG"]  # Add more if needed, e.g., "AWND"
for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data[col] = data[col].fillna(data[col].mean())

if "DATE" in data.columns:
    data["DATE"] = pd.to_datetime(data["DATE"])
    data["Year"] = data["DATE"].dt.year
    data["Month"] = data["DATE"].dt.month
print(data.head())

data.drop("DATE", axis=1, inplace=True, errors='ignore')
data.drop("NAME", axis=1, inplace=True, errors='ignore')
data.drop("STATION", axis=1, inplace=True, errors='ignore')

def clean_attribute_value(value):
    if pd.isna(value):
        return 0  
    if value == ",,S":
        return 1
    if value == "H,,S":
        return 2
    if value == "D,,S":
        return 3
   
    if isinstance(value, str):
        return 99 
    return value 


attribute_cols = [col for col in data.columns if 'ATTRIBUTES' in col]

for col in attribute_cols:
    data[col] = data[col].apply(clean_attribute_value)
    
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int) 


numeric_columns = data.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    data[col] = data[col].fillna(data[col].mean())
sns.boxplot(data =data["PRCP"])
plt.show()


print(data.describe())
plt.figure(figsize=(10, 6))
q1 = data["PRCP"].quantile(0.25)
q3 = data["PRCP"].quantile(0.75)
iqr = q3 - q1
upper_bound = q3+ 3.0 * iqr
lower_bound = q1 - 3.0 * iqr          #removing the outliers in the data  
new_df =data.loc[(data["PRCP"]<lower_bound)|(data["PRCP"]>upper_bound)]
outliers_mask = (data["PRCP"] < lower_bound) | (data["PRCP"] > upper_bound)
print(f"Number of outliers: {outliers_mask.sum()}")
print(new_df.head())  


cleaned_data = data[~outliers_mask].copy()
print(f"Original shape: {data.shape}, Cleaned shape: {cleaned_data.shape}")
plt.figure(figsize=(10, 6))
plt.boxplot(data["PRCP"])
plt.title("Boxplot of PRCP (Outliers Visible)")
plt.ylabel("PRCP Value")
plt.show()  


if "TAVG" not in data.columns:
    raise ValueError("TAVG column not found - check data loading and cleaning.")
X = cleaned_data.drop("TAVG", axis=1, errors="ignore")
y = cleaned_data["TAVG"]

print("X shape:", X.shape if hasattr(X, 'shape') else len(X))
print("y shape:", y.shape if hasattr(y, 'shape') else len(y))
print("X head:", X.head() if hasattr(X, 'head') else X[:5])  # Assuming pandas DataFrame
print("y head:", y.head() if hasattr(y, 'head') else y[:5])


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=48)


scaler = MinMaxScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)


early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15, 
    restore_best_weights=True 
)


model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(Xtrain.shape[1],)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(1) 
])


model.compile(optimizer="adam", loss="mse")


history = model.fit(
    Xtrain,
    ytrain,
    validation_data=(Xtest, ytest),
    batch_size=32,
    epochs=300,
    callbacks=[early_stop],
    verbose=0
)


model.summary()


losses = pd.DataFrame(history.history)
print(losses.head())
losses[['loss', 'val_loss']].plot()
plt.title("Model Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.show()


predictions = model.predict(Xtest).flatten()  


mae = mean_absolute_error(ytest, predictions)
mse = mean_squared_error(ytest, predictions)
rmse = np.sqrt(mse)

print(f"\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


plt.scatter(ytest, predictions)
sorted_indices = np.argsort(ytest)
plt.plot(ytest.iloc[sorted_indices], ytest.iloc[sorted_indices], 'r--', label='Perfect Prediction') 
plt.xlabel("Actual TAVG")
plt.ylabel("Predicted TAVG")
plt.title("Actual vs. Predicted TAVG")
plt.legend()
plt.show()
new_data_point = Xtest[0].reshape(1, -1)
predicted_value = model.predict(new_data_point, verbose=0)  
print(f"Predicted TAVG for the first test data point: {predicted_value[0][0]:.2f}")
=======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error
data = pd.read_csv("task1data.csv")

print(data.head())
print(data.isnull().sum())
data.info()

numeric_cols = ["PRCP", "TMAX", "TMIN", "TAVG"]  # Add more if needed, e.g., "AWND"
for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data[col] = data[col].fillna(data[col].mean())

if "DATE" in data.columns:
    data["DATE"] = pd.to_datetime(data["DATE"])
    data["Year"] = data["DATE"].dt.year
    data["Month"] = data["DATE"].dt.month
print(data.head())

data.drop("DATE", axis=1, inplace=True, errors='ignore')
data.drop("NAME", axis=1, inplace=True, errors='ignore')
data.drop("STATION", axis=1, inplace=True, errors='ignore')

def clean_attribute_value(value):
    if pd.isna(value):
        return 0  
    if value == ",,S":
        return 1
    if value == "H,,S":
        return 2
    if value == "D,,S":
        return 3
   
    if isinstance(value, str):
        return 99 
    return value 


attribute_cols = [col for col in data.columns if 'ATTRIBUTES' in col]

for col in attribute_cols:
    data[col] = data[col].apply(clean_attribute_value)
    
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int) 


numeric_columns = data.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    data[col] = data[col].fillna(data[col].mean())
sns.boxplot(data =data["PRCP"])
plt.show()


print(data.describe())
plt.figure(figsize=(10, 6))
q1 = data["PRCP"].quantile(0.25)
q3 = data["PRCP"].quantile(0.75)
iqr = q3 - q1
upper_bound = q3+ 3.0 * iqr
lower_bound = q1 - 3.0 * iqr          #removing the outliers in the data  
new_df =data.loc[(data["PRCP"]<lower_bound)|(data["PRCP"]>upper_bound)]
outliers_mask = (data["PRCP"] < lower_bound) | (data["PRCP"] > upper_bound)
print(f"Number of outliers: {outliers_mask.sum()}")
print(new_df.head())  


cleaned_data = data[~outliers_mask].copy()
print(f"Original shape: {data.shape}, Cleaned shape: {cleaned_data.shape}")
plt.figure(figsize=(10, 6))
plt.boxplot(data["PRCP"])
plt.title("Boxplot of PRCP (Outliers Visible)")
plt.ylabel("PRCP Value")
plt.show()  


if "TAVG" not in data.columns:
    raise ValueError("TAVG column not found - check data loading and cleaning.")
X = cleaned_data.drop("TAVG", axis=1, errors="ignore")
y = cleaned_data["TAVG"]

print("X shape:", X.shape if hasattr(X, 'shape') else len(X))
print("y shape:", y.shape if hasattr(y, 'shape') else len(y))
print("X head:", X.head() if hasattr(X, 'head') else X[:5])  # Assuming pandas DataFrame
print("y head:", y.head() if hasattr(y, 'head') else y[:5])


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=48)


scaler = MinMaxScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)


early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15, 
    restore_best_weights=True 
)


model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(Xtrain.shape[1],)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(1) 
])


model.compile(optimizer="adam", loss="mse")


history = model.fit(
    Xtrain,
    ytrain,
    validation_data=(Xtest, ytest),
    batch_size=32,
    epochs=300,
    callbacks=[early_stop],
    verbose=0
)


model.summary()


losses = pd.DataFrame(history.history)
print(losses.head())
losses[['loss', 'val_loss']].plot()
plt.title("Model Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.show()


predictions = model.predict(Xtest).flatten()  


mae = mean_absolute_error(ytest, predictions)
mse = mean_squared_error(ytest, predictions)
rmse = np.sqrt(mse)

print(f"\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


plt.scatter(ytest, predictions)
sorted_indices = np.argsort(ytest)
plt.plot(ytest.iloc[sorted_indices], ytest.iloc[sorted_indices], 'r--', label='Perfect Prediction') 
plt.xlabel("Actual TAVG")
plt.ylabel("Predicted TAVG")
plt.title("Actual vs. Predicted TAVG")
plt.legend()
plt.show()
new_data_point = Xtest[0].reshape(1, -1)
predicted_value = model.predict(new_data_point, verbose=0)  
print(f"Predicted TAVG for the first test data point: {predicted_value[0][0]:.2f}")
>>>>>>> 8286c73d01dccfbce151db629022b0978f2cffb6
print(f"Actual TAVG for the first test data point: {ytest.iloc[0]:.2f}")