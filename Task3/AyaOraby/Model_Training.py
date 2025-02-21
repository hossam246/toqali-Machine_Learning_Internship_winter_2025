import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = "Hotel_dataset.csv"
df = pd.read_csv(file_path)

# Data Preprocessing
df.columns = df.columns.str.strip() 
df.drop(columns=['Booking_ID', 'date of reservation'], inplace=True, errors='ignore')

# Encode target variable
df['booking status'] = df['booking status'].map({'Not_Canceled': 0, 'Canceled': 1})

# Encode categorical features using one-hot encoding
categorical_features = ['type of meal', 'room type', 'market segment type']
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Define features (X) and target (y)
X = df.drop(columns=['booking status'])
y = df['booking status']

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save trained model and feature names
model_data = {"model": model, "feature_names": X.columns.tolist()}
pickle.dump(model_data, open("model.pkl", "wb"))

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
