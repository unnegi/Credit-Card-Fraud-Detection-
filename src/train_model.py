# src/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from imblearn.over_sampling import SMOTE
import os

print("--- Starting the training script ---")

# --- 1. Load Data ---
print("Loading data...")
df = pd.read_csv('data/creditcard.csv')

# --- 2. Preprocess Data ---
print("Preprocessing data...")
# Scale 'Amount' and 'Time' - create separate scalers for each column
# This ensures each column is scaled independently with its own mean and std
scaler_amount = StandardScaler()
scaler_time = StandardScaler()
df['scaled_amount'] = scaler_amount.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler_time.fit_transform(df['Time'].values.reshape(-1, 1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 3. Handle Class Imbalance with SMOTE ---
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# --- 4. Define and Train the Model ---
print("Defining and training the model...")
# Define individual models (cost-sensitive)
clf1 = LogisticRegression(class_weight='balanced', random_state=42, solver='liblinear')
clf2 = RandomForestClassifier(class_weight='balanced', random_state=42)

# Define the Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2)],
    voting='soft' # Use soft voting for better performance
)

# Train on the resampled data
voting_clf.fit(X_train_resampled, y_train_resampled)

# --- 5. Save the Model and Scalers ---
print("Saving model and scalers...")
# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)
joblib.dump(voting_clf, 'models/fraud_detection_model.pkl')
# Save both scalers - we need separate scalers for Amount and Time
joblib.dump({'amount': scaler_amount, 'time': scaler_time}, 'models/scaler.pkl')
print("Model and scalers saved successfully.")

# --- 6. Evaluate the Model ---
print("Evaluating the model...")
y_pred = voting_clf.predict(X_test)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Confusion matrix plot saved as confusion_matrix.png")

print("\n--- Training script finished successfully! ---")