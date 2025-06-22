import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# Generate synthetic data
np.random.seed(42)

# Number of samples
n_samples = 5000

# Simulate features:
# Feature 1: IP Address (1 if same, 0 if different)
ip_address = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

# Feature 2: Port number (random value between 1024 and 65535)
port_number = np.random.randint(1024, 65536, size=n_samples)

# Feature 3: Host (1 for known host, 0 for unknown host)
host_type = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

# Feature 4: OTP bypass count (random count from 0 to 5)
otp_bypassed_count = np.random.randint(0, 6, size=n_samples)

# Simulate target variable (1 for fraud, 0 for legitimate):
# Fraud is more likely if OTP bypass attempts are higher and if it's a new IP or unknown host.
fraud_probability = (otp_bypassed_count > 3) | (ip_address == 0) | (host_type == 0)
fraud_label = np.random.randint(0, 2, size=n_samples)
fraud_label[fraud_probability] = 1

# Combine features and target into a DataFrame
df = pd.DataFrame({
    'ip_address': ip_address,
    'port_number': port_number,
    'host_type': host_type,
    'otp_bypassed_count': otp_bypassed_count,
    'label': fraud_label
})

# Feature selection
X = df.drop(columns=['label'])  # Features
y = df['label']  # Target variable (1 for fraud, 0 for legitimate)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model using Linear Regression
model = LinearRegression()

# Use Stratified KFold Cross-Validation to evaluate model performance
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')

# Train on full training data
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Convert predictions to binary (0 or 1) based on a threshold (0.5)
y_pred_binary = (y_pred >= 0.5).astype(int)

# Convert X_test back to a DataFrame with column names for easier manipulation
X_test_df = pd.DataFrame(X_test, columns=X.columns)

# Add predictions and true labels
X_test_df['predicted_label'] = y_pred_binary
X_test_df['true_label'] = y_test

# Map the numerical values back to their categorical names for readability
X_test_df['ip_address'] = X_test_df['ip_address'].map({0: 'Different IP', 1: 'Same IP'})
X_test_df['host_type'] = X_test_df['host_type'].map({0: 'Unknown Host', 1: 'Known Host'})

# Print a sample of the data with predictions and the features
print("Sample Predictions with IP, Port, and Host Info:\n", X_test_df[['ip_address', 'port_number', 'host_type', 'predicted_label', 'true_label']].head())

# Evaluation
print(f"Cross-Validation Mean Squared Error: {-np.mean(cross_val_scores):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_binary))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_binary))

# Visualization
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'] )
