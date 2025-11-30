from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def split_and_scale(df, target_col='target', test_size=0.2, random_state=42):
    X = df.drop([target_col, 'target_name'], axis=1, errors='ignore')
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_scaler(scaler, path="models/scaler.pkl"):
    joblib.dump(scaler, path)
