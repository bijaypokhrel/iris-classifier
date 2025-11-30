import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

from data_loader import load_iris_df
from preprocess import split_and_scale, save_scaler

def main():
    os.makedirs("models", exist_ok=True)
    df = load_iris_df()
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    # Save scaler
    save_scaler(scaler, "models/scaler.pkl")

    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print("CV accuracy mean:", cv_scores.mean())

    # Evaluate on test set
    y_pred = model.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump(model, "models/rf_model.pkl")
    print("Saved model to models/rf_model.pkl")

if __name__ == "__main__":
    main()
