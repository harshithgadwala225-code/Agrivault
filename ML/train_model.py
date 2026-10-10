import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_model():
    print("Loading processed dataset...")
    df = pd.read_csv('data/processed_onion.csv')

    # Define feature set and target label these are the columns where the model takes the input 
    feature_cols = [
        'Price_Per_Kg',
        'Price_30D_Lag',
        'Price_Change_30D_Pct'
    ]
    
    X = df[feature_cols] # Information given to the model it has the input features
    y = df['Target_Should_Store'] # contains the answer/target that the model needs to learn (correct answer)

    # 80/20 train-test split with stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, # 20% of the data will be used for testing
        stratify=y, # It tries to maintain approximately the same proportion in all the proportions.
        random_state=42 # we use the random_state because we wanted to randomly split the data and it should be "Start from the random process from the same starting point (42) every time."
    )

    # Standardize feature metrics
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) # 
    X_test_scaled = scaler.transform(X_test) # Because the scaler should learn its scaling information only from the training data.

    # Fit Logistic Regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train) # Look at the examples (X) and their correct answers (y), and learn the pattern between them.

    # Evaluation
    y_pred = model.predict(X_test_scaled) # now we give the model the unseen test data.
    acc = accuracy_score(y_test, y_pred) # y_test means the correct answers and y_pred means the predicted answers by the model
    
    print(f"\n--- Model Results ---")
    print(f"Accuracy: {acc * 100:.2f}%\n") # .2f means show 2 digits after the decimal
    print("Classification Report:\n", classification_report(y_test, y_pred)) # It gives detailed information about how well the model predicts each class (Sell and Store).
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model and scaler inside the ML directory
    joblib.dump(model, 'ML/agrivault_model.pkl')
    joblib.dump(scaler, 'ML/scaler.pkl')
    print("\nModel saved as ML/agrivault_model.pkl and ML/scaler.pkl successfully!")

if __name__ == '__main__':
    train_model()