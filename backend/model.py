from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import os

def train_model(data, features, target, model_type, model_name):
    X = data[features]
    y = data[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000)  # Adjust max_iter for convergence
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier()
    else:
        raise ValueError("Unsupported model type")

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    if model_type == 'linear_regression':
        metrics = mean_squared_error(y_val, predictions)
    else:
        metrics = accuracy_score(y_val, predictions)

    # Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, f'models/{model_name}.joblib')

    return metrics

def predict_model(data, features, model_name):
    X = data[features]
    model = joblib.load(f'models/{model_name}.joblib')
    predictions = model.predict(X)
    return predictions
