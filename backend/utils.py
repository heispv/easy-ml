import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(file_path, features, target):
    data = pd.read_csv(file_path)
    data = data[features + [target]]
    return data

def plot_data(data, predictions, target, model_type):
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    plot_path = f'plots/plot_{target}.png'
    plt.figure(figsize=(10, 6))

    if model_type == 'linear_regression':
        plt.scatter(data[data.columns[0]], data[target], color='blue', label='Actual')
        plt.plot(data[data.columns[0]], predictions, color='red', label='Predicted', linewidth=2)
    elif model_type in ['logistic_regression', 'decision_tree']:
        plt.scatter(data[data.columns[0]], data[target], c=data[target], cmap='viridis', label='Actual')
        plt.scatter(data[data.columns[0]], predictions, color='red', label='Predicted', alpha=0.5)

    plt.xlabel('Features')
    plt.ylabel(target)
    plt.title(f'Actual vs. Predicted {model_type}')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()

    return plot_path
