from flask import Flask, request, jsonify
from model import train_model, predict_model
from utils import load_data, plot_data
import os

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    file_path = './data/' + file.filename
    file.save(file_path)
    return jsonify({'status': 'file uploaded', 'path': file_path})

@app.route('/train', methods=['POST'])
def train():
    data_path = request.json['data_path']
    features = request.json['features']
    target = request.json['target']
    model_type = request.json['model_type']
    model_name = request.json['model_name']

    data = load_data(data_path, features, target)
    metrics = train_model(data, features, target, model_type, model_name)
    return jsonify({'metrics': metrics})

@app.route('/predict', methods=['POST'])
def predict():
    data_path = request.json['data_path']
    features = request.json['features']
    target = request.json['target']
    model_name = request.json['model_name']
    model_type = request.json['model_type']  # Added model_type for plot_data

    data = load_data(data_path, features, target)
    predictions = predict_model(data, features, model_name)
    plot_url = plot_data(data, predictions, target, model_type)
    return jsonify({'plot_url': plot_url})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
