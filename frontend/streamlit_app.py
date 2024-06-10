import streamlit as st
import requests

st.title('Machine Learning Interface')
st.write("Upload your dataset and choose your model type to begin.")

uploaded_file = st.file_uploader("Choose a CSV file")

model_type = st.selectbox('Choose model type', ['linear_regression', 'logistic_regression', 'decision_tree'])
features = st.text_input('Enter features separated by comma (e.g., feature1, feature2)')
target = st.text_input('Enter target column name')
model_name = st.text_input('Enter model name to save')

if uploaded_file is not None:
    st.write("Uploaded File Details:")
    st.write(uploaded_file.name)
    st.write(uploaded_file.size, "bytes")

    files = {'file': uploaded_file}
    upload_response = requests.post('http://backend:5000/upload', files=files)
    if upload_response.status_code == 200:
        data_path = upload_response.json()['path']
        st.success('File uploaded successfully!')

        if st.button('Train Model'):
            train_data = {
                'data_path': data_path,
                'features': features.split(','),
                'target': target,
                'model_type': model_type,
                'model_name': model_name
            }
            train_response = requests.post('http://backend:5000/train', json=train_data)
            if train_response.status_code == 200:
                metrics = train_response.json()
                st.write("Training completed successfully!")
                st.write("Model Metrics:", metrics)
            else:
                st.error("Error in model training.")

        if st.button('Predict'):
            predict_data = {
                'data_path': data_path,
                'features': features.split(','),
                'target': target,
                'model_name': model_name,
                'model_type': model_type
            }
            predict_response = requests.post('http://backend:5000/predict', json=predict_data)
            if predict_response.status_code == 200:
                plot_url = predict_response.json()['plot_url']
                st.success('Prediction completed successfully!')
                local_plot_path = plot_url.replace('/app/', '')
                st.image(local_plot_path)
            else:
                st.error("Error in prediction.")

st.info('To begin, upload a CSV file, input the necessary parameters, and click the buttons to train or predict.')
