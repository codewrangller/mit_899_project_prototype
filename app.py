# app.py
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load models, scalers, and encoders for both decentralized and centralized models
decentralized_model = joblib.load('models/random_forest_decentralized_model.joblib')
decentralized_scaler = joblib.load('models/scaler_decentralized.joblib')
decentralized_le_protocol = joblib.load('models/label_encoder_decentralized_protocol.joblib')

centralized_model = joblib.load('models/random_forest_centralized_model.joblib')
centralized_scaler = joblib.load('models/scaler_centralized.joblib')
centralized_le_protocol = joblib.load('models/label_encoder_centralized_protocol.joblib')

# Load test data for both models
decentralized_test_data = pd.read_csv('simulated_p2p_data.csv')
centralized_test_data = pd.read_csv('dataset_phishing.csv')

def preprocess_data(data, model_type):
    if model_type == 'decentralized':
        features = [
            'source_port', 'destination_port', 'packet_size', 'connection_duration',
            'num_packets', 'avg_packet_interval', 'payload_entropy', 'suspicious_strings',
            'url_length', 'num_subdomains', 'ssl_cert_validity', 
            'protocol_encoded', 'source_ip_first_octet', 'dest_ip_first_octet'
        ]
        le = decentralized_le_protocol
        scaler = decentralized_scaler
        data['protocol_encoded'] = le.transform(data['protocol'])
        data['source_ip_first_octet'] = data['source_ip'].apply(lambda x: int(x.split('.')[0]))
        data['dest_ip_first_octet'] = data['destination_ip'].apply(lambda x: int(x.split('.')[0]))
        X = data[features]
        X_scaled = scaler.transform(X)
        y = data['label'].map({'phishing': 1, 'legitimate': 0})

    else:  # centralized
        features = [
            'length_url', 'length_hostname', 'nb_dots', 'nb_hyphens', 'nb_subdomains',
            'punycode', 'tld_in_path', 'tld_in_subdomain', 'abnormal_subdomain',
            'prefix_suffix', 'random_domain', 'shortening_service', 'ip', 'port',
            'nb_redirection', 'nb_external_redirection', 'dns_record', 'web_traffic',
            'ratio_digits_url', 'ratio_digits_host', 'char_repeat', 'phish_hints',
            'statistical_report', 'domain_registration_length', 'domain_age',
            'google_index', 'page_rank'
        ]
        le = centralized_le_protocol
        scaler = centralized_scaler
        X = data[features]
        X_scaled = scaler.transform(X)
        y = data['status']
        # Encode the target labels: 'phishing' as 1, 'legitimate' as 0
        y = le.fit_transform(y)
    
    return X_scaled, y

@app.route('/')
def home():
    return render_template('model_selection.html')

@app.route('/run_test/<model_type>')
def run_test(model_type):
    return render_template('index.html', model_type=model_type)

@app.route('/test', methods=['POST'])
def test_model():
    num_records = int(request.form['num_records'])
    model_type = request.form['model_type']
    
    if model_type == 'decentralized':
        model = decentralized_model
        test_data = decentralized_test_data
    else:
        model = centralized_model
        test_data = centralized_test_data
    
    sample_data = test_data.sample(n=num_records)
    
    X, y_true = preprocess_data(sample_data, model_type)
    y_pred = model.predict(X)
    
    accuracy = np.mean(y_pred == y_true)
    # Calculate metrics for legitimate packets
    legitimate_true = np.sum((y_true == 0) & (y_pred == 0))
    legitimate_false = np.sum((y_true == 0) & (y_pred == 1))
    
    # Calculate metrics for phishing packets
    phishing_true = np.sum((y_true == 1) & (y_pred == 1))
    phishing_false = np.sum((y_true == 1) & (y_pred == 0))
    
    results = {
        'accuracy': float(accuracy),
        'num_records': num_records,
        'predictions': y_pred.tolist(),
        'true_labels': y_true.tolist(),
        'legitimate_true': int(legitimate_true),
        'legitimate_false': int(legitimate_false),
        'phishing_true': int(phishing_true),
        'phishing_false': int(phishing_false)
    }
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)