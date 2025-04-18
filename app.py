from flask import Flask, render_template, request
import torch
import joblib
import numpy as np
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = Flask(__name__)

# Define the model class (same as when you trained it)
class PersonalityModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Load the trained model, vectorizer, and label encoder
model = PersonalityModel(input_size=1000, num_classes=16)
model.load_state_dict(torch.load('model/mbti_model.pth'))  # Load the model weights
model.eval()  # Set the model to evaluation mode

vectorizer = joblib.load('model/vectorizer.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']  # Capture user input
        
        # Convert the user input into features
        input_vector = vectorizer.transform([user_input]).toarray()

        # Make the prediction
        with torch.no_grad():
            prediction = model(torch.tensor(input_vector, dtype=torch.float32))
            predicted_class = torch.argmax(prediction, dim=1).item()
        
        # Decode the predicted class to MBTI type
        predicted_mbti = label_encoder.inverse_transform([predicted_class])[0]
        
        return render_template('result.html', prediction=predicted_mbti)  # Pass prediction to the template

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)