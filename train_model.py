import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_csv('data/mbti.csv')
df['clean_text'] = df['posts'].str.replace(r'https?://\S+|www\.\S+', '', regex=True).str.lower()  # clean URLs

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['type'])

# Text vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['clean_text']).toarray()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PyTorch model
class PersonalityModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),  # input -> 256 hidden units
            nn.ReLU(),  # activation function
            nn.Linear(256, num_classes)  # 256 hidden -> 16 output classes (personality types)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize model, loss, and optimizer
model = PersonalityModel(input_size=1000, num_classes=16)  # 1000 features and 16 MBTI types
criterion = nn.CrossEntropyLoss()  # for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# Training loop
for epoch in range(10):
    model.train()  # make sure the model is in training mode
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    
    # Optional: Evaluate the model on the test set periodically (every 2 epochs)
    if (epoch + 1) % 2 == 0:
        model.eval()  # set model to evaluation mode
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            accuracy = accuracy_score(y_test, predicted.numpy())
            print(f'Epoch {epoch+1} Test Accuracy: {accuracy:.4f}')

# Save model and vectorizer
torch.save(model.state_dict(), 'model/mbti_model.pth')
joblib.dump(vectorizer, 'model/vectorizer.pkl')
joblib.dump(label_encoder, 'model/label_encoder.pkl')

print("Model and vectorizer saved successfully!")