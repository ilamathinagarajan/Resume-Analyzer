import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D, Bidirectional, Dropout, SimpleRNN
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import nltk
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Ensure you have the NLTK stopwords downloaded
nltk.download('stopwords', quiet=True)

# Load your dataset
data = pd.read_csv('C:/Users/Aswathy/OneDrive/Desktop/Github/Resume Analyser RNN LSTM/dataset.csv')

# Preprocessing the skills and recommendations
def preprocess_skills(skills):
    return ' '.join(eval(skills))

data['Resume Skills'] = data['Resume Skills'].apply(preprocess_skills)
data['Job Description Skills'] = data['Job Description Skills'].apply(preprocess_skills)

# Combine skills into one column for text processing
data['Combined Skills'] = data['Resume Skills'] + ' ' + data['Job Description Skills']

# Function to calculate match score
def calculate_match_score_and_missing(resume_skills, job_skills):
    resume_set = set(resume_skills.split())  # Convert string skills to set
    job_set = set(job_skills.split())  # Convert string skills to set
    
    common_skills = resume_set.intersection(job_set)
    missing_skills = job_set - resume_set  # Skills in job description but not in resume
    total_skills = resume_set.union(job_set)
    
    if not total_skills:  # Prevent division by zero
        match_score = 0.0
    else:
        match_score = len(common_skills) / len(total_skills)
    
    return match_score

# Apply the match score calculation to each row
data['Calculated Match Score'] = data.apply(lambda row: calculate_match_score_and_missing(row['Resume Skills'], row['Job Description Skills']), axis=1)

# Now, use 'Calculated Match Score' as the target variable (y)
y = data['Calculated Match Score']

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['Combined Skills'])
X = tokenizer.texts_to_sequences(data['Combined Skills'])
X = pad_sequences(X)

# Prepare labels and recommendations
recommendations = data['Recommendations']

# Splitting the dataset
X_train, X_test, y_train, y_test, rec_train, rec_test = train_test_split(X, y, recommendations, test_size=0.2, random_state=77)

# Build the updated model with SimpleRNN
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=250))
model.add(SpatialDropout1D(0.3))
model.add(SimpleRNN(100, return_sequences=True))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(105, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(105)))
model.add(Dense(1, activation='linear'))

# Compile with a custom optimizer
optimizer = Adam(learning_rate=0.0003)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

# Learning rate scheduler and early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# Train the model and capture training history with early stopping
history = model.fit(X_train, y_train, epochs=100, batch_size=56, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model on the training set
train_predictions = model.predict(X_train)
train_mae = mean_absolute_error(y_train, train_predictions)
train_mse = mean_squared_error(y_train, train_predictions)

# Evaluate the model on the test set
test_predictions = model.predict(X_test)
test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)


# Print the results
print(f"Training Mean Absolute Error: {train_mae}")
print(f"Testing Mean Absolute Error: {test_mae}")

# Print training and testing accuracy
print(f"Training Accuracy: {1 - train_mae}") 
print(f"Testing Accuracy: {1 - test_mae}") 

#Visualize Training and Validation Loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


model.save('C:/Users/Aswathy/OneDrive/Desktop/Github/Resume Analyser RNN LSTM/rnn_lstm_model.keras')

import pickle
with open('C:\\Users\\Aswathy\\OneDrive\\Desktop\\Github\\Resume Analyser RNN LSTM\\tokenizer_rnn_lstm.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)