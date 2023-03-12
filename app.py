import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle

# Load the data from the CSV file
data = pd.read_csv('data.csv')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Extract the features using TF-IDF vectorizer
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(train_data['utterance'])
test_features = vectorizer.transform(test_data['utterance'])

# Train the model using Linear SVM classifier
model = LinearSVC()
model.fit(train_features, train_data['intent'])

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the vectorizer to a file
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

# Define a function to predict the intent of a given utterance
# Define a function to predict the intent of a given utterance
def predict_intent(utterance):
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    features = vectorizer.transform([utterance])
    intent = model.predict(features)[0]
    if intent == 'fallback':
        return 'I am sorry, I did not understand your question.'
    else:
        return intent
print('Hello! I am a chatbot. How can I help you today?')

while True:
    user_input = input('You: ')
    intent = predict_intent(user_input)
    if intent == 'cancel_order':
        print('Chatbot: Hi there!')
    elif intent == 'goodbye':
        print('Chatbot: Goodbye!')
        break
    elif intent == 'thankyou':
        print('Chatbot: You\'re welcome!')
    else:
        print('Chatbot: Sorry, I didn\'t understand what you meant.')
