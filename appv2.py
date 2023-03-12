import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
#from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random
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
def predict_intent(utterance):
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    features = vectorizer.transform([utterance])
    intent = model.predict(features)[0]
    if intent == 'fallback':
        responses = ['I am sorry, I did not understand your question.', 
                     'Can you please rephrase your question?', 
                     'I didn\'t quite get that. Can you try again?', 
                     'I\'m not sure I understand. Could you please clarify?', 
                     'I didn\'t catch that. Can you please repeat your question?']
        return random.choice(responses)
    else:
        return intent
   

#def greet():
   # responses = ['Hi there!', 'Hello!', 'Greetings!', 'Hey!']
    #return random.choice(responses)
#
# Define a function to provide help
#def provide_help():
   ## responses = ['How can I assist you?', 'What can I help you with?', 'How can I be of service?']
   # return random.choice(responses)

# Define a function to check order status
def check_order_status():
    responses = ['Please provide your order number.', 'May I have your order number?', 'What is your order number?']
    return random.choice(responses)

# Define a function to extract entities
#def extract_entities(utterance, intent):
 #   if intent == 'cancel_order':
        # Extract order number
        # You can use regular expressions or other techniques to extract the order number
       # return order_number
    #else:
       # return None

# Define a function to analyze sentiment
def analyze_sentiment(utterance):
    # TextBlob
    # blob = TextBlob(utterance)
    # sentiment = blob.sentiment

    # VADER
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(utterance)
    sentiment = 'neutral'
    if scores['compound'] > 0.5:
        sentiment = 'positive'
    elif scores['compound'] < -0.5:
        sentiment = 'negative'
    return sentiment, scores['compound']



# Define a function to suggest intents based on the user's previous input
def suggest_intents(previous_input):
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    features = vectorizer.transform([previous_input])
    predicted_intents = model.predict(features)
    unique_intents = list(set(predicted_intents))
    if len(unique_intents) > 1:
        return f"Did you mean {', '.join(unique_intents[:-1])} or {unique_intents[-1]}?"
    else:
        return f"Did you mean {unique_intents[0]}?"
# Define a function to generate a response based on the user's intent
def generate_response(intent, sentiment, score):
    if intent == 'greet':
        responses = ['Hi there!', 'Hello!', 'Greetings!', 'Hey!']
        return random.choice(responses)

    elif intent == 'help':
        responses = ['How can I assist you?', 'What can I help you with?', 'How can I be of service?']
        return random.choice(responses)

    elif intent == 'cancel_order':
        if sentiment == 'positive':
            return 'That\'s great to hear! How can I help you today?'
        elif sentiment == 'negative':
            return 'I am sorry to hear that. Can you please provide more information about your request?'
        else:
            return 'Hi there! How can I help you today?'

    elif intent == 'goodbye':
        return 'Goodbye!'

    elif intent == 'thankyou':
        if sentiment == 'positive':
            return 'You\'re welcome!'
        elif sentiment == 'negative':
            return 'I am sorry I couldn\'t be of more help. Is there anything else I can assist you with?'
        else:
            return 'You\'re welcome!'

    else:
        if sentiment == 'positive':
            return 'Thank you for your kind words! How can I assist you today?'
        elif sentiment == 'negative':
            return 'I am sorry to hear that. Can you please rephrase your request?'
        else:
            return 'Sorry, I didn\'t understand what you meant. Can you please rephrase your'


# Define a function to handle multi-turn conversations
def handle_conversation(user_input):
    print('Hello! I am a chatbot. How can I help you today?')
    prev_intent = None
    while True:
        user_input = input('You: ')
        sentiment, score = analyze_sentiment(user_input)
        intent = predict_intent(user_input)
        if intent == prev_intent:
            print('Chatbot: Please provide more information about your previous request.')
        else:
            response = generate_response(intent, sentiment, score)
            print('Chatbot:', response)
        prev_intent = intent
        if intent == 'goodbye':
            break

# Run the chatbot
handle_conversation('hi')
