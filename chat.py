from app import predict_intent

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
