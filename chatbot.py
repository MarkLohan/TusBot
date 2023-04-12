# chatbot.py
import numpy as np
import tensorflow as tf
import pickle

# Load the tokenizer
with open('tokens/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the model
model = tf.keras.models.load_model('models/model_user_updated.h5')
model.summary()

# Get max_length from the model
max_length = model.layers[0].input_shape[1]

def user_update_model(question, answer, epochs=5):
    question_seq = tokenizer.texts_to_sequences([question])[0]
    question_seq = tf.keras.preprocessing.sequence.pad_sequences([question_seq], maxlen=max_length, padding='post')
    answer_seq = tokenizer.texts_to_sequences([answer])[0]
    answer_seq = tf.keras.preprocessing.sequence.pad_sequences([answer_seq], maxlen=max_length, padding='post')
    model.fit(question_seq, np.expand_dims(answer_seq, axis=-1), epochs=epochs)

# Define the function to generate the response
def generate_response(question, min_confidence=0.5):
    question_seq = tokenizer.texts_to_sequences([question])[0]
    question_seq = tf.keras.preprocessing.sequence.pad_sequences([question_seq], maxlen=max_length, padding='post')
    prediction = model.predict(question_seq)[0]
    index = np.argmax(prediction, axis=-1)
    confidence = np.mean([prediction[i, idx] for i, idx in enumerate(index) if idx > 0])
    response = ' '.join([tokenizer.index_word[i] for i in index if i > 0])
    print(confidence)

    # Check if the confidence is above the threshold
    if confidence > min_confidence:
        return response
    else:
        return "I'm sorry, I don't have an answer for that."


# return response

# Test the chatbot
while True:
    question = input('You: ')
    if question == 'exit':
        break
    response = generate_response(question)
    print('Bot:', response)

    # Ask for user feedback
    feedback = input('Is this response good or bad? (y/n): ')
    if feedback == 'n':
        proper_answer = input('Please provide the correct response: ')
        user_update_model(question, proper_answer)

model.save('models/model_user_updated.h5')
print("model updated")