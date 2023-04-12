# chatbot.py
import numpy as np
import tensorflow as tf
import pickle

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the model
model = tf.keras.models.load_model('model6.h5')
model.summary()

# Get max_length from the model
max_length = model.layers[0].input_shape[1]

# Define the function to generate the response
def generate_response(question):
    question_seq = tokenizer.texts_to_sequences([question])[0]
    question_seq = tf.keras.preprocessing.sequence.pad_sequences([question_seq], maxlen=max_length, padding='post')
    prediction = model.predict(question_seq)[0]
    index = np.argmax(prediction, axis=-1)
    response = ' '.join([tokenizer.index_word[i] for i in index if i > 0])
    return response

# Test the chatbot
while True:
    question = input('You: ')
    if question == 'exit':
        break
    response = generate_response(question)
    print('Bot:', response)
