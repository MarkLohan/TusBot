# train_model.py
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# Load the CSV file
data = pd.read_csv('data/datasetahaon.csv', sep=";", encoding='utf8')

# Split the data into questions and answers
questions = data['Question'].values
answers = data['Answer'].values

texts = pd.concat([data['Question'], data['Answer']], axis=0).astype("str")

# Tokenize the text
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(texts)
vocab_size = len(tokenizer.word_index) + 1

# Save tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Convert text to sequences
question_sequences = tokenizer.texts_to_sequences(questions.astype("str"))
answer_sequences = tokenizer.texts_to_sequences(answers.astype("str"))

# Pad the sequences
max_length = max([len(seq) for seq in question_sequences])
question_sequences = tf.keras.preprocessing.sequence.pad_sequences(question_sequences, maxlen=max_length, padding='post')
answer_sequences = tf.keras.preprocessing.sequence.pad_sequences(answer_sequences, maxlen=max_length, padding='post')

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 128, input_length=max_length),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation='softmax'))
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(question_sequences, answer_sequences, epochs=1000)

# Save the model
model.save('model6.h5')
