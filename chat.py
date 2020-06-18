import tensorflow
from tensorflow import keras
from keras.preprocessing.text import Tokenizer, tokenizer_from_json 
from keras.preprocessing.sequence import pad_sequences
import json
import tensorflow as tf
import random
import numpy as np


def loadSentenceTokenizer(filepath):
    '''
    Load the sentences tokenizer
    '''
    with open(filepath) as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    return tokenizer

def loadLabelTokenizer(filepath):
    '''
    Load the label tokenizer
    '''
    with open(filepath) as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    return tokenizer

def process_input(input, max_len, tokenizer_filepath):
    '''
    Process the input and return the corresponding sequence
    '''
    input = [input]
    tokenizer = loadSentenceTokenizer(tokenizer_filepath)
    input_sequence = tokenizer.texts_to_sequences(input)
    padded_input = np.array(pad_sequences(input_sequence, maxlen=max_len, padding='post', truncating='post'))
    return padded_input

def loadResponses(data_filepath, label_tokenizer_filepath):
    '''
    Return a dict mapping from index of label to corresponding responses
    '''
    responses = {}
    with open(data_filepath) as file:
        data = json.load(file)

    label_tokenizer = loadLabelTokenizer(label_tokenizer_filepath)
    label_word_index = label_tokenizer.word_index

    for intent in data['intents']:
        tag_index = label_word_index[intent['tag']]
        responses[str(tag_index)] = intent['responses']
    return responses

def getResponse(input_seq):
    result = model.predict(input_seq)
    result_index = np.argmax(result)
    answer = random.choice(responses[str(result_index)])
    return answer

# Load the trained model instance
model = tf.keras.models.load_model('model/model.h5')
max_len = 16
tokenizer_filepath = 'model/sentence_tokenizer.json'
label_tokenizer_filepath = 'model/label_tokenizer.json'
data_filepath = 'intents.json'

# load the available responses for each tag into dictionary
responses = loadResponses(data_filepath, label_tokenizer_filepath)
# print('----------------------------------------')
# print("Bot started, type 'stop' to exit")
# start chat
# while True:
#     user_input = input("User: ")

#     if user_input.lower() == 'stop':
#         break

#     input_seq = process_input(user_input, max_len, tokenizer_filepath)

#     answer = getResponse(input_seq)

#     # print(result)
#     print('Bot: ' + answer)

