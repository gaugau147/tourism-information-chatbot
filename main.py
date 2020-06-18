import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import json
import io
import os

stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]


def loadData(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)

    sentences = []
    labels = []

    for intent in data['intents']:
        sentences += intent['patterns']
        # for word in stopwords:
        #     if word in sentences[-1]:
        #         sentences[-1].replace(' '+word+' ', ' ')
        for i in range(len(intent['patterns'])):
            labels.append(intent['tag'])

    return sentences, labels

def preprocessData(sentences, labels, vocab_size, max_length):
    # tokenizer
    sentence_tokenizer = Tokenizer(num_words = vocab_size, oov_token='<OOV>')
    sentence_tokenizer.fit_on_texts(sentences)
    sentence_sequence = sentence_tokenizer.texts_to_sequences(sentences)
    padded_sentence = pad_sequences(sentence_sequence, maxlen=max_length, padding='post', truncating='post')
    training_sentence = np.array(padded_sentence)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    label_sequence = np.array(label_tokenizer.texts_to_sequences(labels))

    return sentence_tokenizer, training_sentence, label_tokenizer, label_sequence

def createModel(vocab_size, embedding_dim, max_length, output_size):
    # the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(max_length, return_sequences=True)),
        # tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8, activation='softmax'),
        tf.keras.layers.Dense(8, activation='softmax'),
        tf.keras.layers.Dense(output_size, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

def main():

    # hyper parameters
    vocab_size = 64
    embedding_dim = 16
    max_length = 16

    filepath = 'intents.json'

    # get the sentences and labels in numpy array
    sentences, labels = loadData(filepath)

    # process text data into pad_sequences
    sentence_tokenizer, training_sentence, label_tokenizer, label_sequence = preprocessData(sentences, labels, vocab_size, max_length)

    # create the model
    output_size = len(set(labels))+1
    model = createModel(vocab_size, embedding_dim, max_length, output_size)

    # start training
    num_epochs = 1000
    batch_size = 8
    model.fit(training_sentence, label_sequence, batch_size, num_epochs)

    # save the model
    model.save('model/model.h5')

    #save tokenizer for chat
    tokenizer_json = sentence_tokenizer.to_json()
    with io.open('model/sentence_tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    tokenizer_json = label_tokenizer.to_json()
    with io.open('model/label_tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

if __name__ == "__main__":
    main()
