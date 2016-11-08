#!/usr/bin/python

import numpy
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector
import os
import preprocess

#Fixed random seed for reproducability. 
numpy.random.seed(10)

#Sequence batch length. 
traj_seq_len = 18
word_seq_len = 18
num_h = 50
traj_dim = 8

#Get training data. 
dirs = []
path = '../dataset/robobarista_dataset/dataset/' 

#Extract all directories. 
for file_name in os.listdir(path):
    if os.path.isdir(path + file_name):
        dirs.append(file_name)

train = dirs[:20]
test = dirs[20:40]

train = preprocess.load_data_set(path, train)
test = preprocess.load_data_set(path, test)

train_l = train[2]
train_t = train[0]
original_text = train_l

vocab = preprocess.create_vocabulary(train_l)
train_l = preprocess.to_one_hot(vocab, train_l, word_seq_len)

#Pad trajectories. 
train_t = preprocess.pad_trajectory_sequences(train_t, traj_seq_len)

#Instantiate model. 
model = Sequential()

#LSTM for encoding trajectories into fixed dimensionality vector. 
model.add(LSTM(num_h, input_dim=traj_dim, input_length=traj_seq_len, return_sequences=False))

#Hands encoder output to decoder. 
model.add(RepeatVector(traj_seq_len))

#Decoder for taking trajectory vectors and generating natural language descriptions. 
model.add(LSTM(len(vocab.keys()), input_dim=num_h, input_length=word_seq_len, return_sequences=True))

#Compiles model for use. 
model.compile(loss='mse', optimizer='rmsprop')

#Trains model. 
model.fit(train_t, train_l, nb_epoch = 1000, batch_size = 80)

#Predicts from training. 
predictions =  model.predict_classes(train_t, batch_size = 1)

print predictions

#Converts predictions to phrases in natural language. 
predictions = preprocess.class_vectors_to_phrases(predictions, vocab)

for i in range(len(predictions)):
    print predictions[i] + ' :::: ' + original_text[i]
