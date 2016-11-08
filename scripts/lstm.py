#!/usr/bin/python

import numpy
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector
import os
import preprocess
import sys

def evaluate_with_meteor(results_file_name, ref_file_name, test_file_name):
    results_file = open(results_file_name, 'r')
    ref_file = open(ref_file_name, 'w')
    test_file = open(test_file_name, 'w')

    #Gets all files and encodes them for METEOR. 
    for line in results_file:
        prediction, ground_truth = line.split('::::')

        #Clean up formatting. 
        prediction = prediction.strip('. ').strip() + '.'
        ground_truth = ground_truth.strip()

        #Write it to reference and test files. 
        test_file.write(prediction.encode('UTF-8') + u'\n')
        ref_file.write(ground_truth.encode('UTF-8') + u'\n')

    ref_file.close()
    test_file.close()

def run_lstm(fold_num):
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

    folds = preprocess.get_folds_dictionary(path + 'folds.json')

    train = [name.encode('ascii') for name in folds[fold_num]['train']]
    test = [name.encode('ascii') for name in folds[fold_num]['test']]

    train = preprocess.load_data_set(path, train)
    test = preprocess.load_data_set(path, test)

    #Splits train and test into their parts. 
    train_t, train_p, train_l = train
    test_t, test_p, test_l = test

    #Prepare vocabulary and create one-hot vectors for training set. 
    vocab = preprocess.create_vocabulary(train_l)
    train_l = preprocess.to_one_hot(vocab, train_l, word_seq_len)

    #Pad trajectories. 
    train_t = preprocess.pad_trajectory_sequences(train_t, traj_seq_len)
    test_t = preprocess.pad_trajectory_sequences(test_t, traj_seq_len)

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
    predictions =  model.predict_classes(test_t, batch_size = 1)

    #Converts predictions to phrases in natural language. 
    predictions = preprocess.class_vectors_to_phrases(predictions, vocab)

    results_file = open('lstm_fold_' + str(fold_num) + '_test.txt', 'w')

    for i in range(len(predictions)):
        results_file.write(predictions[i] + ' :::: ' + test_l[i] + '\n')

    results_file.close()

def print_usage():
    print 'Run lstm training: ./lstm.py run_lstm [fold_num]'
    print 'Evaluate LSTM results file using METEOR: ./lstm.py evaluate [results_file] [ref_file] [test_file]'

if __name__ == '__main__':
    if not len(sys.argv) >= 2:
        print_usage()

    elif sys.argv[1] == 'run_lstm':
        if not len(sys.argv) == 3:
            print_usage()
        else:
            run_lstm(int(sys.argv[2]))

    elif sys.argv[1] == 'evaluate':
        if len(sys.argv) == 5:
            evaluate_with_meteor(sys.argv[2], sys.argv[3], sys.argv[4])
        else:
            print_usage()

    else:
        print_usage()
