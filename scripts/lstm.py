#!/scratch/cluster/rcorona/deep_learning_env/bin/python3.4

import sys
import numpy
from keras.models import Model, Sequential
from keras.layers import LSTM, RepeatVector, Dropout, Input, merge
import os
import preprocess
import pickle

kmeans_path = '/scratch/cluster/rcorona/Robot_Learning_Project/Robot-Learning-Project/Models/Baseline/kmeans_fold_' 

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

def run_lstm(fold_num, parameter):
    #Fixed random seed for reproducability. 
    numpy.random.seed(10)

    #Sequence batch length. 
    traj_seq_len = 18
    word_seq_len = 18
    traj_dim = 8

    #Get training data. 
    dirs = []
    path = '/scratch/cluster/rcorona/Robot_Learning_Project/robobarista_dataset/dataset/' 

    folds = preprocess.get_folds_dictionary(path + 'folds.json')

    train = [name for name in folds[fold_num]['train']]
    test = [name for name in folds[fold_num]['validation']]

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

    #Load point clouds using their file names.
    print('Loading pcs')
    kmeans_model = pickle.load(open(kmeans_path + str(fold_num) + '.p', 'rb'), encoding='latin1')

    train_p = preprocess.load_point_clouds(train_p, kmeans_model)
    test_p = preprocess.load_point_clouds(test_p, kmeans_model)

    #Instantiate encoder and decoder. 
    encoder = Sequential()
    decoder = Sequential()

    #Specifies inputs. 
    trajectory_input = Input(shape=(traj_seq_len, traj_dim))

    #LSTM for encoding trajectories into fixed dimensionality vector. 
    encoder.add(LSTM(len(vocab.keys()), input_shape=(traj_seq_len, traj_dim), return_sequences=True))
#encoder.add(Dropout(0.2))
    encoder.add(LSTM(50))
    
    #Encodes trajectory. 
    encoded_trajectory = encoder(trajectory_input)

    #Point cloud input vector. 
    pc_input = Input(shape=(50,))

    #Concatenates trajectory embedding with point cloud BOF vector.
    embedding = merge([encoded_trajectory, pc_input], mode='concat')

    #Repeats it for input into decoder. 
    embedding = RepeatVector(word_seq_len)(embedding)

    #Decoder for taking trajectory vectors and generating natural language descriptions. 
    decoder.add(LSTM(len(vocab.keys()), input_dim=100, input_length=word_seq_len, return_sequences=True))
#decoder.add(Dropout(0.2))
    decoder.add(LSTM(len(vocab.keys()), input_dim=len(vocab.keys()), input_length=word_seq_len, return_sequences=True))

    decoded_word_sequence = decoder(embedding)

    #Compiles model for use. 
    model = Model(input=[trajectory_input, pc_input], output=decoded_word_sequence)
    model.compile(loss='mse', optimizer='rmsprop')

    #Trains model. 
    model.fit([train_t, train_p], train_l, nb_epoch = int(parameter), batch_size = 1)

    #Predicts from training. 
    predictions =  model.predict([test_t, test_p], batch_size = 1)
    
    #Turns to one-hot vectors.
    predictions = preprocess.prob_vectors_to_classes(predictions) 

    #Converts predictions to phrases in natural language. 
    predictions = preprocess.class_vectors_to_phrases(predictions, vocab)

    results_file = open('lstm_fold_' + str(fold_num) + '_' + str(parameter) + '.txt', 'w')

    for i in range(len(predictions)):
        results_file.write(predictions[i] + ' :::: ' + test_l[i] + '\n')

    results_file.close()

def print_usage():
    print('Run lstm training: ./lstm.py run_lstm [fold_num]')
    print('Evaluate LSTM results file using METEOR: ./lstm.py evaluate [results_file] [ref_file] [test_file]')

if __name__ == '__main__':
    if not len(sys.argv) >= 2:
        print_usage()

    elif sys.argv[1] == 'run_lstm':
        if not len(sys.argv) == 4:
            print_usage()
        else:
            run_lstm(int(sys.argv[2]), float(sys.argv[3]))

    elif sys.argv[1] == 'evaluate':
        if len(sys.argv) == 5:
            evaluate_with_meteor(sys.argv[2], sys.argv[3], sys.argv[4])
        else:
            print_usage()

    else:
        print_usage()
