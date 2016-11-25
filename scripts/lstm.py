#!/scratch/cluster/rcorona/deep_learning_env/bin/python3.4

import sys
import numpy
from keras.models import Model, Sequential
from keras.layers import LSTM, RepeatVector, Dropout, Input, merge
import os
import preprocess
import pickle
import glob
import random

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
        test_file.write(prediction + '\n')
        ref_file.write(ground_truth + '\n')

    ref_file.close()
    test_file.close()

def create_meteor_files_for_folder(results_folder, ref_folder, test_folder, prefix):
    for file_name in glob.glob(results_folder + prefix + '*'):
        base_name = os.path.basename(file_name)

        evaluate_with_meteor(results_folder + base_name, ref_folder + base_name, test_folder + base_name)

def run_lstm(results_folder):
    #First randomly configure LSTM architecture. 

    while (True): 
        #Dimensions for input. 
        seq_len = 18
        traj_dim = 8

        #Decide number of LSTM encoder layers to use, between 1 and 3. 
        num_encoder_layers = random.randint(1, 3)

        #Now determine number of hidden layers for these layers. 
        encoder_h = [random.randint(25, 200) for i in range(num_encoder_layers)]

        #Determine input shapes to each encoder layer. 
        encoder_inputs = [(seq_len, traj_dim)]

        for i in range(1, num_encoder_layers): 
            encoder_inputs.append((seq_len, encoder_h[i - 1]))

        #Determine dropout between each layer. 
        encoder_dropouts = [random.uniform(0.0, 0.5) for i in range(1, num_encoder_layers)]

        #Determine number of layers in decoder. 
        num_decoder_layers = random.randint(1, 3)

        #Now determine number of hidden units for each layer except last one.
        decoder_h = [random.randint(25, 200) for i in range(1, num_decoder_layers)]
        
        #Input to first decoder layer depends on output of last encoder layer. 
        decoder_inputs = [(seq_len, 50 + encoder_h[-1])] 

        #Now determine input to rest of decoder layers.     
        for i in range(1, num_decoder_layers):
            decoder_inputs.append((seq_len, decoder_h[i - 1]))

        #Determine dropout between each layer. 
        decoder_dropouts = [random.uniform(0.0, 0.5) for i in range(1, num_decoder_layers)]

        #Set hyperparameters for training. 
        nb_epochs = random.randint(10, 500)
        batch_size = random.randint(1, 100)

        #Prepare folder to store results for architecture with these specs. 
        found_unique_name = False

        #Come up with unique name for architecture. 
        while not found_unique_name: 
            name = str(random.randint(0, 10000000))
            
            if not os.path.exists(results_folder + name): 
                found_unique_name = True
                os.mkdir(results_folder + name)

        result_path = results_folder + name + '/'

        #Write architecture specs to file. 
        specs_file = open(result_path + 'specs.txt', 'w')

        specs_file.write('NUM ENCODER LAYERS:' + str(num_encoder_layers) + '\n')
        
        for i in range(len(encoder_h)):
            specs_file.write('EH' + str(i) + ':' + str(encoder_h[i]) + '\n')

        for i in range(len(encoder_dropouts)): 
            specs_file.write('ED' + str(i) + ':' + str(encoder_dropouts[i]) + '\n')

        for i in range(len(encoder_inputs)): 
            specs_file.write('EI' + str(i) + ':' + str(encoder_inputs[i]) + '\n')

        specs_file.write('NUM DECODER LAYERS:' + str(num_decoder_layers) + '\n')

        for i in range(len(decoder_h)):
            specs_file.write('DH' + str(i) + ':' + str(num_decoder_layers) + '\n')

        for i in range(len(decoder_dropouts)):
            specs_file.write('DD' + str(i) + ':' + str(decoder_dropouts[i]) + '\n')

        for i in range(len(decoder_inputs)):
            specs_file.write('DI' + str(i) + ':' + str(decoder_inputs[i]) + '\n')

        specs_file.write('EPOCHS:' + str(nb_epochs) + '\n')
        specs_file.write('BATCH SZ:' + str(batch_size))

        specs_file.close()

        #Keep track of children. 
        pids = []

        #Now build a model for each fold. 
        for fold_num in range(1, 6):
            pid = os.fork()
            
            if not pid == 0:
                pids.append(pid)

            else:
                STDIN_FILENO = 0
                STDOUT_FILENO = 1
                STDERR_FILENO = 2

                #redirect stdout            
                new_stdout = os.open(result_path + str(fold_num) + '.stdout', os.O_WRONLY|os.O_CREAT|os.O_TRUNC)
                os.dup2(new_stdout, STDOUT_FILENO)

                #redirect stderr
                new_stderr = os.open(result_path + str(fold_num) + '.stderr', os.O_WRONLY|os.O_CREAT|os.O_TRUNC)
                os.dup2(new_stderr, STDERR_FILENO)

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
                train_l = preprocess.to_one_hot(vocab, train_l, seq_len)

                #Pad trajectories. 
                train_t = preprocess.pad_trajectory_sequences(train_t, seq_len)
                test_t = preprocess.pad_trajectory_sequences(test_t, seq_len)

                #Load point clouds using their file names.
                print('Loading pcs')
                kmeans_model = pickle.load(open(kmeans_path + str(fold_num) + '.p', 'rb'), encoding='latin1')

                train_p = preprocess.load_point_clouds(train_p, kmeans_model)
                test_p = preprocess.load_point_clouds(test_p, kmeans_model)

                #Last layer outputs a vector of vocabulary dimensionality. 
                decoder_h.append(len(vocab.keys()))

                #Instantiate encoder and decoder. 
                encoder = Sequential()
                decoder = Sequential()

                #Specifies inputs. 
                trajectory_input = Input(shape=(seq_len, traj_dim))

                #Now build encoder. 
                for i in range(num_encoder_layers): 
                    #Last layer does not return a sequence and has no dropout. 
                    if i < num_encoder_layers - 1:
                        encoder.add(LSTM(encoder_h[i], input_shape=encoder_inputs[i], return_sequences=True))
                        encoder.add(Dropout(encoder_dropouts[i]))
                    else:
                        encoder.add(LSTM(encoder_h[i], input_shape=encoder_inputs[i], return_sequences=False))
                    
                #Encodes trajectory. 
                encoded_trajectory = encoder(trajectory_input)

                #Point cloud input vector. 
                pc_input = Input(shape=(50,))

                #Concatenates trajectory embedding with point cloud BOF vector.
                embedding = merge([encoded_trajectory, pc_input], mode='concat')

                #Repeats it for input into decoder. 
                embedding = RepeatVector(seq_len)(embedding)

                #Now build decoder. 
                for i in range(num_decoder_layers):
                    decoder.add(LSTM(decoder_h[i], input_shape=decoder_inputs[i], return_sequences=True))

                    #Don't add dropout if last layer. 
                    if i < num_decoder_layers - 1:
                        decoder.add(Dropout(decoder_dropouts[i]))

                decoded_word_sequence = decoder(embedding)

                #Compiles model for use. 
                model = Model(input=[trajectory_input, pc_input], output=decoded_word_sequence)
                model.compile(loss='mse', optimizer='rmsprop')

                #Trains model. 
                model.fit([train_t, train_p], train_l, nb_epoch = nb_epochs, batch_size = batch_size)

                #Predicts from training. 
                predictions =  model.predict([test_t, test_p], batch_size = batch_size)
                
                #Turns to one-hot vectors.
                predictions = preprocess.prob_vectors_to_classes(predictions) 

                #Converts predictions to phrases in natural language. 
                predictions = preprocess.class_vectors_to_phrases(predictions, vocab)

                results_file = open(result_path + str(fold_num) + '.results', 'w')

                for i in range(len(predictions)):
                    results_file.write(predictions[i] + ' :::: ' + test_l[i] + '\n')

                results_file.close()

                #Child exit. 
                sys.exit()

        #Wait for all children to finish. 
        for pid in pids:
            os.waitpid(pid, 0)

def print_usage():
    print('Run lstm training: ./lstm.py run_lstm [results_folder]')
    print('Generate METEOR files for one results file: ./lstm.py evaluate_file [results_file] [ref_file] [test_file]')
    print('Generate METEOR files for one entire folder: ./lstm.py evaluate_folder [folder] [ref_folder] [test_folder] [prefix]')

if __name__ == '__main__':
    if not len(sys.argv) >= 2:
        print_usage()

    elif sys.argv[1] == 'run_lstm':
        if not len(sys.argv) == 3:
            print_usage()
        else:
            run_lstm(sys.argv[2])

    elif sys.argv[1] == 'evaluate_file':
        if len(sys.argv) == 5:
            evaluate_with_meteor(sys.argv[2], sys.argv[3], sys.argv[4])
        else:
            print_usage()

    elif sys.argv[1] == 'evaluate_folder':
        if len(sys.argv) == 6:
            create_meteor_files_for_folder(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        else:
            print_usage()

    else:
        print_usage()
