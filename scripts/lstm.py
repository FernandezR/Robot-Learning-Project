#!/scratch/cluster/rcorona/deep_learning_env/bin/python3.4

import sys
import numpy
from keras.models import Model, Sequential
from keras.layers import LSTM, RepeatVector, Dropout, Input, merge
from keras.optimizers import RMSprop
import os
import preprocess
import pickle
import glob
import random
import subprocess

kmeans_path = '/scratch/cluster/rcorona/Robot_Learning_Project/Robot-Learning-Project/Models/Baseline/kmeans_fold_' 

def generate_meteor_files(results_file_name, ref_file_name, test_file_name):
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

def evalute_with_meteor(ref_file_name, test_file_name, extension, out_file_name): 
    #First prepare arguments for calling METEOR. 
    args = ['java', '-Xmx2G', 
            '-jar', 'meteor/meteor-1.5/meteor-1.5.jar', 
            test_file_name, ref_file_name, '-norm', 
            '-writeAlignments', '-f', 
            extension]

    #Rederict output to get final score into a file. 
    out_file = open(out_file_name, 'w')

    #Score usin METEOR. 
    subprocess.call(args, stdout=out_file, stderr=out_file)

def extract_meteor_score(file_name): 
    results_file = open(file_name, 'r')
    lines = results_file.readlines()
    results_file.close()

    #Parse score from text.
    try:
        score = float(lines[-1].strip().split(':')[1].strip())
    except: 
        score = 0.0

    return score

def get_second_element(l): 
    return l[1]

def consolidate_meteor_scores(folder_path, out_file_name): 
    meteor_files = {}
    
    #Gather all meteor score files. 
    for folder_name in os.listdir(folder_path): 
        folder_meteor_files = glob.glob(folder_path + folder_name + '/*.meteor')
    
        if len(folder_meteor_files) > 0: 
            meteor_files[folder_name] = folder_meteor_files

    #Get average scores over all results. 
    averages = []

    for folder in meteor_files: 
        scores = []

        for result_file in meteor_files[folder]:
            scores.append(extract_meteor_score(result_file))

        #Now average them. 
        averages.append([folder, float(sum(scores)) / float(len(scores))])

    #Now sort them. 
    averages = sorted(averages, key=get_second_element)

    print(averages)

def evaluate_folder_with_meteor(folder_path): 
    #Not all folders will necessarily have result files yet, so only process ones that do. 
    result_files = []

    for folder_name in os.listdir(folder_path):
        folder_result_files = glob.glob(folder_path + folder_name + '/*.results') 

        if len(folder_result_files) > 0: 
            for result_file in folder_result_files:
                result_files.append(result_file)

    #Remove files that already have results to save computation time.
    pruned_results = []

    for result_file in result_files: 
        path, file_name = os.path.split(result_file)
        path += '/'
        result_name = file_name.split('.')[0]
       
        if True: #not os.path.isfile(path + result_name + '-align.out'):
            pruned_results.append(result_file)

    #Evaluate every result file with METEOR. 
    for result_file in pruned_results:
        print ("Evaluating " + result_file)

        #Get path to file and file itself in separate variables. 
        path, file_name = os.path.split(result_file)
        path += '/'
        result_name = file_name.split('.')[0]

        #Files for evaluation. 
        ref_file = path + result_name + '.ref'
        test_file = path + result_name + '.test'
        out_file = path + result_name + '.meteor'

        #First generate reference and test files for evaluation from LSTM results. 
        generate_meteor_files(path + file_name, ref_file, test_file) 

        #Now evaluate with METEOR. 
        evalute_with_meteor(ref_file, test_file, path + result_name, out_file)

def read_in_lstm_specs(specs_file_name):
    specs_file = open(specs_file_name, 'r')
    lines = specs_file.readlines()
    specs_file.close()

    specs = {}

    for line in lines: 
        #Specifies number of encoder layers. 
        if line.startswith('NUM ENCODER LAYERS:'): 
            specs['num_encoder_layers'] = int(line.split(':')[1].strip())

            #Initialize other values accordingly. 
            specs['encoder_h'] = [None] * specs['num_encoder_layers']
            specs['encoder_dropouts'] = [None] * (specs['num_encoder_layers'] - 1)
            specs['encoder_inputs'] = [None] * specs['num_encoder_layers']

        #Specifies encoder hidden layers. 
        elif line.startswith('EH'): 
            layer_num, num_units = line.strip().split(':')
            layer_num = int(layer_num.split('H')[1])
            num_units = int(num_units)

            specs['encoder_h'][layer_num] = num_units

        #Specifies encoder dropouts. 
        elif line.startswith('ED'):
            layer_num, dropout = line.strip().split(':')
            layer_num = int(layer_num.split('D')[1])
            dropout = float(dropout)

            specs['encoder_dropouts'][layer_num] = dropout
            
        #Specifies encoder input dimensions. 
        elif line.startswith('EI'): 
            layer_num, dimensions = line.strip().split(':')
            layer_num = int(layer_num.split('I')[1])
            dimensions = tuple([int(e) for e in dimensions.strip(')').strip('(').split(', ')])

            specs['encoder_inputs'][layer_num] = dimensions

        #Specifies number of decoder layers. 
        elif line.startswith('NUM DECODER LAYERS'):
            specs['num_decoder_layers'] = int(line.split(':')[1].strip())

            #Initialize other values  accordingly. 
            specs['decoder_h'] = [None] * (specs['num_decoder_layers'] -  1)
            specs['decoder_dropouts'] = [None] * (specs['num_decoder_layers'] - 1)
            specs['decoder_inputs'] = [None] * specs['num_decoder_layers']

        #Specifies decoder hidden layers. 
        elif line.startswith('DH'):
            layer_num, num_units = line.strip().split(':')
            layer_num = int(layer_num.split('H')[1])
            num_units = int(num_units)

            #Fixes error in original specs files. 
            if num_units < 10: 
                num_units = -1

            specs['decoder_h'][layer_num] = num_units

        #Specifies decoder dropouts. 
        elif line.startswith('DD'): 
            layer_num, dropout = line.strip().split(':')
            layer_num = int(layer_num.split('DD')[1])
            dropout = float(dropout)

            specs['decoder_dropouts'][layer_num] = dropout

        #Specifies decoder input dimensions. 
        elif line.startswith('DI'): 
            layer_num, dimensions = line.strip().split(':')
            layer_num = int(layer_num.split('I')[1])
            dimensions = tuple([int(e) for e in dimensions.strip(')').strip('(').split(', ')])

            specs['decoder_inputs'][layer_num] = dimensions

        #Specifies number of epochs for training. 
        elif line.startswith('EPOCHS'):
            specs['nb_epochs'] = int(line.strip().split(':')[1])

        #Specifies size of batch size for training. 
        elif line.startswith('BATCH SZ'): 
            values = line.strip().split(':')

            #Get batch size.
            if len(values) == 2:
                batch_sz = int(values[1])

            else:
                #Fixes mistake where learning rate was being written to same line. 
                specs['lr'] = int(values[2])

                batch_sz = float(values[1].split('LR')[0])

            specs['batch_sz'] = batch_sz

        #Specifies the learning rate. 
        elif line.startswith('LR'): 
            specs['lr'] = float(line.strip().split(':')[1])

    #Some files had a default unwritten learning rate. 
    if not 'learning_rate' in specs:
        specs['learning_rate'] = 0.001

    #Have to infer decoder hidden units for some fails due to an error. 
    if -1 in specs['decoder_h']: 
        for i in range(len(specs['decoder_h'])):
            specs['decoder_h'][i] = specs['decoder_inputs'][i + 1][1]

    return specs

def test_lstm(architecture_file, results_folder):
    print(read_in_lstm_specs(architecture_file))

def tune_lstm(results_folder):
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
        learning_rate = random.uniform(0.001, 1.0)

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
            specs_file.write('DH' + str(i) + ':' + str(decoder_h[i]) + '\n')

        for i in range(len(decoder_dropouts)):
            specs_file.write('DD' + str(i) + ':' + str(decoder_dropouts[i]) + '\n')

        for i in range(len(decoder_inputs)):
            specs_file.write('DI' + str(i) + ':' + str(decoder_inputs[i]) + '\n')

        specs_file.write('EPOCHS:' + str(nb_epochs) + '\n')
        specs_file.write('BATCH SZ:' + str(batch_size) + '\n')
        specs_file.write('LR:' + str(learning_rate))

        specs_file.close()

        fold_num = random.randint(1, 5)
        
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

        train = preprocess.load_data_set(path, train)[:-1]
        test = preprocess.load_data_set(path, test)[:-1]

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
        rms = RMSprop(lr=learning_rate)
        model.compile(loss='mse', optimizer=rms)

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


def print_usage():
    print('Run lstm tuning: ./lstm.py tune_lstm [results_folder]')
    print('Generate METEOR files for one results file: ./lstm.py evaluate_file [results_file] [ref_file] [test_file]')
    print('Generate METEOR files for one entire folder: ./lstm.py evaluate_folder [folder]')
    print('Consolidate METEOR scores in folder: ./lstm.py consolidate_meteor [folder] [results_file]')
    print('Test LSTM architecture: ./lstm.py test_lstm [specs_file] [results_folder]')

if __name__ == '__main__':
    if not len(sys.argv) >= 2:
        print_usage()

    elif sys.argv[1] == 'tune_lstm':
        if not len(sys.argv) == 3:
            print_usage()
        else:
            run_lstm(sys.argv[2])

    elif sys.argv[1] == 'consolidate_meteor':
        if len(sys.argv) == 4: 
            consolidate_meteor_scores(sys.argv[2], sys.argv[3])            
        else:
            print_usage()

    elif sys.argv[1] == 'evaluate_file':
        if len(sys.argv) == 5:
            generate_meteor_files(sys.argv[2], sys.argv[3], sys.argv[4])
        else:
            print_usage()

    elif sys.argv[1] == 'evaluate_folder':
        if len(sys.argv) == 3:
            evaluate_folder_with_meteor(sys.argv[2])
        else:
            print_usage()

    elif sys.argv[1] == 'test_lstm':
        if len(sys.argv) == 4:
            test_lstm(sys.argv[2], sys.argv[3])
        else:
            print_usage()

    else:
        print_usage()
