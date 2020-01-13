import pickle
import glob
from scipy.io import wavfile
import numpy as np
from sklearn.utils import shuffle
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from sklearn.externals import joblib
import time

path = "/home/kulyukin-lab1/Nikhil Project/Bees and Buzz Pickle Files/"

def save(ann, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(ann, fp)

def load(file_name):
    with open(path + file_name, 'rb') as fp:
        nn = pickle.load(fp)
    return nn

# The below function preprocesses the data of BEE1, BEE2_1S, BEE2_2S based on the respective paths and this is used for pickling the data
# and then used for loading the data.
def beesPreProcessor():
    dataX = []
    dataY = []
    beepath = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/BEE2_1S/'
    nonbeepath = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/BEE2_1S/'
    bees = glob.glob(beepath + '/*/testing/bee/*/*.png')
    nonbees = glob.glob(nonbeepath + '/*/testing/no_bee/*/*.png')
    filePaths = []
    for file in bees:
        filePaths.append((file,(1,0)))
    for file in nonbees:
        filePaths.append((file,(0,1)))
    filePaths = shuffle(filePaths)
    for i in range(len(filePaths)):
        image = cv2.imread(filePaths[i][0])
        scaled_image = image/255.0
        scaled_image = cv2.resize(scaled_image,(90,90))
        scaled_image = np.array(scaled_image)
        dataX.append(scaled_image)
        dataY.append(filePaths[i][1])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    return dataX,dataY

def buzz1ValidPreProcessor():
    dataX = []
    dataY = []
    beepath = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/BUZZ1/out_of_sample_data_for_validation/'
    cricketpath = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/BUZZ1/out_of_sample_data_for_validation/'
    noisepath = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/BUZZ1/out_of_sample_data_for_validation/'
    bees = glob.glob(beepath + '/bee_test/*.wav')
    crickets = glob.glob(cricketpath + '/cricket_test/*.wav')
    noises = glob.glob(noisepath + '/noise_test/*.wav')
    filePaths = []
    for file in bees:
        filePaths.append((file,(1,0,0)))
    for file in crickets:
        filePaths.append((file,(0,1,0)))
    for file in noises:
        filePaths.append((file,(0,0,1)))
    filePaths = shuffle(filePaths)
    for i in range(len(filePaths)):
        samplerate, audio = wavfile.read(filePaths[i][0])
        audio = audio/float(np.max(audio))

        # Audio Processing
        mid = len(audio)//2
        piece1 = audio[mid-1000:mid+1]
        piece2 = audio[mid+1:mid+1000]
        audio = np.concatenate((piece1,piece2))

        dataX.append(audio)
        dataY.append(filePaths[i][1])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    return dataX,dataY

def buzz1TestTrainPreProcessor():
    dataX = []
    dataY = []
    beepath = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/BUZZ1/bee/'
    cricketpath = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/BUZZ1/cricket/'
    noisepath = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/BUZZ1/noise/'
    bees = glob.glob(beepath + '/*.wav')
    crickets = glob.glob(cricketpath + '/*.wav')
    noises = glob.glob(noisepath + '/*.wav')
    filePaths = []
    for file in bees:
        filePaths.append((file,(1,0,0)))
    for file in crickets:
        filePaths.append((file,(0,1,0)))
    for file in noises:
        filePaths.append((file,(0,0,1)))
    filePaths = shuffle(filePaths)
    for i in range(len(filePaths)):
        samplerate, audio = wavfile.read(filePaths[i][0])
        audio = audio/float(np.max(audio))

        # Audio Processing
        mid = len(audio)//2
        piece1 = audio[mid-1000:mid+1]
        piece2 = audio[mid+1:mid+1000]
        audio = np.concatenate((piece1,piece2))

        dataX.append(audio)
        dataY.append(filePaths[i][1])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    return dataX,dataY

def buzz2PreProcessor(dataType):
    dataX = []
    dataY = []
    beepath = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/BUZZ2/'
    cricketpath = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/BUZZ2/'
    noisepath = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/BUZZ2/'
    bees = glob.glob(beepath + dataType + '/bee_' + dataType +'/*.wav')
    crickets = glob.glob(cricketpath + dataType + '/cricket_' + dataType +'/*.wav')
    noises = glob.glob(noisepath + dataType + '/cricket_' + dataType +'/*.wav')
    filePaths = []
    for file in bees:
        filePaths.append((file,(1,0,0)))
    for file in crickets:
        filePaths.append((file,(0,1,0)))
    for file in noises:
        filePaths.append((file,(0,0,1)))
    filePaths = shuffle(filePaths)
    for i in range(len(filePaths)):
        samplerate, audio = wavfile.read(filePaths[i][0])
        audio = audio/float(np.max(audio))

        # Audio Processing
        mid = len(audio)//2
        piece1 = audio[mid-1000:mid+1]
        piece2 = audio[mid+1:mid+1000]
        audio = np.concatenate((piece1,piece2))

        dataX.append(audio)
        dataY.append(filePaths[i][1])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    return dataX,dataY

######################################################################################################################################
######################################################################################################################################
# BEE1 Dataset

bee1_train_d = load('bee1_train_d.pck')
bee1_test_d = load('bee1_test_d.pck')
bee1_valid_d = load('bee1_valid_d.pck')

bee1_train_dX = np.array(bee1_train_d[0])
bee1_train_dY = bee1_train_d[1]
bee1_test_dX = np.array(bee1_test_d[0])
bee1_test_dY = bee1_test_d[1]
bee1_valid_dX = np.array(bee1_valid_d[0])
bee1_valid_dY = bee1_valid_d[1]

bee1_train_dX = bee1_train_dX.reshape([-1,32,32,1])
bee1_test_dX = bee1_test_dX.reshape([-1,32,32,1])

def train_bee1_ann():
    input_layer = input_data(shape=[None,32,32,1])
    fc_layer_1 = fully_connected(input_layer, 60,
                                  activation='relu',
                                  regularizer='L2',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.01)
    model = tflearn.DNN(network)
    return model

def load_train_bee1_ann(beepath):
    input_layer = input_data(shape=[None,32,32,1])
    fc_layer_1 = fully_connected(input_layer, 60,
                                  activation='relu',
                                  regularizer='L2',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(beepath)
    return model

def test_tflearn_bee1_ann_model(ann_model, validX, validY):
    results = []
    for i in range(len(validX)):
        prediction = ann_model.predict(validX[i].reshape([-1, 32, 32, 1]))
        results.append(np.argmax(prediction, axis=1)[0] == \
                       np.argmax(validY[i]))
    return float(sum((np.array(results) == True)))/float(len(results))

NUM_EPOCHS = 50
BATCH_SIZE = 10
MODEL = train_bee1_ann()
MODEL.fit(bee1_train_dX, bee1_train_dY, n_epoch=NUM_EPOCHS,
          shuffle=True,
          validation_set=(bee1_test_dX, bee1_test_dY),
          show_metric=True,
          batch_size=BATCH_SIZE,
          run_id='BEE1_ANN_1')
SAVE_ANN_PATH = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/workspace/part1/BEE1_ANN.tfl'
MODEL.save(SAVE_ANN_PATH)
# For checking whether the network has been trained correctly
print(MODEL.predict(bee1_test_dX[0].reshape([-1, 32, 32, 1])))

# Classifying the images on validation data and deriving the validation accuracy

bee1_ann_path = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/workspace/part1/BEE1_ANN.tfl'
bee1_ann = load_train_bee1_ann(bee1_ann_path)

if __name__ == '__main__':
    print('BEE1 ANN accuracy = {}'.format(test_tflearn_bee1_ann_model(bee1_ann, bee1_valid_dX, bee1_valid_dY)))


######################################################################################################################################
######################################################################################################################################
# BEE2_1S Dataset

bee2_1S_train_d = load('bee2_1S_train_d.pck')
bee2_1S_test_d = load('bee2_1S_test_d.pck')
bee2_1S_valid_d = load('bee2_1S_valid_d.pck')

bee2_1S_train_dX = np.array(bee2_1S_train_d[0])
bee2_1S_train_dY = bee2_1S_train_d[1]
bee2_1S_test_dX = np.array(bee2_1S_test_d[0])
bee2_1S_test_dY = bee2_1S_test_d[1]
bee2_1S_valid_dX = np.array(bee2_1S_valid_d[0])
bee2_1S_valid_dY = bee2_1S_valid_d[1]

bee2_1S_train_dX = bee2_1S_train_dX.reshape([-1,90,90,1])
bee2_1S_test_dX = bee2_1S_test_dX.reshape([-1,90,90,1])


def train_bee2_1S_ann():
    input_layer = input_data(shape=[None,90,90,1])
    fc_layer_1 = fully_connected(input_layer, 8100,
                                  activation='relu',
                                  name='fc_layer_1')
    fc_layer_3 = fully_connected(fc_layer_1, 60,
                                 activation='relu',
                                 name='fc_layer_3')
    fc_layer_4 = fully_connected(fc_layer_3, 2,
                                 activation='softmax',
                                 name='fc_layer_4')
    network = regression(fc_layer_4, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.01)
    model = tflearn.DNN(network)
    return model

def load_train_bee2_1S_ann(beepath):
    input_layer = input_data(shape=[None,90,90,1])
    fc_layer_1 = fully_connected(input_layer, 8100,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_3 = fully_connected(fc_layer_1, 60,
                                 activation='relu',
                                 name='fc_layer_3')
    fc_layer_4 = fully_connected(fc_layer_3, 2,
                                 activation='softmax',
                                 name='fc_layer_4')
    model = tflearn.DNN(fc_layer_4)
    model.load(beepath)
    return model

def test_tflearn_bee2_1S_ann_model(ann_model, validX, validY):
    results = []
    for i in range(len(validX)):
        prediction = ann_model.predict(validX[i].reshape([-1, 90, 90, 1]))
        results.append(np.argmax(prediction, axis=1)[0] == \
                       np.argmax(validY[i]))
    return float(sum((np.array(results) == True)))/float(len(results))

NUM_EPOCHS = 30
BATCH_SIZE = 10
MODEL = train_bee2_1S_ann()
MODEL.fit(bee2_1S_train_dX, bee2_1S_train_dY, n_epoch=NUM_EPOCHS,
          shuffle=True,
          validation_set=(bee2_1S_test_dX, bee2_1S_test_dY),
          show_metric=True,
          batch_size=BATCH_SIZE,
          run_id='BEE2_1S_ANN_1')
SAVE_ANN_PATH = '/home/kulyukin-lab1/Nikhil Project/project1 workspace/part1/BEE2_1S_ANN.tfl'
MODEL.save(SAVE_ANN_PATH)
# For checking whether the network has been trained correctly
print(MODEL.predict(bee2_1S_train_dX[0].reshape([-1, 90, 90, 1])))

# Classifying the images on validation data and deriving the validation accuracy

bee2_1S_ann_path = '/home/kulyukin-lab1/Nikhil Project/project1 workspace/part1/BEE2_1S_ANN.tfl'
bee2_1S_ann = load_train_bee2_1S_ann(bee2_1S_ann_path)

if __name__ == '__main__':
    print('BEE2_1S ANN accuracy = {}'.format(test_tflearn_bee2_1S_ann_model(bee2_1S_ann, bee2_1S_valid_dX, bee2_1S_valid_dY)))


######################################################################################################################################
######################################################################################################################################
# BEE2_2S Dataset

bee2_2S_train_d = load('bee2_2S_train_d.pck')
bee2_2S_test_d = load('bee2_2S_test_d.pck')
bee2_2S_valid_d = load('bee2_2S_valid_d.pck')

bee2_2S_train_dX = np.array(bee2_2S_train_d[0])
bee2_2S_train_dY = bee2_2S_train_d[1]
bee2_2S_test_dX = np.array(bee2_2S_test_d[0])
bee2_2S_test_dY = bee2_2S_test_d[1]
bee2_2S_valid_dX = np.array(bee2_2S_valid_d[0])
bee2_2S_valid_dY = bee2_2S_valid_d[1]

bee2_2S_train_dX = bee2_2S_train_dX.reshape([-1,90,90,1])
bee2_2S_test_dX = bee2_2S_test_dX.reshape([-1,90,90,1])

def train_bee2_2S_ann():
    input_layer = input_data(shape=[None,90,90,1])
    fc_layer_1 = fully_connected(input_layer, 100,
                                  activation='relu',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 60,
                                  activation='relu',
                                  name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 2,
                                 activation='softmax',
                                 name='fc_layer_3')
    network = regression(fc_layer_3, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.01)
    model = tflearn.DNN(network)
    return model

def load_train_bee2_2S_ann(beepath):
    input_layer = input_data(shape=[None,90,90,1])
    fc_layer_1 = fully_connected(input_layer, 100,
                                  activation='relu',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 60,
                                  activation='relu',
                                  name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 2,
                                 activation='softmax',
                                 name='fc_layer_3')
    model = tflearn.DNN(fc_layer_3)
    model.load(beepath)
    return model

def test_tflearn_bee2_2S_ann_model(ann_model, validX, validY):
    results = []
    for i in range(len(validX)):
        prediction = ann_model.predict(validX[i].reshape([-1, 90, 90, 1]))
        results.append(np.argmax(prediction, axis=1)[0] == \
                       np.argmax(validY[i]))
    return float(sum((np.array(results) == True)))/float(len(results))


NUM_EPOCHS = 50
BATCH_SIZE = 10
MODEL = train_bee2_2S_ann()
MODEL.fit(bee2_2S_train_dX, bee2_2S_train_dY, n_epoch=NUM_EPOCHS,
          shuffle=True,
          validation_set=(bee2_2S_test_dX, bee2_2S_test_dY),
          show_metric=True,
          batch_size=BATCH_SIZE,
          run_id='BEE2_2S_ANN_1')
SAVE_ANN_PATH = '/home/kulyukin-lab1/Nikhil Project/project1 workspace/part1/BEE2_2S_ANN.tfl'
MODEL.save(SAVE_ANN_PATH)
# For checking whether the network has been trained correctly
print(MODEL.predict(bee2_2S_train_dX[0].reshape([-1, 90, 90, 1])))

# Classifying the images on validation data and deriving the validation accuracy

bee2_2S_ann_path = '/home/kulyukin-lab1/Nikhil Project/project1 workspace/part1/BEE2_2S_ANN.tfl'
bee2_2S_ann = load_train_bee2_2S_ann(bee2_2S_ann_path)

if __name__ == '__main__':
    print('BEE2_2S ANN accuracy = {}'.format(test_tflearn_bee2_2S_ann_model(bee2_2S_ann, bee2_2S_valid_dX, bee2_2S_valid_dY)))


######################################################################################################################################
######################################################################################################################################
# BUZZ1 Dataset

buzz1_train_test = buzz1TestTrainPreProcessor()
buzz1_train_d = buzz1_train_test[0][:7001],buzz1_train_test[1][:7001]
buzz1_test_d = buzz1_train_test[0][7001:],buzz1_train_test[1][7001:]
buzz1_train_dX = np.array(buzz1_train_d[0])
buzz1_train_dY = buzz1_train_d[1]
buzz1_test_dX = np.array(buzz1_test_d[0])
buzz1_test_dY = buzz1_test_d[1]

buzz1_train_dX = buzz1_train_dX.reshape([-1,100,20,1])
buzz1_test_dX = buzz1_test_dX.reshape([-1,100,20,1])

buzz1_valid_d = buzz1ValidPreProcessor()
buzz1_valid_dX = np.array(buzz1_valid_d[0])
buzz1_valid_dY = buzz1_valid_d[1]


def train_buzz1_ann():
    input_layer = input_data(shape=[None,100,20,1])
    fc_layer_1 = fully_connected(input_layer, 100,
                                  activation='relu',
                                  regularizer='L2',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 60,
                                  activation='relu',
                                  regularizer='L2',
                                  name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 3,
                                 activation='softmax',
                                 name='fc_layer_3')
    network = regression(fc_layer_3, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.01)
    model = tflearn.DNN(network)
    return model

def load_train_buzz1_ann(beepath):
    input_layer = input_data(shape=[None,100,20,1])
    fc_layer_1 = fully_connected(input_layer, 100,
                                  activation='relu',
                                  regularizer='L2',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 60,
                                  activation='relu',
                                  regularizer='L2',
                                  name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 3,
                                 activation='softmax',
                                 name='fc_layer_3')
    model = tflearn.DNN(fc_layer_3)
    model.load(beepath)
    return model

def test_tflearn_buzz1_ann_model(ann_model, validX, validY):
    results = []
    for i in range(len(validX)):
        prediction = ann_model.predict(validX[i].reshape([-1, 100, 20, 1]))
        results.append(np.argmax(prediction, axis=1)[0] == \
                       np.argmax(validY[i]))
    return float(sum((np.array(results) == True)))/float(len(results))

# buzz1 Dataset
NUM_EPOCHS = 100
BATCH_SIZE = 10
MODEL = train_buzz1_ann()
MODEL.fit(buzz1_train_dX, buzz1_train_dY, n_epoch=NUM_EPOCHS,
          shuffle=True,
          validation_set=(buzz1_test_dX, buzz1_test_dY),
          show_metric=True,
          batch_size=BATCH_SIZE,
          run_id='buzz1_ANN_1')
SAVE_ANN_PATH = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/Final Project Workspace/Project 1 Part 1/buzz1_ANN.tfl'
MODEL.save(SAVE_ANN_PATH)
# For checking whether the network has been trained correctly
print(MODEL.predict(buzz1_train_dX[0].reshape([-1,100,20,1])))

buzz1_ann_path = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/Final Project Workspace/Project 1 Part 1/buzz1_ANN.tfl'
buzz1_ann = load_train_buzz1_ann(buzz1_ann_path)

if __name__ == '__main__':
    print('buzz1 ANN accuracy = {}'.format(test_tflearn_buzz1_ann_model(buzz1_ann, buzz1_valid_dX, buzz1_valid_dY)))

######################################################################################################################################
######################################################################################################################################
# BUZZ2 Dataset

# BUZZ2 Data
buzz2_train_d = buzz2PreProcessor('train')
buzz2_test_d = buzz2PreProcessor('test')
buzz2_train_dX = np.array(buzz2_train_d[0])
buzz2_train_dY = buzz2_train_d[1]
buzz2_test_dX = np.array(buzz2_test_d[0])
buzz2_test_dY = buzz2_test_d[1]

buzz2_train_dX = buzz2_train_dX.reshape([-1,100,20,1])
buzz2_test_dX = buzz2_test_dX.reshape([-1,100,20,1])

buzz2_valid_d = buzz2PreProcessor('valid')
buzz2_valid_dX = np.array(buzz2_valid_d[0])
buzz2_valid_dY = buzz2_valid_d[1]


def train_buzz2_ann():
    input_layer = input_data(shape=[None,100,20,1])
    fc_layer_1 = fully_connected(input_layer, 90,
                                  activation='relu',
                                  regularizer='L2',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 50,
                                  activation='relu',
                                  regularizer='L2',
                                  name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 3,
                                 activation='softmax',
                                 name='fc_layer_3')
    network = regression(fc_layer_3, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.01)
    model = tflearn.DNN(network)
    return model

def load_train_buzz2_ann(beepath):
    input_layer = input_data(shape=[None,100,20,1])
    fc_layer_1 = fully_connected(input_layer, 90,
                                  activation='relu',
                                  regularizer='L2',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 50,
                                  activation='relu',
                                  regularizer='L2',
                                  name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 3,
                                 activation='softmax',
                                 name='fc_layer_3')
    model = tflearn.DNN(fc_layer_3)
    model.load(beepath)
    return model

def test_tflearn_buzz2_ann_model(ann_model, validX, validY):
    results = []
    for i in range(len(validX)):
        prediction = ann_model.predict(validX[i].reshape([-1, 100, 20, 1]))
        results.append(np.argmax(prediction, axis=1)[0] == \
                       np.argmax(validY[i]))
    return float(sum((np.array(results) == True)))/float(len(results))

# buzz2 Dataset
NUM_EPOCHS = 100
BATCH_SIZE = 10
MODEL = train_buzz2_ann()
MODEL.fit(buzz2_train_dX, buzz2_train_dY, n_epoch=NUM_EPOCHS,
          shuffle=True,
          validation_set=(buzz2_test_dX, buzz2_test_dY),
          show_metric=True,
          batch_size=BATCH_SIZE,
          run_id='buzz2_ANN_1')
SAVE_ANN_PATH = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/Final Project Workspace/Project 1 Part 1/buzz2_ANN.tfl'
MODEL.save(SAVE_ANN_PATH)
# For checking whether the network has been trained correctly
print(MODEL.predict(buzz2_train_dX[0].reshape([-1, 100, 20, 1])))

buzz2_ann_path = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/Final Project Workspace/Project 1 Part 1/buzz2_ANN.tfl'
buzz2_ann = load_train_buzz2_ann(buzz2_ann_path)

if __name__ == '__main__':
    print('buzz2 ANN accuracy = {}'.format(test_tflearn_buzz2_ann_model(buzz2_ann, buzz2_valid_dX, buzz2_valid_dY)))
