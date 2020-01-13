import cv2
from scipy.io import wavfile
import tflearn
import numpy as np
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

# For loading BEE1 ANN Model
def load_image_ann_bee1(path):
    input_layer = input_data(shape=[None,32,32,1])
    fc_layer_1 = fully_connected(input_layer, 60,
                                  activation='relu',
                                  regularizer='L2',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(path)
    return model

# Fit function for BEE1
def fit_image_ann_bee1(ann,image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scaled_gray_image = gray_image/255.0
    scaled_gray_image = np.array(scaled_gray_image)
    prediction = ann.predict(scaled_gray_image.reshape([-1, 32, 32, 1]))
    output = [0,0]
    index = np.argmax(prediction, axis=1)[0]
    output[index] = 1
    return output

# For loading BEE2_1S ANN Model
def load_image_ann_bee2_1S(path):
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
    model.load(path)
    return model

# For loading BEE2_2S ANN Model
def load_image_ann_bee2_2S(path):
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
    model.load(path)
    return model

# Fit function for BEE2_1S and BEE2_2S
def fit_image_ann_bee2(ann,image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scaled_gray_image = gray_image/255.0 
    scaled_gray_image = np.array(scaled_gray_image)
    scaled_gray_image = cv2.resize(scaled_gray_image,(90,90))
    prediction = ann.predict(scaled_gray_image.reshape([-1, 90, 90, 1]))
    output = [0,0]
    index = np.argmax(prediction, axis=1)[0]
    output[index] = 1
    return output

# For loading BUZZ1 ANN Model
def load_audio_ann_buzz1(path):
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
    model.load(path)
    return model

# Fit function for BUZZ1
def fit_audio_ann_buzz1(ann, audio_path):
    samplerate, audio = wavfile.read(audio_path)
    audio = audio/float(np.max(audio))

    mid = len(audio)//2
    piece1 = audio[mid-1000:mid+1]
    piece2 = audio[mid+1:mid+1000]
    audio = np.concatenate((piece1,piece2))

    prediction = ann.predict(audio.reshape([-1, 100, 20, 1]))
    output = [0,0,0]
    index = np.argmax(prediction, axis=1)[0]
    output[index] = 1
    return output

# For loading BUZZ2 ANN Model
def load_audio_ann_buzz2(path):
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
    model.load(path)
    return model

# Fit function for BUZZ2
def fit_audio_ann_buzz2(ann, audio_path):
    samplerate, audio = wavfile.read(audio_path)
    audio = audio/float(np.max(audio))

    mid = len(audio)//2
    piece1 = audio[mid-1000:mid+1]
    piece2 = audio[mid+1:mid+1000]
    audio = np.concatenate((piece1,piece2))

    prediction = ann.predict(audio.reshape([-1, 100, 20, 1]))
    output = [0,0,0]
    index = np.argmax(prediction, axis=1)[0]
    output[index] = 1
    return output


# # BEE1
# path = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/Final Project Workspace/Project 1 Part 1/BEE1_ANN.tfl'
# ann = load_image_ann_bee1(path)

# # Yes BEE
# image_path = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/BEE1/bee_valid/img0/74_7_yb.png'
# print(fit_image_ann_bee1(ann,image_path))

# # No BEE
# image_path = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/BEE1/no_bee_valid/img0/192_168_4_5-2017-05-13_16-38-06_58_109_22.png'
# print(fit_image_ann_bee1(ann,image_path))


# # BEE2_1S
# path = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/Final Project Workspace/Project 1 Part 1/BEE2_1S_ANN.tfl'
# ann = load_image_ann_bee2_1S(path)

# # Yes BEE
# image_path = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/BEE2_1S/one_super/validation/bee/1/3796.png'
# print(fit_image_ann_bee2(ann,image_path))

# # No BEE
# image_path = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/BEE2_1S/one_super/validation/no_bee/2/1506.png'
# print(fit_image_ann_bee2(ann,image_path))


# BEE2_2S
# path = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/Final Project Workspace/Project 1 Part 1/BEE2_2S_ANN.tfl'
# ann = load_image_ann_bee2_2S(path)

# # Yes BEE
# image_path = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/BEE2_2S/two_super/validation/bee/1/6885.png'
# print(fit_image_ann_bee2(ann,image_path))

# # No BEE
# image_path = '/home/nikhilganta/Documents/Intelligent Systems/Project 1/BEE2_2S/two_super/validation/no_bee/1/833.png'
# print(fit_image_ann_bee2(ann,image_path))


# BUZZ 1
path = 'C:/Masters in Computer Science/Semester 1/CS6600 Intelligent Systems/Project 1/Project 1 Part 1/buzz1_ANN.tfl'
ann = load_audio_ann_buzz1(path)

# BEE
audio_path = 'D:/Masters in Computer Science/Semester 1/CS6600 Intelligent Systems/Project 1/BUZZ1/out_of_sample_data_for_validation/bee_test/192_168_4_6-2017-08-09_14-15-01_1.wav'
print(fit_audio_ann_buzz1(ann,audio_path))

# Cricket
audio_path = 'D:/Masters in Computer Science/Semester 1/CS6600 Intelligent Systems/Project 1/BUZZ1/out_of_sample_data_for_validation/cricket_test/cricket50_192_168_4_10-2017-07-31_02-15-01.wav'
print(fit_audio_ann_buzz1(ann,audio_path))

# Noise
audio_path = 'D:/Masters in Computer Science/Semester 1/CS6600 Intelligent Systems/Project 1/BUZZ1/out_of_sample_data_for_validation/noise_test/noise264_192_168_4_10-2017-07-09_05-00-01_2.wav'
print(fit_audio_ann_buzz1(ann,audio_path))


# BUZZ 2
path = 'C:/Masters in Computer Science/Semester 1/CS6600 Intelligent Systems/Project 1/Project 1 Part 1/buzz2_ANN.tfl'
ann = load_audio_ann_buzz2(path)

# BEE
audio_path = 'D:/Masters in Computer Science/Semester 1/CS6600 Intelligent Systems/Project 1/BUZZ2/out_of_sample_data_for_validation/bee_valid/192_168_4_5-2018-05-26_17-00-02_9.wav'
print(fit_audio_ann_buzz2(ann,audio_path))

# Cricket
audio_path = 'D:/Masters in Computer Science/Semester 1/CS6600 Intelligent Systems/Project 1/BUZZ2/out_of_sample_data_for_validation/cricket_valid/192_168_4_8-2017-08-13_03-45-01_4.wav'
print(fit_audio_ann_buzz2(ann,audio_path))

# Noise
audio_path = 'D:/Masters in Computer Science/Semester 1/CS6600 Intelligent Systems/Project 1/BUZZ2/out_of_sample_data_for_validation/noise_valid/192_168_4_5-2018-05-12_19-45-01_0.wav'
print(fit_audio_ann_buzz2(ann,audio_path))