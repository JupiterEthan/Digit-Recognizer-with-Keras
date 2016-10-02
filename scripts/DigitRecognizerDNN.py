# This file and other scripts are in folder "scripts"
# The data files are in folder "data"

from helper import dataReader, onehotRepresentation, dataNormalization, dataRandomize
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import os


# set configuration
readTrainData = True
readTestData = True
constructModel = True
trainModel = True
saveModel = True
predictResults = True
outputResults = True

# read training data and test data
if readTrainData:
    trainData, trainLabel = dataReader('train.csv','train')    
    trainData = dataNormalization(trainData)
    
    print 'data normalized.'
    
    trainLabel = onehotRepresentation(trainLabel,10)
    
    trainData, trainLabel = dataRandomize(trainData,trainLabel)
    
    print 'data randomized.'
    
    trainData = np.array(trainData)
    trainLabel = np.array(trainLabel)

if readTestData:
    testData = dataReader('test.csv','test')
    testData = dataNormalization(testData)
    
    print 'data normalized.'
    
    testData = np.array(testData)
    

# train DNN
if constructModel:
    print 'net constructing...'
    
    model = Sequential()
    
    
    model.add(Dense(32,batch_input_shape=(None,28*28),init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32,init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10,init='uniform'))
    model.add(Activation('softmax'))
    
    print 'net constructed.'

    sgd = SGD(lr=0.005,decay=1e-4,momentum=0.9)

    print 'net compiling...'
    
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    
    print 'net compiled.'

    if trainModel:
        print 'model training...'
        model.fit(trainData,trainLabel,nb_epoch=60,batch_size=32,verbose=2)
        print 'training finished.'
    
    model_path = '../models/digitDNN.h5'
    d = os.path.dirname(model_path)
    if not os.path.exists(d):
        os.makedirs(d)
    model.save(model_path,overwrite=False)
    print 'model saved.'
else:
    model = load_model(model_path)
    print 'model loaded.'


# DNN prediction
if predictResults:
    testPredictions = model.predict_classes(testData,batch_size=32,verbose=1)
    


# generate the output file
if outputResults:
    output = '../results/prediction.csv'
    d = os.path.dirname(output)
    if not os.path.exists(d):
        os.makedirs(d)
    f = open(output,'w')
    f.write('ImageId,Label\n')
    for index, predict in enumerate(testPredictions.tolist()):
        f.write(str(index+1))
        f.write(',')
        f.write(str(predict))
        f.write('\n')
    f.close()

print '\nfinished.'