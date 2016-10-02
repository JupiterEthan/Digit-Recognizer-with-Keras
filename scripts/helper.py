import pickle
import os
import random

def dataReader(FileName,FileType='train'):
    if FileType == 'train':
        if checkExist('../temp/train.dat'):
            data = saveLoad('../temp/train.dat','load')
            trainData = data['trainData']
            trainLabel = data['trainLabel']
            return trainData, trainLabel
        else:
            trainData = []
            trainLabel = []
            f = open('../data/'+FileName)
            filelines = f.readlines()
            filelines = filelines[1:]
            for line in filelines:
                line = line.strip()
                line = line.split(',')
                trainLabel.append(int(line[0]))
                trainData.append([float(x) for x in line[1:]])
            f.close()
            data = {'trainData':trainData,'trainLabel':trainLabel}
            saveLoad('../temp/train.dat','save',data)
            return trainData, trainLabel
    elif FileType == 'test':
        if checkExist('../temp/test.dat'):
            data = saveLoad('../temp/test.dat','load')
            testData = data
            return testData
        else:
            testData = []
            f = open('../data/'+FileName)
            filelines = f.readlines()
            filelines = filelines[1:]
            for line in filelines:
                line = line.strip()
                line = line.split(',')
                testData.append([float(x) for x in line])
            f.close()
            saveLoad('../temp/test.dat','save',testData)
            return testData
    else:
        print 'Invalid filetype.'
        

def saveLoad(FileName,Option,Data=None):
    if Option == 'save':
        d = os.path.dirname(FileName)
        if not os.path.exists(d):
            os.makedirs(d)
        f = open(FileName,'wb')
        pickle.dump(Data,f)
        f.close()
        
        print 'Data saved.'
    elif Option == 'load':
        f = file(FileName,'rb')
        data = pickle.load(f)
        print 'Data loaded.'
        return data
    else:
        print 'Invalid saveLoad option.'

def checkExist(FileName):
    if os.path.isfile(FileName):
        return True
    else:
        return False
        
def dataNormalization(data):
    data_norm = []
    for line in data:
        data_line = []
        for num in line:
            data_line.append((num-128.0)/128.0)
        data_norm.append(data_line)
    return data_norm
        
        
def dataRandomize(data,label=None):
    index = range(len(data))
    random.shuffle(index)
    data_rand = [data[i] for i in index]
    if label is None:
        return data_rand
    else:
        label_rand = [label[i] for i in index]
        return data_rand, label_rand
    
            
        
def onehotRepresentation(label,category_num):
    one_hot_label = []
    for l in label:
        labelLine = [0]*category_num
        labelLine[l] = 1
        one_hot_label.append(labelLine)
    return one_hot_label