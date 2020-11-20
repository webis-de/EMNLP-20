import numpy as np
import pickle
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from scipy import spatial
import math
from sklearn.manifold import TSNE
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
import sys
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import json
import os
import copy
from nltk.tokenize import word_tokenize
from scipy.special import softmax
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV


dataDir = '../../data/emnlp19-BASIL/data/'
# pickleDir = '../../pickle/'
pickleDir = '../../pickle_test/'
times = 100

low = 0.000000001

source = 'gt'
# source = 'pred'

target = 'info'
# target = 'lex'
# target = 'any'

type = 'rel'
# type = 'abs'



print('You are running:', source, 'sentence-bias,', target, 'bias,', type, 'frequency features:')

if target == 'any':
    pickleBERTDir = '../../../transformers/output/bert_local/'
else:
    pickleBERTDir = '../../../transformers/output/bert_local_' + target + '/'
    

def norValues(valuess,norLen=times):
    rValuess = []
    
    for values in valuess:
        rValues = []
        for value in values:
            rValue = np.zeros((norLen))
            norT = 100/(len(value)-1)
            for i in range(len(value)-1):
                last = round((i+1)*norT)
                if last >= norLen:
                    last = norLen-1
                vRange = np.arange(int(i*norT),last)
                rValue[vRange] = value[i] + (vRange-round(i*norT))/(round((i+1)*norT)-round(i*norT))*(value[i+1]-value[i])
            
            rValue[-1] = value[-1]
            rValues.append(rValue)
        rValuess.append(rValues)
        
    return rValuess


def sampling(biass, sample_rate, method='sample'):
    sampleds = []
    dataSize = len(biass)
    dataLen = len(biass[0])
    sample_len = int(dataLen/sample_rate)
    for i in range(dataSize):
        sampled = []
        for j in range(sample_len):
            bRange = biass[i][int(j*sample_rate):int((j+1)*sample_rate)]
            if method == 'sample':
                bValue = bRange[-1]
            elif method == 'max':
                bValue = np.max(bRange)
            else:
                bValue = np.mean(bRange)
            
            sampled.append(bValue)
        sampleds.append(sampled)    
    return np.array(sampleds)

bias_flows_all_train = []
bias_flows_info_train = []
bias_flows_lex_train = []

bias_flows_all_dev = []
bias_flows_info_dev = []
bias_flows_lex_dev = []

bias_flows_all_test = []
bias_flows_info_test = []
bias_flows_lex_test = []

bias_flows_all_test_text = []

for _ in range(3):
    bias_flows_all_train.append([])
    bias_flows_info_train.append([])
    bias_flows_lex_train.append([])
    
    bias_flows_all_dev.append([])
    bias_flows_info_dev.append([])
    bias_flows_lex_dev.append([])
    
    bias_flows_all_test.append([])
    bias_flows_info_test.append([])
    bias_flows_lex_test.append([])
    
    bias_flows_all_test_text.append([])

trainSize = 60
devSize = 20


# random test files
# fids = list(range(100))
# random.shuffle(fids)
# 
# with open(pickleDir + 'fid.pickle','wb') as f:
#     pickle.dump(fids,f)
# 
# with open(pickleDir + 'fid.pickle','rb') as f:
#     fids = pickle.load(f)

# use the test file in the paper:
with open(pickleDir + 'fid_emnlp_paper.pickle','rb') as f:
    fids = pickle.load(f)

files = os.listdir(dataDir)

for file in files:
    fid = int(file.split('_')[0])
    
    if fid in fids[:trainSize]:
        ftype = 'train'
    elif fid in fids[trainSize:trainSize+devSize]:
        ftype = 'dev'
    else:
        ftype = 'test'
        
    with open(dataDir + file, 'r') as f:
        jsonString = ''.join([line.strip() for line in f])
        jsonData = json.loads(jsonString)
        
        sentiment = jsonData['article-level-annotations']['stance']

        if sentiment == 'Center':
            bias = 1
        elif sentiment == 'Right':
            bias = 2
        else:
            bias = 0
        
        bias_flow_all = []
        bias_flow_info = []
        bias_flow_lex = []
        
        bias_flow_all_text = []
        
        for s in jsonData['body']:
            sent_text = ' '.join(word_tokenize(s['sentence'].encode('ascii','ignore').decode('utf-8')))
            
            if s['annotations']: # if there is a label
                sent_label = 1
                if s['annotations'][0]['bias'] == 'Informational':
                    sent_label_info = 1
                else:
                    sent_label_lex = 1
            else:
                sent_label = 0
                sent_label_info = 0
                sent_label_lex = 0
            
            if target == 'any':
                bias_flow_all.append(sent_label)
            elif target == 'info':
                bias_flow_all.append(sent_label_info)
            else:
                bias_flow_all.append(sent_label_lex)
                
            bias_flow_all_text.append(sent_text)
            bias_flow_info.append(sent_label_info)
            bias_flow_lex.append(sent_label_lex)
        
        if ftype == 'train':
            bias_flows_all_train[bias].append(bias_flow_all)
            bias_flows_info_train[bias].append(bias_flow_info)
            bias_flows_lex_train[bias].append(bias_flow_lex)
        elif ftype == 'dev':
            bias_flows_all_dev[bias].append(bias_flow_all)
            bias_flows_info_dev[bias].append(bias_flow_info)
            bias_flows_lex_dev[bias].append(bias_flow_lex)
        else:
            bias_flows_all_test[bias].append(bias_flow_all)
            bias_flows_info_test[bias].append(bias_flow_info)
            bias_flows_lex_test[bias].append(bias_flow_lex)


if source == 'pred':
    bias_flows_all_testP_L = []
    bias_flows_all_testP_C = []
    bias_flows_all_testP_R = []
     
    th = 0.5 
    
    def processBERT(preds):
        return softmax(preds,axis=1)[:,1]
    
    with open(pickleBERTDir+'eval_results_detail_test_L.pickle','rb') as f:
        predTe_prL = pickle.load(f)
        preId = 0
        for idd,d in enumerate(bias_flows_all_test[0]):
            docLen = len(d)
            bias_flows_all_testP_L.append([int(b) for b in predTe_prL[preId:preId+docLen,1]>th])

            preId += docLen
          
    with open(pickleBERTDir+'eval_results_detail_test_C.pickle','rb') as f:
        predTe_prC = pickle.load(f)
        preId = 0
        for d in bias_flows_all_test[1]:
            docLen = len(d)
            bias_flows_all_testP_C.append([int(b) for b in predTe_prC[preId:preId+docLen,1]>th])
            preId += docLen 
            
    with open(pickleBERTDir+'eval_results_detail_test_R.pickle','rb') as f:
        predTe_prR = pickle.load(f)
        preId = 0
        for d in bias_flows_all_test[2]:
            docLen = len(d)
            bias_flows_all_testP_R.append([int(b) for b in predTe_prR[preId:preId+docLen,1]>th])
            preId += docLen
    
    
    bias_flows_all_test = [bias_flows_all_testP_L,bias_flows_all_testP_C,bias_flows_all_testP_R]

    with open(pickleDir + 'bert_test.pickle','wb') as f:
        pickle.dump([bias_flows_all_test],f)
    
    with open(pickleDir + 'bert_test.pickle','rb') as f:
        [bias_flows_all_test] = pickle.load(f)


bias_flows_all_train = norValues(bias_flows_all_train)
bias_flows_info_train = norValues(bias_flows_info_train)
bias_flows_lex_train = norValues(bias_flows_lex_train)

bias_flows_all_dev = norValues(bias_flows_all_dev)
bias_flows_info_dev = norValues(bias_flows_info_dev)
bias_flows_lex_dev = norValues(bias_flows_lex_dev)

bias_flows_all_test = norValues(bias_flows_all_test)
bias_flows_info_test = norValues(bias_flows_info_test)
bias_flows_lex_test = norValues(bias_flows_lex_test)

with open(pickleDir + 'train_features.pickle','wb') as f:
    pickle.dump([bias_flows_all_train,bias_flows_info_train,bias_flows_lex_train],f)
    
with open(pickleDir + 'dev_features.pickle','wb') as f:
    pickle.dump([bias_flows_all_dev,bias_flows_info_dev,bias_flows_lex_dev],f)
        
with open(pickleDir + 'test_features.pickle','wb') as f:
    pickle.dump([bias_flows_all_test,bias_flows_info_test,bias_flows_lex_test],f)
 
with open(pickleDir + 'train_features.pickle','rb') as f:
    [bias_flows_all_train,bias_flows_info_train,bias_flows_lex_train] = pickle.load(f)
   
with open(pickleDir + 'dev_features.pickle','rb') as f:
    [bias_flows_all_dev,bias_flows_info_dev,bias_flows_lex_dev] = pickle.load(f)
       
with open(pickleDir + 'test_features.pickle','rb') as f:
    [bias_flows_all_test,bias_flows_info_test,bias_flows_lex_test] = pickle.load(f)

if target == 'any':
    bias_flows_trL = bias_flows_all_train[0]
    bias_flows_trC = bias_flows_all_train[1]
    bias_flows_trR = bias_flows_all_train[2]
     
    bias_flows_teL = bias_flows_all_test[0]
    bias_flows_teC = bias_flows_all_test[1]
    bias_flows_teR = bias_flows_all_test[2]
     
    bias_flows_deL = bias_flows_all_dev[0]
    bias_flows_deC = bias_flows_all_dev[1]
    bias_flows_deR = bias_flows_all_dev[2]
elif target == 'info':
    bias_flows_trL = bias_flows_info_train[0]
    bias_flows_trC = bias_flows_info_train[1]
    bias_flows_trR = bias_flows_info_train[2]
     
    bias_flows_teL = bias_flows_info_test[0]
    bias_flows_teC = bias_flows_info_test[1]
    bias_flows_teR = bias_flows_info_test[2]
     
    bias_flows_deL = bias_flows_info_dev[0]
    bias_flows_deC = bias_flows_info_dev[1]
    bias_flows_deR = bias_flows_info_dev[2]
else:
    bias_flows_trL = bias_flows_lex_train[0]
    bias_flows_trC = bias_flows_lex_train[1]
    bias_flows_trR = bias_flows_lex_train[2]
     
    bias_flows_teL = bias_flows_lex_test[0]
    bias_flows_teC = bias_flows_lex_test[1]
    bias_flows_teR = bias_flows_lex_test[2]
     
    bias_flows_deL = bias_flows_lex_dev[0]
    bias_flows_deC = bias_flows_lex_dev[1]
    bias_flows_deR = bias_flows_lex_dev[2]
    


bL_tr = [sum(b)/len(b) for b in bias_flows_trL]
bC_tr = [sum(b)/len(b) for b in bias_flows_trC]
bR_tr = [sum(b)/len(b) for b in bias_flows_trR]

bL_tr_c = [sum(b) for b in bias_flows_trL]
bC_tr_c = [sum(b) for b in bias_flows_trC]
bR_tr_c = [sum(b) for b in bias_flows_trR]

bL_de = [sum(b)/len(b) for b in bias_flows_deL]
bC_de = [sum(b)/len(b) for b in bias_flows_deC]
bR_de = [sum(b)/len(b) for b in bias_flows_deR]

bL_de_c = [sum(b) for b in bias_flows_deL]
bC_de_c = [sum(b) for b in bias_flows_deC]
bR_de_c = [sum(b) for b in bias_flows_deR]

bL_te = [sum(b)/len(b) if len(b) > 0 else 0 for b in bias_flows_teL]
bC_te = [sum(b)/len(b) if len(b) > 0 else 0 for b in bias_flows_teC]
bR_te = [sum(b)/len(b) if len(b) > 0 else 0 for b in bias_flows_teR]

bL_te_c = [sum(b) for b in bias_flows_teL]
bC_te_c = [sum(b) for b in bias_flows_teC]
bR_te_c = [sum(b) for b in bias_flows_teR]

bB_tr = np.concatenate([bL_tr,bR_tr])
bB_tr_c = np.concatenate([bL_tr_c,bR_tr_c])

bB_te = np.concatenate([bL_te,bR_te])
bB_te_c = np.concatenate([bL_te_c,bR_te_c])

bB_de = np.concatenate([bL_de,bR_de])
bB_de_c = np.concatenate([bL_de_c,bR_de_c])


if type == 'abs':
    fTrain = np.concatenate([bB_tr_c,bC_tr_c],axis=0).reshape(-1,1)
    lTrain = np.concatenate([np.ones(len(bB_tr_c)),np.zeros(len(bC_tr_c))])
        
    fDev = np.concatenate([bB_de_c,bC_de_c],axis=0).reshape(-1,1)
    lDev = np.concatenate([np.ones(len(bB_de_c)),np.zeros(len(bC_de_c))])
        
    fTest = np.concatenate([bB_te_c,bC_te_c],axis=0).reshape(-1,1)
    lTest = np.concatenate([np.ones(len(bB_te_c)),np.zeros(len(bC_te_c))])
else:
    fTrain = np.concatenate([bB_tr,bC_tr],axis=0).reshape(-1,1)
    lTrain = np.concatenate([np.ones(len(bB_tr)),np.zeros(len(bC_tr))])
       
    fDev = np.concatenate([bB_de,bC_de],axis=0).reshape(-1,1)
    lDev = np.concatenate([np.ones(len(bB_de_c)),np.zeros(len(bC_de))])
       
    fTest = np.concatenate([bB_te,bC_te],axis=0).reshape(-1,1)
    lTest = np.concatenate([np.ones(len(bB_te)),np.zeros(len(bC_te))])

best_c = 0
best_dev = 0
best_train = 0
best_test = 0

cs = [0.001,0.01,0.1,1,10,100,1000]

for c in cs:
    classifier = SVC(C=c, probability=True,kernel='linear')
    classifier.fit(fTrain,lTrain)
    
    predTr = classifier.predict(fTrain)
    predDe = classifier.predict(fDev)
    predTe = classifier.predict(fTest)
    
    if accuracy_score(lDev,predDe) > best_dev:
        best_c = c
        best_dev = accuracy_score(lDev,predDe)
        best_train = accuracy_score(lTrain,predTr)
        best_test = accuracy_score(lTest,predTe)

print('best c in SVM:', best_c)

classifier = SVC(C=best_c, probability=True,kernel='linear')
classifier.fit(np.concatenate([fTrain,fDev],axis=0),np.concatenate([lTrain,lDev],axis=0))

predTr = classifier.predict_proba(np.concatenate([fTrain,fDev],axis=0))
predTr2 = classifier.predict_proba(fTest)
predTe = classifier.predict(fTest)

print('Accuracy: %.2f' % accuracy_score(lTest,predTe))
# print(precision_recall_fscore_support(lTest,predTe))

with open(pickleDir+'svm_Freq_' + type + '_' + target + '_' + source + '_best.pickle','wb') as f:
    pickle.dump([predTr,predTr2,np.concatenate([lTrain,lDev]),lTest],f)