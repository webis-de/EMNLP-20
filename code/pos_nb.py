import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
import pickle
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from scipy import spatial
import math
from sklearn.manifold import TSNE
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
import sys
from sklearn.metrics import accuracy_score
import random
import json
import os
import copy
from scipy.special import softmax

dataDir = '../../data/emnlp19-BASIL/data/'
pickleDir = '../../pickle_test/'
pickleBERTDir = '../../../transformers/output/bert_local/'

times = 100

low = 0.000000001

source = 'gt'
# source = 'pred'

# target = 'info'
# target = 'lex'
target = 'any'

print('You are running:', source, 'sentence-bias,', target, 'bias:')



if target == 'any':
    pickleBERTDir = '../../../transformers/output/bert_local/'
else:
    pickleBERTDir = '../../../transformers/output/bert_local_' + target + '/'

def trainGMM(timestep,bias,bias_dev,gmm_range,rint):
    b = np.array([[x[timestep]] for x in bias])
    gmms = [GaussianMixture(n_components=g_size,random_state=rint,reg_covar=1e-02).fit(b) for g_size in gmm_range]
    aics = np.array([m.aic(b) for m in gmms])
    best_n = np.argmin(aics)
    
    return best_n+1, gmms[best_n]

def checkGM(x, mean, var, weight):
    mu = mean
    variance = var
    sigma = math.sqrt(variance)
    xx = np.linspace(x-0.01, x+0.01, 10)
    return sum(stats.norm.pdf(xx, mu, sigma))*0.02*weight

def testGMM(ob, gmms, trans, pr_gmm, pr_trans, bias,times):
    logPrs_b = np.zeros(len(bias)) # of each flow
    logPrs_nb = np.zeros(len(bias)) # of each flow

    pr_loc_pres = []
    for t in range(times):
        bs = [x[t] for x in bias]
        
        prs = []
        prs_nb = []
        pr_locs = []
        
        for idxb,b in enumerate(bs):            
            pr_loc = gmms[t].predict([[b]])

            pr = pr_gmm[t][pr_loc]
            pr_nb = 1-pr_gmm[t][pr_loc]
            
            pr_locs.append(pr_loc)
                
            prs.append(pr)
            prs_nb.append(pr_nb)
        
        pr_loc_pres = [loc for loc in pr_locs]
        
        prs = np.array(prs)
        prs_nb = np.array(prs_nb)
        
        logPrs_b += np.log(prs.flatten()) 
        logPrs_nb += np.log(prs_nb.flatten()) 
    
    logPrs_b += np.log(np.ones(len(bias))*ob)*(1-times) 
    logPrs_nb += np.log(np.ones(len(bias))*(1-ob))*(1-times) 

    pred = np.argmax(np.stack([logPrs_nb,logPrs_b]),axis=0)

    return pred

def estTrans(gmms, bias,len_b):
    bLen = len(bias[0]) # datalen
    bsLen = len(bias) # pattern

    transPs = []
    pr_gmm = []
    pr_trans = []
    
    if bLen == 1:
        this_gmms = gmms[0]
        this_len = len(this_gmms.means_)
        this_C = np.ones(this_len)
        this_C_b = np.ones(this_len)

        for j in range(bsLen):
            b = bias[j][0]
            this_loc = this_gmms.predict([[b]])
            
            this_C[this_loc] += 1
                        
            if j < len_b:
                this_C_b[this_loc] += 1

        this_pr_gmm = this_C_b / (this_C + 1)
        pr_gmm.append(this_pr_gmm)
    else:
        for i in range(bLen-1):
            this_gmms = gmms[i]
            next_gmms = gmms[i+1]
            
            this_len = len(this_gmms.means_)
            next_len = len(next_gmms.means_)
            
            transC = np.ones((this_len,next_len))
            pr_transC = np.ones((this_len,next_len))
            
            this_C = np.ones(this_len)
            this_C_b = np.ones(this_len)
            
            next_C = np.ones(next_len)
            next_C_b = np.ones(next_len)
    
            for j in range(bsLen):
                b = bias[j][i]
                this_loc = this_gmms.predict([[b]])
                
                this_C[this_loc] += 1
                
                b = bias[j][i+1]
                
                next_loc = next_gmms.predict([[b]])
                next_C[next_loc] += 1
                
                transC[this_loc,next_loc] += 1
                if j < len_b:
                    pr_transC[this_loc,next_loc] += 1
                    this_C_b[this_loc] += 1
                    next_C_b[next_loc] += 1
                    
            pr_transP = pr_transC / (transC + 1)
            transP = np.array([transC[idx,:]/np.sum(transC,axis=1)[idx] for idx in range(this_len)])
            
            transPs.append(transP)
            pr_trans.append(pr_transP)
    
            this_pr_gmm = this_C_b / (this_C + 1)
            pr_gmm.append(this_pr_gmm)

            if i == bLen-2: # last one
                next_pr_gmm = next_C_b / (next_C + 1)

                pr_gmm.append(next_pr_gmm)
    
    return transPs, pr_gmm, pr_trans



def norValues(valuess,norLen=times):
    rValuess = []
    
    for values in valuess:
        rValues = []
        for value in values:
            rValue = np.zeros((norLen))
            norT = 100/(len(value)-1)
            
            if len(value) > 0:
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
        for s in jsonData['body']:
            sent_text = s['sentence'].encode('ascii','ignore').decode('utf-8')
            
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


# combine L&R
bias_flows_trB = np.concatenate([bias_flows_trL,bias_flows_trR],axis=0)
bias_flows_deB = np.concatenate([bias_flows_deL,bias_flows_deR],axis=0)
bias_flows_teB = np.concatenate([bias_flows_teL,bias_flows_teR],axis=0)


max_dataPointsLen = 10
min_dataPointsLen = 1
dataPointsLen_step = 1

max_mix_rate = 5
min_mix_rate = 1
mix_step = 1

best_de = 0
best_rint = 0
best_mix = 0
best_data = 0
best_sampling = 'none'

samplings = ['avg','sample','max']

for s in samplings:
    for mixtureSize in range(min_mix_rate,max_mix_rate+mix_step,mix_step):
        for dataPointsLen in range(min_dataPointsLen, max_dataPointsLen+dataPointsLen_step, dataPointsLen_step):
            sample_rate = times/dataPointsLen
            sampled_times = int(times/sample_rate)

            s_bias_flows_trB = sampling(bias_flows_trB, sample_rate, s)
            s_bias_flows_trC = sampling(bias_flows_trC, sample_rate, s)
              
            s_bias_flows_teB = sampling(bias_flows_teB, sample_rate, s)
            s_bias_flows_teC = sampling(bias_flows_teC, sample_rate, s)
              
            s_bias_flows_deB = sampling(bias_flows_deB, sample_rate, s)
            s_bias_flows_deC = sampling(bias_flows_deC, sample_rate, s)
            
            s_bias_flows_tr = np.concatenate([s_bias_flows_trB, s_bias_flows_trC], axis=0)
            s_bias_flows_te = np.concatenate([s_bias_flows_teB, s_bias_flows_teC], axis=0)
            s_bias_flows_de = np.concatenate([s_bias_flows_deB, s_bias_flows_deC], axis=0)
            
            len_b = s_bias_flows_trB.shape[0]

            trs = []
            tes = []
            des = []
            
            n_gmms = range(1,mixtureSize+1)
            
            for rint in range(10):
                gmms=[]
                
                gmms_n=[]
                for i in range(sampled_times):
                    gmm_n, gmm = trainGMM(i, s_bias_flows_tr, s_bias_flows_de, n_gmms,rint)
                    
                    gmms.append(gmm)
                    gmms_n.append(gmm_n)
                    
                
                trans, pr_gmm, pr_trans = estTrans(gmms,s_bias_flows_tr,len_b)

                ob = len_b/s_bias_flows_tr.shape[0]
                
                pred0 = testGMM(ob, gmms, trans, pr_gmm, pr_trans, s_bias_flows_trC, sampled_times)
                ans0 = np.zeros_like(pred0)
            
                pred1 = testGMM(ob, gmms, trans, pr_gmm, pr_trans, s_bias_flows_trB, sampled_times)
                ans1 = np.ones_like(pred1)
                
                this_tr = accuracy_score(np.concatenate([ans0,ans1]),np.concatenate([pred0,pred1]))
                
                tr_all = accuracy_score(np.concatenate([ans0,ans1]),np.ones_like(np.concatenate([ans0,ans1])))

                pred0 = testGMM(ob, gmms, trans, pr_gmm, pr_trans, s_bias_flows_teC, sampled_times)
                ans0 = np.zeros_like(pred0)
            
                pred1 = testGMM(ob, gmms, trans, pr_gmm, pr_trans, s_bias_flows_teB, sampled_times)
                ans1 = np.ones_like(pred1)
                
                this_te = accuracy_score(np.concatenate([ans0,ans1]),np.concatenate([pred0,pred1]))
                
                te_all = accuracy_score(np.concatenate([ans0,ans1]),np.ones_like(np.concatenate([ans0,ans1])))
  
                pred0 = testGMM(ob, gmms, trans, pr_gmm, pr_trans, s_bias_flows_deC, sampled_times)
                ans0 = np.zeros_like(pred0)
            
                pred1 = testGMM(ob, gmms, trans, pr_gmm, pr_trans, s_bias_flows_deB, sampled_times)
                ans1 = np.ones_like(pred1)
                
                this_de = accuracy_score(np.concatenate([ans0,ans1]),np.concatenate([pred0,pred1]))
                
                de_all = accuracy_score(np.concatenate([ans0,ans1]),np.ones_like(np.concatenate([ans0,ans1])))
 
                trs.append(this_tr)
                tes.append(this_te)
                des.append(this_de)
                
                if this_de > best_de:
                    best_de = this_de
                    best_rint = rint
                    best_mix = mixtureSize
                    best_data = dataPointsLen
                    best_sampling = s
            
print('*************')
print('Best hyperparameters:')
print('mixture size:', best_mix)
print('position len:', best_data)
print('best sampling', best_sampling)

sample_rate = times/best_data

sampled_times = int(times/sample_rate)

s_bias_flows_trB = sampling(bias_flows_trB, sample_rate, best_sampling)
s_bias_flows_trC = sampling(bias_flows_trC, sample_rate, best_sampling)
  
s_bias_flows_teB = sampling(bias_flows_teB, sample_rate, best_sampling)
s_bias_flows_teC = sampling(bias_flows_teC, sample_rate, best_sampling)
  
s_bias_flows_deB = sampling(bias_flows_deB, sample_rate, best_sampling)
s_bias_flows_deC = sampling(bias_flows_deC, sample_rate, best_sampling)

s_bias_flows_trB = np.concatenate([s_bias_flows_trB, s_bias_flows_deB], axis=0)
s_bias_flows_trC = np.concatenate([s_bias_flows_trC, s_bias_flows_deC], axis=0)


s_bias_flows_tr = np.concatenate([s_bias_flows_trB, s_bias_flows_trC], axis=0)
s_bias_flows_te = np.concatenate([s_bias_flows_teB, s_bias_flows_teC], axis=0)


len_b = s_bias_flows_trB.shape[0]

n_gmms = range(1,best_mix+1)

gmms=[]

gmms_n=[]
for i in range(sampled_times):
    gmm_n, gmm = trainGMM(i, s_bias_flows_tr, s_bias_flows_de, n_gmms,best_rint)
    
    gmms.append(gmm)
    gmms_n.append(gmm_n)
    

trans, pr_gmm, pr_trans = estTrans(gmms,s_bias_flows_tr,len_b)
    
ob = len_b/s_bias_flows_tr.shape[0]

pred0 = testGMM(ob, gmms, trans, pr_gmm, pr_trans, s_bias_flows_trC, sampled_times)
ans0 = np.zeros_like(pred0)

pred1 = testGMM(ob, gmms, trans, pr_gmm, pr_trans, s_bias_flows_trB, sampled_times)
ans1 = np.ones_like(pred1)

this_tr = accuracy_score(np.concatenate([ans0,ans1]),np.concatenate([pred0,pred1]))

tr_all = accuracy_score(np.concatenate([ans0,ans1]),np.ones_like(np.concatenate([ans0,ans1])))
    
pred0 = testGMM(ob, gmms, trans, pr_gmm, pr_trans, s_bias_flows_teC, sampled_times)
ans0 = np.zeros_like(pred0)

pred1 = testGMM(ob, gmms, trans, pr_gmm, pr_trans, s_bias_flows_teB, sampled_times)
ans1 = np.ones_like(pred1)

this_te = accuracy_score(np.concatenate([ans0,ans1]),np.concatenate([pred0,pred1]))


print('Accuracy: %.2f' %this_te)

