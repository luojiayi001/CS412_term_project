import json
from anytree import Node, RenderTree
from copy import deepcopy
import numpy as np
import os
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import matplotlib.pyplot as plt
# change the tense of verb
def change_tense(sen):
    tags = nltk.pos_tag(sen)
    l = len(tags)
    ret = deepcopy(sen)
    count = 0
    for i in range(0,l-1):
        if (sen[i] in ['will','would']) and (tags[i+1][1] == 'VB'):
            del ret[i-count]
            count = count+1
        elif (sen[i] in ['is','be','are','was','were']) and (tags[i+1][1] == 'VBG'): #is doing -> do
            ret[i-count+1] = lemmatizer.lemmatize(sen[i+1],'v')
            del ret[i-count]
            count = count+1
        elif (sen[i] in ['have','has']) and (tags[i+1][1] == 'VBN'): #has done -> do
            ret[i-count+1] = lemmatizer.lemmatize(sen[i+1],'v')
            del ret[i-count]
            count = count+1
        elif (tags[i][1] == 'VBD') or (tags[i][1] == 'VBZ') or (tags[i][1] == 'VBG'):
            ret[i-count] = lemmatizer.lemmatize(sen[i],'v')
    return ret
#read in data as a list
lst = os.listdir("./noisy_or_data")
train = []
for filename in lst:
    temp = []
    if filename[0] != '.':
        print("Processing: ",filename)
        filename1 = "./noisy_or_data/" + filename
        with open(filename1, 'r') as f:
            train1 = json.load(f)
        for key in train1.keys():
            temp.append(train1[key])
        train.append(temp)
#use dictionary to get the list of different patterns
pattern_dict = {}
for i in range(0,len(train)):
    for before_CT in train[i]:
        after_CT = change_tense(before_CT)
        separator = ' '
        mixed_pattern = separator.join(after_CT)
        if mixed_pattern in pattern_dict:
            pattern_dict[mixed_pattern].append(i)
        else:
            pattern_dict[mixed_pattern] = [i]
#get the list of occured patterns in each paper
pattern_list = list(pattern_dict.keys())

pattern_dict_f = {}
for_graph = []
for cur_key in pattern_list:
    l = len(pattern_dict[cur_key])
    if(l>=10):
        pattern_dict_f[cur_key] = l
        for_graph.append(l)
for_graph.sort(reverse = True)
pattern_list_f = list(pattern_dict_f.keys())
good_patterns = {}
graph_2 = []
for cur_key in pattern_list_f:
    occurance = pattern_dict[cur_key]
    occurance = set(occurance)
    occurance = list(occurance)
    temp_graph = []
    temp_graph.append(len(pattern_dict[cur_key]))
    temp_graph.append(len(occurance))
    graph_2.append(temp_graph)
    if(len(occurance)>=5):
        temp = []
        temp.append(len(occurance))
        temp.append(pattern_dict_f[cur_key])
        good_patterns[cur_key] = temp
#a noisy-or class
class Noisy_OR(object):
    def __init__(self, num_event = 5, num_pattern = 3, D = 10):
        self.num_event = num_event
        self.num_pattern = num_pattern
        self.Pr_ek_1_P_list = []
        self.Pr_zik_1_P_list = []
        self.D = D
        self.qk = np.random.rand(1, num_event)
        #self.qk = np.zeros((1, num_event + 1))
        self.qk[0][0] = 1
        #self.qik = np.random.rand(num_event, num_pattern)
        self.qik = np.random.rand(num_pattern, num_event)
    #e_step uses self.qik and input list of patterns to calculate probabilities and returns the probability of events
    def _e_step(self, p): # p is of size (1, num_pattern)
#        qk = np.random.rand(1, num_event)
#        for i in range(num_event):
#            ret = -q_to_theta(self.noise)
#            for j in range(num_pattern):
#                if c[j] == 1:   
#                    ret -= q_to_theta(self.qik[i][j])
#            qk[i] = 1 - ret
#        self.qk_list += [qk]
        zik = self.cal_zik(p)
        res = np.ones(self.num_event)
        for k in range(self.num_event):
            ret = self.qk[0][k]
            for i in range(self.num_pattern):
                ret *= self.qik[i][k]**zik[i][k]*(1 - self.qik[i][k])**(1 - zik[i][k])
                ret /= (self.qik[i][k]*self.qk[0][k])**zik[i][k]
                ret /= (1 - self.qik[i][k]*self.qk[0][k])**(1 - zik[i][k])
            res[k] = ret
        self.Pr_ek_1_P_list += [res]
        return res
    #m_step updates self.qik and self.qk by the results of e_step
    def _m_step(self):
        for k in range(self.num_event):
            up = 0
            down = 0
            for d in range(self.D):
                pr_zik = self.Pr_zik_1_P_list[d]
                pr_ek = self.Pr_ek_1_P_list[d]
                for k in range(self.num_event):
                    up += pr_zik[:,k]
                    down += pr_ek[k]
            self.qik[:,k] = up/down
            self.qk[0][k] = down/self.D
            self.qk[0][0] = 1
    def cal_zik(self, p):
        qik = self.qik
        qk = self.qk
        zik = np.zeros(qik.shape)
        p_zik_1 = np.zeros(qik.shape)
        for i in range(self.num_pattern):
            for k in range(self.num_event):
                if p[i] == 0:
                    zik[i][k] = 0
                    p_zik_1[i][k] = 0
                else:     
                    zik_1_pi_1 = qik[i][k]*qk[0][k]
                    temp = 1
                    for m in range(self.num_event):
                        if(m!=k):
                            #print(i,m,qik.shape,qk.shape)
                            temp *= (1 - qik[i][m]*qk[0][m])
                    left = 1 - temp
                    right = (1 - qik[i][k]*qk[0][k])
                    zik_0_pi_1 = left*right
                    #print(zik_1_pi_1,zik_0_pi_1)
                    if zik_1_pi_1 >= zik_0_pi_1:
                        zik[i][k] = 1
                    p_zik_1[i][k] = zik_1_pi_1
        self.Pr_zik_1_P_list += [p_zik_1]
        return zik
good_patterns_list = list(good_patterns.keys())
c1 = [0]*len(good_patterns_list)
c = []
for i in range(0,len(lst)):
    c.append(deepcopy(c1))
for i in range(0,len(good_patterns_list)):
    temp = pattern_dict[good_patterns_list[i]]
    for j in temp:
        c[j][i] = 1
# input the number of events to be considered
events = 11
noisy_model = Noisy_OR(events, len(good_patterns_list), len(lst))
cur_qk = noisy_model.qk
print(cur_qk)
# input the maximum steps
steps = 40
for i in range(0,steps):
    print("Step:",i)
    for j in range(0,len(lst)):
        noisy_model._e_step(c[j])
    noisy_model._m_step()
    if(list(noisy_model.qk[0])==list(cur_qk)):
        break
    else:
        cur_qk = noisy_model.qk
print(noisy_model.qk)
print(noisy_model.qik)

plt.plot(for_graph)
graph_2.sort(reverse = True)
graph_2_value = []
for i in range(0, len(graph_2)):
    graph_2_value.append(graph_2[i][0]*(graph_2[i][1]-1))
plt.plot(graph_2_value)
plt.xlabel('frequent pattern ID')
plt.ylabel('F value')
plt.plot(for_graph)
plt.xlabel('frequent pattern ID')
plt.ylabel('F value')
def kl_div(p_x):
    temp_dict = {}
    ret = 0
    for cur_key in p_x:
        try: 
            temp_dict[cur_key] += 1
        except KeyError:
            temp_dict[cur_key] = 1
    qx = len(p_x)/len(lst)
    key_list = list(temp_dict.keys())
    for new_key in key_list:
        px = temp_dict[new_key]/len(p_x)
        ret+=px*np.log(px/qx)
    return ret
graph_3 = []
for cur_key in pattern_list_f:
    temp = []
    temp.append(pattern_dict_f[cur_key])
    kl = kl_div(pattern_dict[cur_key])
    temp.append(kl)
    graph_3.append(temp)
graph_3.sort(reverse = True)
graph_3_value = []
for i in range(0, len(graph_3)):
    graph_3_value.append(graph_3[i][1])
plt.plot(graph_3_value)
plt.xlabel('frequent pattern ID')
plt.ylabel('F value')
plt.grid(True)
plt.savefig('kl_div1.png',quality = 0)

qik5_1 = noisy_model.qik
graph_5_1 = [0]*6
for i in range(0,304):
    temp_m = max(qik5_1[i])
    for j in range(0,6):
        if(qik5_1[i][j]==temp_m):
            graph_5_1[j]+=1
plt.bar([0,1,2,3,4,5],graph_5_1)
qik10_1 = noisy_model.qik
graph_10_1 = [0]*11
for i in range(0,304):
    temp_m = max(qik10_1[i])
    for j in range(0,11):
        if(qik10_1[i][j]==temp_m):
            graph_10_1[j]+=1
plt.bar([0,1,2,3,4,5,6,7,8,9,10],graph_10_1)
