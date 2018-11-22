#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 20:39:36 2018

@author: luojiayi
"""
import numpy as np
import json
from stanfordcorenlp import StanfordCoreNLP
import os
from implement import *
nlp = StanfordCoreNLP(r'/Fall 2018/CS412/project/stanfordcorenlp')
lst = os.listdir("./origin")
# Input
# ==================================================== #
#filename = "./origin/test.txt" # Original file
# ==================================================== #

def Entity_Extract(target, list1, list2):
    l = len(list1)
    # assert(np.floor(l/2) == l/2)
    entity_label = []
    entity_set = []
    for i in range(int(l/2)):
        temp1 = target[list1[2*i]+1:list2[2*i]]
        entity_label.append(temp1)
        temp2 = target[list2[2*i]+1:list1[2*i+1]]
        entity_set.append(temp2)
    assert(len(entity_label) == len(entity_set))
    return entity_set, entity_label
        
def Line_Parse(target):
    list_1 = [];
    list_2 = [];
    l = len(target)
    for j in range(l):
        if target[j] == '<':
            list_1.append(j)
        elif target[j] == '>':
            list_2.append(j)
    # assert(len(list_1) == len(list_2))
    return list_1, list_2

def Line_Rephrase(target, list1, list2):
    l = len(list1)
    ret = ''
    if l == 0:
        ret = target
    else:
        ret += target[0:list1[0]]
        for i in range(int(l-1)):
            ret += target[list2[i]+1:list1[i+1]]
        if len(list2) == 0:
            ret += target[1:]
        else:

            ret += target[list2[-1]+1:]
    return ret

def get_entity_list(dict, sen):
    ret = []
    for i in range(len(sen)):
        if sen[i] in dict:
            ret.append(i)
    return ret

def process_file(filename):
    f = open(filename,"r") 
    line_count = 0;
    text = []
    for x in f:
        line_count += 1;
        text.append(x)
    #    print(x)
    f.close()
    rephrase = "./temp/test_rephrase.txt"
    text_new = []
    f1 = open(rephrase,"w")
    rephrase = {}
    for i in range(line_count):
        list1, list2 = Line_Parse(text[i])
        temp1, temp2 = Entity_Extract(text[i], list1,list2)
        for j in range(len(temp1)):
            rephrase[temp1[j]] = temp2[j]
        temp = Line_Rephrase(text[i], list1, list2)
        text_new.append(temp)
        f1.write(temp)
        if (temp[-1] != '\n' and temp[-1] != '\t' and temp[-1] != '\t\n'):
            f1.write('\n')
    f1.close() 
    return rephrase

def process_input(filename):
    # Input
    # ============================================ #
#    filename = 'test_rephrase.txt'
    jsonname1 = './temp/result.json'
    jsonname2 = './temp/result1.json'
    # nlp = StanfordCoreNLP(r'/Fall 2018/CS412/project/stanfordcorenlp')
    # ============================================ #
    
    f = open(filename,"r") 
    line_count = 0;
    text = []
    for x in f:
        line_count += 1;
        text.append(x)
    f.close()
    data = {}
    for i in range(line_count):
        temp = nlp.dependency_parse(text[i])
        # temp1 = nlp.word_tokenize(text[i])
        data[i] = temp
        
    json_str = json.dumps(data)
    
    with open(jsonname1,'w+') as f:
        json.dump(data, f)

    data1 = {}
    for i in range(line_count):
#        temp = nlp.dependency_parse(text[i])
        temp1 = nlp.word_tokenize(text[i])
        data1[i] = temp1
        
    json_str = json.dumps(data1)   
    with open(jsonname2,'w+') as f:
        json.dump(data1, f)
    # nlp.close() 


def final_process(filename):
    rephrase = process_file(filename)
    process_input("./temp/test_rephrase.txt")
    filename = './temp/result.json'
    with open(filename, 'r') as f1:
        data = json.load(f1)
    filename = './temp/result1.json'
    with open(filename, 'r') as f2:
        sentences = json.load(f2)
    return data, rephrase, sentences

def save_file(pattern, savename):
    data = {}
    for i in range(len(pattern)):
        data[i] = pattern[i]
    with open(savename,'w+') as f:
        json.dump(data, f)

for filename in lst:
    if filename[0] != '.':
        print("Processing: ",filename)
        filename1 = "./origin/" + filename
        data, rephrase, sentences = final_process(filename1)
        pattern = get_patterns(data,rephrase,sentences)
        savename = "./modified/" + "new_" + filename
        save_file(pattern, savename)
nlp.close() 

