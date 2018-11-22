from itertools import chain
from shortpath import *

def find_ent(words1, ent_dict):
    ent1 = []
    l = len(words1)
    count = 0
    for i in range(0,l):
        if count !=0:
            count = count-1
        else:
            if words1[i] in ent_dict:
                cur_ent = []
                cur_ent.append(i)
                ent1.append(cur_ent)
            if (i<l-1):
                if (words1[i]+' '+words1[i+1]) in ent_dict:
                    cur_ent = []
                    cur_ent.append(i)
                    cur_ent.append(i+1)
                    ent1.append(cur_ent)
                    count = 1
            if (i<l-2):
                if (words1[i]+' '+words1[i+1]+' '+words1[i+2]) in ent_dict:
                    cur_ent = []
                    cur_ent.append(i)
                    cur_ent.append(i+1)
                    cur_ent.append(i+2)
                    ent1.append(cur_ent)
                    count = 2
    return ent1

def ent_pair(ent1):
    l = len(ent1)
    ent2 = []
    if(l<3):
        ent2.append(list(chain.from_iterable(ent1)))
    else:
        for i in range(0,l-1):
            for j in range(i+1,l):
                cur_ent = []
                cur_ent.append(ent1[i])
                cur_ent.append(ent1[j])
                ent2.append(list(chain.from_iterable(cur_ent)))
    return ent2

def add_conj(tree1,ent1,path2):
    l = len(tree1)
    for i in range(0,l):
        if(tree1[i][0]=='cc'):
            conj_par = tree1[i][1]-1
            if(conj_par in ent1):
                if(i>ent1[0] and i<ent1[-1]):
                    path2.append(tree1[i][2]-1)
    return path2

def add_root(path2,tree1):
    path2.append(tree1[0][2]-1)
    return path2

def add_subject(path2,tree1):
    l = len(tree1)
    subs = ['csubj','csubjpass','nsubj','nsubjpass']
    for i in range(0,l):
        if(tree1[i][0] in subs and ((tree1[i][1]-1) in path2)):
            path2.append(tree1[i][2]-1)
    return path2

def add_punct(path2,tree1):
    l = len(tree1)-1
    for i in range(0,l):
        if(tree1[i][0]=='punct'):
            path2.append(tree1[i][2]-1)
    return path2

def whole_path(tree1, ent1,punct):
    tree2 = tree_list(tree1)
    if tree2 == None:
        return None
    path1 = shortest_path(ent1,tree2)
    path2 = list_path(path1)
    path2 = add_conj(tree1,ent1,path2)
    path2 = add_root(path2,tree1)
    path2 = add_subject(path2,tree1)
    if(punct):
        path2 = add_punct(path2,tree1)
    path2 = set(path2)
    path2 = list(path2)
    path2.sort()
    return path2

def show_pattern(path2, words1, ent_dict):
    words2 = []
    words3 = []
    path2.sort()
    l = len(path2)
    for i in range(0,l):
        words2.append(words1[path2[i]])
    count = 0
    for i in range(0,l):
        if count == 0:
            found = 0
            if words2[i] in ent_dict:
                found = 1
                words3.append('$'+ent_dict[words2[i]])
            if (i<l-1):
                cur_key = words2[i]+' '+words2[i+1]
                if (cur_key in ent_dict):
                    found = 1
                    words3.append('$'+ent_dict[cur_key])
                    count = 1
            if (i<l-2):
                cur_key = words2[i]+' '+words2[i+1]+' '+words2[i+2]
                if (cur_key in ent_dict):
                    found = 1
                    words3.append('$'+ent_dict[cur_key])
                    count = 2
            if (found == 0):
                words3.append(words2[i])
        else:
            count = count-1
    return words3
    
def pattern_extraction(tree1,ent_dict,words1,punct):
    ent1 = find_ent(words1,ent_dict)
    if(len(ent1)<2):
        return []
    ent2 = ent_pair(ent1)
    ret = []
    l = len(ent2)
    for i in range(0,l):
        path2 = whole_path(tree1,ent2[i],punct)
        if(path2!=None):
            ret.append(show_pattern(path2,words1,ent_dict))
    return ret