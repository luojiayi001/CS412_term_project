from add_things import *
from shortpath import *

import numpy as np

def get_patterns(data,rephrase,sentences):
    dict_keys = list(data.keys())    
    l = len(dict_keys)
    patterns = {}
    for i in range(0,l):
        tree = data[dict_keys[i]]
        words = sentences[dict_keys[i]]
        pattern = pattern_extraction(tree,rephrase,words,1)
        patterns[dict_keys[i]] = pattern
        all_patterns = []
    for i1 in range(0,l):
        cur_pattern = patterns[dict_keys[i1]]
        l1 = len(cur_pattern)
        if(l1>0):
            for i2 in range(0,l1):
                all_patterns.append(cur_pattern[i2])
    return all_patterns
