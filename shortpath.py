from itertools import chain
def shortest_path(entities, tree):
    #get entity length
    l = len(entities)
    ret = []
    #get path of each entity to root
    for i1 in range(0, l):
        cur_path = []
        cur_path.append(entities[i1])
        cur_par = tree[entities[i1]]
        while(cur_par!=-1):
            cur_path.append(cur_par)
            cur_par=tree[cur_par]
        ret.append(cur_path)
#    #delete the same paths at the end of list
#    same = 1
#    while (same):
#        temp_first=[]
#        for i2 in range(0,l):
#            if(len(ret[i2])==0):
#                same = 0
#            else:
#                temp_first.append(ret[i2][-1])
#        if(len(temp_first)==l):
#            temp_set=set(temp_first)
#            if(len(temp_set)==1):
#                for i3 in range(0,l):
#                    ret[i3]=ret[i3][0:-1]
#            else:
#                same = 0
#        else:
#            same = 0
#    #add the new root so everypath ends at the same point
#    for i5 in range(0,l):
#        if(len(ret[i5])>0):
#            new_root=tree[ret[i5][-1]]
#            break
#    for i4 in range(0,l):
#        ret[i4].append(new_root)
    return ret

def tree_list(tree):
    l = len(tree)
    ret = [None]*l
    for i in range(0,l):
        temp_place = tree[i][2]-1
        ret[temp_place] = tree[i][1]-1
    if None in ret:
        return None
    else:
        return ret

def list_path(paths):
    ret = list(chain.from_iterable(paths))
    ret = set(ret)
    ret = list(ret)
    return ret
