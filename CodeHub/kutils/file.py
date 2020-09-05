# coding: utf-8

# author : bwlee@kbfg.com

import re, sys, os.path

from os import listdir
from os.path import isfile, join, isdir

def get_dirs(dir_s, recursive=None):
    current_dirs = [dir_s1 for dir_s1 in listdir(dir_s) if isdir(join(dir_s, dir_s1))]
    if recursive is None:
        return current_dirs
    
    for dir_s1 in current_dirs:
        combine_dir = lambda x: join(dir_s1, x)
        current_dirs = current_dirs + list(map(combine_dir, get_dirs(join(dir_s, dir_s1), recursive=True)))

    return current_dirs

def get_files(dir_s, recursive=None):
    current_files = [f1_s for f1_s in listdir(dir_s) if isfile(join(dir_s, f1_s))]
    if recursive is None:
        return current_files
    
    current_dirs = [dir_s1 for dir_s1 in listdir(dir_s) if isdir(join(dir_s, dir_s1))]
    for dir_s1 in current_dirs:
        combine_dir = lambda x: join(dir_s1, x)
        current_files = current_files + list(map(combine_dir, get_files(join(dir_s, dir_s1), recursive=True)))
    
    return current_files

def list_dirs(dir_s, recursive=None):
    return {'files': get_files(dir_s, recursive), 'dirs': get_dirs(dir_s, recursive)}

def dump(obj, f1_s):
    import re
    ext = re.search('\.(\w+)$', f1_s).groups()[0]  
    if ext == 'pk':
        import pickle as pk
        with open(f1_s, 'wb') as f1:
            pk.dump(obj, f1)        
    elif ext == 'npz':
        import numpy as np
        np.savez(f1_s, obj)
    elif ext == 'yaml':
        import yaml
        with open(f1_s, "w") as f1:
            yaml.dump(obj, f1)
    elif ext == 'json':
        import json
        with open(f1_s, "w") as f1:
            json.dump(obj, f1, indent=4, sort_keys=True)
        
def load(f1_s):
    import re
    ext = re.search('\.(\w+)$', f1_s).groups()[0]  

    if ext == 'pk':
        import pickle as pk
        with open(f1_s, 'rb') as f1:
            return pk.load(f1)
    elif ext == 'npz':
        import numpy as np
        return np.load(f1_s)
    elif ext == 'yaml':
        import yaml
        with open(f1_s) as f1:
            return yaml.load(f1)
    elif ext == 'json':
        import json
        with open(f1_s) as f1:
            return json.load(f1)


if __name__ == '__main__':
    print( get_files('.') )
    print()
    print( get_dirs('.', True) )
    print()
    print()
    print(list_dirs('.', True) )

    obj1 = {"abc": {"val1": 3, "val2": "kkkkk"}, 
            "bcdf": {"val1": 33, "val2": "bbbb"}}

    f1_s = 'test2.json'
    f2_s = 'test2.yaml'
    dump(obj1, f1_s)
    dump(obj1, f2_s)
    
    print( load(f1_s) )
    print( load(f2_s) )
