import re, sys, os.path, shutil

from os import listdir, mkdir
from os.path import isfile, join, isdir, exists

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

# example directories
# you should put names of your directories
target_dir = 'target_dir'
dest_dir = 'dest_dir'

if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

dirs = get_dirs(target_dir, True)
files = get_files(target_dir, True)

# make directory if not existing
for dir1 in dirs:
    dir2 = dest_dir + '/' + dir1
    if not os.path.exists(dir2):
        os.mkdir(dir2)
        
# copy files if not existing
exts = ['py', 'ipynb', 'sh', 'rb', 'md']
for file1 in files:
    file_src = target_dir + '/' + file1
    file_dest = dest_dir + '/' + file1
    if not os.path.exists(file_dest):
        ext1 = os.path.splitext(file1)[1][1:]
        if ext1 in exts:
            shutil.copy(file_src, file_dest)
