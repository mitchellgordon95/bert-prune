import os
import sys
import shutil
import random
from os.path import join as pathjoin

random.seed(1)

if len(sys.argv) < 2:
    print("Usage: train_test_split.py [data_dir]")
    exit()

data_dir = sys.argv[1]

data_files = [fname for fname in os.listdir(data_dir) if
              os.path.isfile(pathjoin(data_dir, fname))]

dev_size = int(len(data_files) * .01)
print(f'Found {len(data_files)} data files, splitting {dev_size} off for dev and test')

os.mkdir(pathjoin(data_dir, 'train'))
os.mkdir(pathjoin(data_dir, 'test'))
os.mkdir(pathjoin(data_dir, 'dev'))

random.shuffle(data_files)

for fname in data_files[:dev_size]:
    shutil.move(pathjoin(data_dir, fname), pathjoin(data_dir, 'dev'))

for fname in data_files[dev_size:dev_size*2]:
    shutil.move(pathjoin(data_dir, fname), pathjoin(data_dir, 'test'))

for fname in data_files[dev_size*2:]:
    shutil.move(pathjoin(data_dir, fname), pathjoin(data_dir, 'train'))
