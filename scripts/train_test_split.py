import os
import sys
import shutil
import random
from os.path import join as pathjoin

random.seed(1)

if len(sys.argv) < 5:
    print("Usage: train_test_split.py [data_dir] [train_dir] [test_dir] [dev_dir]")
    exit()

data_dir = sys.argv[1]
train_dir = sys.argv[2]
test_dir = sys.argv[3]
dev_dir = sys.argv[4]

for dir_name in [train_dir, test_dir, dev_dir]:
    os.makedirs(dir_name)

data_files = [fname for fname in os.listdir(data_dir) if
              os.path.isfile(pathjoin(data_dir, fname))]

dev_size = int(len(data_files) * .01)
print(f'Found {len(data_files)} data files, splitting {dev_size} off for dev and test')

random.shuffle(data_files)

for fname in data_files[:dev_size]:
    os.link(pathjoin(data_dir, fname), pathjoin(dev_dir, fname))

for fname in data_files[dev_size:dev_size*2]:
    os.link(pathjoin(data_dir, fname), pathjoin(test_dir, fname))

for fname in data_files[dev_size*2:]:
    os.link(pathjoin(data_dir, fname), pathjoin(train_dir, fname))
