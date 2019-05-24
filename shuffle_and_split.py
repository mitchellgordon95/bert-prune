import sys
import random
import subprocess
random.seed(1)

if len(sys.argv) != 3:
    print ('shuffle_and_split.py [input-file] [num-output]')
    print('Shuffles documents in a file (separated by newlines) and writes them to multiple sub files.')
    print('Deletes the original.')
    exit()

input_fn = sys.argv[1]
num_output = int(sys.argv[2])

print("Reading the file")
documents = []
doc = []
for line in open(input_fn, 'r'):
    if line != '\n':
        doc.append(line)
    else:
        documents.append(doc)
        doc = []

if doc and doc != documents[-1]:
    documents.append(doc)

print("Shuffling the file")
random.shuffle(documents)

print("Writing the output files")
docs_per_file = len(documents) // num_output
for out_index in range(num_output):
    print(f'Writing output file {out_index}')
    out_f = open(f'{input_fn}_{out_index}', 'w+')
    begin = out_index*docs_per_file
    end = -1 if out_index == num_output - 1 else (out_index + 1) * docs_per_file
    for doc in documents[begin:end]:
        for line in doc:
            print(line, file=out_f, end='')
        print(file=out_f)
