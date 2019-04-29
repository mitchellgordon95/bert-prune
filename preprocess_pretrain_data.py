import spacy
import io
import subprocess
from spacy.lang.en import English
import os
import json
from univa_grid import TaskRunner

# Number of tasks to run concurrently
TASKS = 8

# Locations of wikipedia dump and bookcorpus
WIKI_LOC = 'data/enwiki'
BOOK_LOC = 'data/bookcorpus/out_txts'

# Spacy model for sentence segmentation (among other things, but we only do segmentation)
nlp_lg = spacy.load("en_core_web_lg")
# nlp_lg.max_length = 1500000

# Also spacy, but using the rule-based sentencizer
# Note: we use this for bookcorpus, since the dependency-parser-based sentencizer
# tends to put quotation marks as their own sentences, and bookcorpus has a lot of quotations.
nlp = English()
nlp.add_pipe(nlp.create_pipe("sentencizer"))
# Increase the max length of documents from 1M to 5M characters for this sentencizer. Books are long.
# We can do this because we're using the rule-based sentencizer, not the parser, so we don't need as much memory.
nlp.max_length = 5000000

def write_doc(doc, output_f):
    """From the BERT readme:
    The input is a plain text file, with one sentence per line. (It is
    important that these be actual sentences for the "next sentence prediction"
    task). Documents are delimited by empty lines.
    """
    for sent in doc.sents:
        text = sent.text.replace('\n', ' ')
        if text == '"':
            print(text, file=output_f, end='')
        elif text:
            print(text, file=output_f)
    print(file=output_f)

def maybe_print_progress(doc_counter):
    if doc_counter % 10000 == 0:
        print("Processed {} documents".format(doc_counter))

def process_docs(task_id):
    print("Processing every {}'th doc".format(task_id))
    doc_counter = 0
    with open('data/pretrain_sentencized_{}.txt'.format(task_id), 'w+') as output_f:
        # Do all the wikipedia stuff
        for subdir in os.listdir(WIKI_LOC):
            for wiki_fname in os.listdir(os.path.join(WIKI_LOC, subdir)):
                input_path = os.path.join(WIKI_LOC, subdir, wiki_fname)
                if os.path.isfile(input_path):
                    # Every line is a json object representing a wikipedia article
                    for line in open(input_path, 'r'):
                        if doc_counter % TASKS == task_id:
                            doc = json.loads(line)
                            # Use the large spacy model for wikipedia data, since it's supposed to be nice.
                            write_doc(nlp_lg(doc['text']), output_f)
                        doc_counter += 1
                        maybe_print_progress(doc_counter)

        # Do all the bookcorpus stuff
        for book_fname in os.listdir(BOOK_LOC):
            input_path = os.path.join(BOOK_LOC, book_fname)
            if os.path.isfile(input_path):
                if doc_counter % TASKS == task_id:
                    # Use the rule-based sentecizer for books, because it handles quotes better.
                    write_doc(nlp(open(input_path).read()), output_f)
                doc_counter += 1
                maybe_print_progress(doc_counter)

if __name__ == '__main__':
    task_runner = TaskRunner()
    for task_id in range(TASKS):
        task_runner.do_task(process_docs, task_id)
