import spacy
import io
import subprocess
from spacy.lang.en import English
import os
import json

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
        if text and text != '"':
            print(text, file=output_f)
    print(file=output_f)

# TODO (mitchg) - maybe don't write everything to a single file?
with open('data/pretrain_sentencized.txt', 'w+') as output_f:
    # Do all the wikipedia stuff
    for subdir in os.listdir(WIKI_LOC):
        for wiki_fname in os.listdir(os.path.join(WIKI_LOC, subdir)):
            input_path = os.path.join(WIKI_LOC, subdir, wiki_fname)
            if os.path.isfile(input_path):
                # Every line is a json object representing a wikipedia article
                for line in open(input_path, 'r'):
                    doc = json.loads(line)
                    # Use the large spacy model for wikipedia data, since it's supposed to be nice.
                    write_doc(nlp_lg(doc['text']), output_f)

    # Do all the bookcorpus stuff
    for book_fname in os.listdir(BOOK_LOC):
        input_path = os.path.join(BOOK_LOC, book_fname)
        if os.path.isfile(input_path):
            # Use the rule-based sentecizer for books, because it handles quotes better.
            write_doc(nlp(open(input_path).read()), output_f)
