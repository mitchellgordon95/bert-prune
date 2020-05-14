import spacy
import io
import subprocess
from spacy.lang.en import English
import os
import json
import fire

# Spacy model for sentence segmentation (among other things, but we only do segmentation)
nlp_lg = spacy.load("en_core_web_lg")
# nlp_lg.max_length = 1500000

# Also spacy, but using the rule-based sentencizer
# Note: we use this for bookcorpus, since the dependency-parser-based sentencizer
# tends to put quotation marks as their own sentences, and bookcorpus has a lot
# of quotations.
nlp = English()
nlp.add_pipe(nlp.create_pipe("sentencizer"))
# Increase the max length of documents from 1M to 14M characters for this
# sentencizer. Books are long. The longest book in bookcorpus is:
# 13961563 out_txts/682810__debunkanji-chinese-glyphs-used-in-japanese.txt
# We can do this because we're using the rule-based sentencizer, not the
# parser, so we don't need as much memory.
nlp.max_length = 14000000

def write_doc(doc):
    """From the BERT readme:
    The input is a plain text file, with one sentence per line. (It is
    important that these be actual sentences for the "next sentence prediction"
    task). Documents are delimited by empty lines.
    """
    for sent in doc.sents:
        text = sent.text.replace('\n', ' ')
        if text == '"':
            print(text, end='')
        elif text:
            print(text)
    print()

def process_docs(chunk_id: int, total_chunks: int, wiki_loc, book_loc):
    assert chunk_id < total_chunks
    doc_counter = 0
    # Do all the wikipedia stuff
    for subdir in os.listdir(wiki_loc):
        for wiki_fname in os.listdir(os.path.join(wiki_loc, subdir)):
            input_path = os.path.join(wiki_loc, subdir, wiki_fname)
            if os.path.isfile(input_path):
                # Every line is a json object representing a wikipedia article
                for line in open(input_path, 'r'):
                    if doc_counter % total_chunks == chunk_id:
                        doc = json.loads(line)
                        # Use the large spacy model for wikipedia data, since it's supposed to be nice.
                        write_doc(nlp_lg(doc['text']))
                    doc_counter += 1

    # Do all the bookcorpus stuff
    for book_fname in os.listdir(book_loc):
        input_path = os.path.join(book_loc, book_fname)
        if os.path.isfile(input_path):
            if doc_counter % total_chunks == chunk_id:
                # Use the rule-based sentecizer for books, because it handles quotes better.
                write_doc(nlp(open(input_path).read()))
            doc_counter += 1

if __name__ == '__main__':
    fire.Fire(process_docs)
