wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 -o data/enwiki-latest-pages-articles.xml.bz2
bunzip2 data/enwiki-latest-pages-articles.xml.bz2
# TODO - this should be a package or something
git clone git@github.com:attardi/wikiextractor.git data/wikiextractor
python data/wikiextractor/WikiExtractor.py --json -o data/enwiki enwiki-latest-pages-articles.xml
