mkdir data
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 -o data/enwiki-latest-pages-articles.xml.bz2
bunzip2 data/enwiki-latest-pages-articles.xml.bz2
git clone git@github.com:attardi/wikiextractor.git data/wikiextractor
python data/wikiextractor/WikiExtractor.py -o data/enwiki data/enwiki-latest-pages-articles.xml
