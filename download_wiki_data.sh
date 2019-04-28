#! /bin/bash
#$ -cwd
#$ -V
#$ -l num_proc=2,h_rt=10:00:00
#$ -j y
#$ -m ase
#$ -M mitchell.gordon95@gmail.com

# Above: configuration for running things on the univa grid engine

mkdir data
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 -o data/enwiki-latest-pages-articles.xml.bz2
bunzip2 data/enwiki-latest-pages-articles.xml.bz2
git clone git@github.com:attardi/wikiextractor.git data/wikiextractor
python data/wikiextractor/WikiExtractor.py --json -o data/enwiki data/enwiki-latest-pages-articles.xml
