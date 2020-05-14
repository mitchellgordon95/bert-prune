git clone git@github.com:soskek/bookcorpus.git data/bookcorpus
pip install -r data/bookcorpus/requirements.txt
python data/bookcorpus/download_files.py --list url_list.jsonl --out out_txts --trash-bad-count
