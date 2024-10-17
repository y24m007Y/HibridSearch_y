import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import MeCab
import unidic
import re
import string
import json
import argparse
from pathlib2 import Path
import time
import sys, os

pattern = re.sub('-', '', string.punctuation) + '。、'
with open('custom_stopwords_ja.json', mode='r') as f:
    jsondata = json.load(f)
stopwords = jsondata['stopwords']
tagger = MeCab.Tagger(f'-Owakati {unidic.DICDIR}')
text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
)
model = SentenceTransformer("sentence-transformers/LaBSE")
start = time.perf_counter()

def preprocess(text):
    text = re.sub(r'\n', '', text)
    text = re.sub(r"[{}]".format(pattern), '', text)
    sub_words = tagger.parse(text).split()
    text = "".join([word for word in sub_words if word not in stopwords])
    sub_texts = text_spliter.create_documents([text])
    sub_texts = [sub_text.page_content for sub_text in sub_texts]
    return sub_texts

def text_to_vec(texts):
    total_vec = None
    for text in texts:
        text_vec = model.encode(text)
        if total_vec is None:
            total_vec = text_vec
        else:
            total_vec += text_vec
    return total_vec / len(texts)

def corpus_to_vec(corpus, filename):
    corpus = corpus.apply(lambda x: preprocess(x))
    corpus_vec = corpus.apply(lambda x: text_to_vec(x))
    savepath = Path("corpus_vector/")
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    corpus_vec.to_pickle(savepath.joinpath(f"{filename}"+".pkl"))
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="please textfile or csvfile")
    args = parser.parse_args()
    filename, filetype = args.file.split('.')
    filepath = Path("livedoor/csv/")
    filepath = filepath.joinpath(args.file) if args.file in [f.name for f in filepath.glob("*.csv")] else "該当のファイルがありません"
    if filetype == 'txt':
        contents = []
        with open(args.file, mode='r', encoding='utf-8') as f:
            contents.append(f.leadlines())
        df = pd.Series(contents, name="content")
    elif filetype == "csv":
        df = pd.read_csv(filepath, encoding='utf-8')
    else:
       if filepath == "該当のファイルがありません":
           print(filepath)
       else:
           print("指定された形式のファイルを入力してください")
       sys.exit()
    corpus_to_vec(df.content, filename)
    end = time.perf_counter
    exetime = int(end-start)
    print("実行時間は{}時間{}分{}秒でした".format(exetime/3600, exetime/60, exetime%60))
    
