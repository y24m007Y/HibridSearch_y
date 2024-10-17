import numpy as np
import pandas as pd
import MeCab
import unidic
import re
import json
import string
from pathlib import Path
from rank_bm25 import BM25Okapi
import argparse
import os, sys

class Searcher():
    def __init__(self, filename="sample"):
        self.newstitle = filename if filename != "sample" else "kaden-channel"
        filename += ".csv"
        folder = Path("livedoor/csv")
        news = [f for f in folder.glob("*.csv")]
        if filename in [f.name for f in news]:
            filepath = folder / filename
        else:
            filepath = folder / "kaden-channel.csv"
        data = pd.read_csv(filepath, encoding="utf-8")
        self.set_data(data)
        self.pattern = string.punctuation.replace(r'-','')+'。、'
        with open("custom_stopwords_ja.json", 'r') as f:
            jsondata = json.load(f)
        self.stopwords = jsondata['stopwords']
        self.tagger = MeCab.Tagger(f"-Owakati {unidic.DICDIR}")
    
    def set_data(self, data):
        self.title = data.title
        self.url = data.url
        self.contents = data.content
        return
    
    def get_newstitle(self):
        return self.newstitle
    
    def get_countsnews(self):
        return len(self.title)
    
    def simulate(self, query):
        return query
    
    def search(self):
        query = input("検索:")
        scores = self.simulate(query)
        rank = np.argsort(scores)[::-1]
        print("{}:検索結果".format(query))
        for i in rank[:20]:
            if scores[i] > 0:
                print(f"{i}: {self.title[i]} \n url: {self.url[i]} \n スコア:{scores[i]}")
            else:
                break
        if input("検索を続けますか:y or n") == "y":
            self.search()
        else:
            return
    
class bm25(Searcher):
    def __init__(self, filename="sample"):
        super().__init__(filename)
        self.get_model(self.contents)
        
    def preprocess(self, text):
        text = re.sub('\n', '', text)
        text = re.sub('[{}]'.format(self.pattern), '', text)
        sub_words = self.tagger.parse(text).split()
        sub_words = [word for word in sub_words if word not in self.stopwords]
        return sub_words

    def get_model(self, texts):
        texts = texts.apply(lambda x: self.preprocess(x))
        texts = texts.to_list()
        self.bm = BM25Okapi(texts)
        if bool(bm25):
            print("モデルの構築ができました")
            print(self.bm)
        else:
            print("モデルの構築ができませんでした")
            return 
    
    def simulate(self, query):
        tokenized_query = self.preprocess(query)
        scores = self.bm.get_scores(tokenized_query)
        return scores

class laai(Searcher):
    def __init__(self, filename="kaden-channel"):
        from sentence_transformers import SentenceTransformer
        super().__init__(filename)
        self.model = SentenceTransformer("sentence-transformers/LaBSE")
        print("モデルの構築ができました")
        filepath = "corpus_vector/"+filename+".pkl"
        if os.path.exists(filepath):
            self.corpus_vec = pd.read_pickle(filepath)
        else:
            print("ベクトルコーパスがありません")
            sys.exit()
            
    def get_embedding(self, text):
        return self.model.encode(text)
    
    def get_score(self, query):
        score = np.zeros(len(self.corpus_vec))
        for id, vec in zip(range(1,len(self.corpus_vec)), self.corpus_vec):
            score[id] = query @ vec.T / np.sqrt((query@query.T)*(vec@vec.T))
        return score
    
    def simulate(self, query):
        query_vec = self.get_embedding(query)
        scores = self.get_score(query_vec)
        return scores
    
class hibrid(Searcher):
    def __init__(self, filename="sample"):
        super().__init__(filename)
        self.bm = bm25(filename)
        self.labse = laai(filename)
        self.created()

    def created():
        return "Created!"
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bm25", 
                         help="press -b keyword", action="store_true")
    parser.add_argument("-l", "--laai", help="press -l vector", action="store_true")
    parser.add_argument("-hi", "--hibrid", help="press -hi hibrid", action="store_true")
    parser.add_argument("filename",  default="kaden-channel", help="prease set file(default=kaden-channel)",)
    args = parser.parse_args()
    searcher = 0
    file = args.filename
    if args.bm25:
        searcher = bm25(file)
    elif args.laai:
        searcher = laai(file)
    elif args.hibrid:
        searcher = hibrid(file)
    searcher.search()


        
        
        
    