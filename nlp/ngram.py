import pandas as pd
from typing import List
from time import time


class NGram():
    '''The N-Gram model is a Markov Chain that approximates the conditional probability of a word, given a specific pretext,
    by "looking" back only the last n-1 pretext words. I.e. a trigram approximates the probability of a word by looking only at
    the previous two words. A 4-gram will look back 3 words and so on...'''
    def __init__(self, n=3, ):
        self.n = n
        self.prob_table = {}
        self.f_ngram = {}
        self.f_seq = {}
        
    def fit(self, texts:List[str]):     # O(MN)
        start = time()
        for text in texts:     # O(N) n of sentences
            l = len(text)
            if l > self.n + 1:
                tokens = text.strip().split(" ")
                for i in range(self.n, len(tokens)):   # O(M) n of words
                    target = (tokens[i],)
                    #target = (target,)
                    ngram = tuple(tokens[i-self.n + 1 : i])
                    seq = ngram + target
                    self.f_ngram[ngram] = 1 if ngram not in self.f_ngram.keys() else\
                                        self.f_ngram[ngram]+1 # O(1)
                    self.f_seq[seq] = 1 if seq not in self.f_seq.keys() else\
                                        self.f_seq[seq]+1 # O(1)
                
        # Compute the probability table
        for seq in self.f_seq.keys():
            ngram = seq[:-1]
            target = seq[-1]
            prob = self.f_seq[seq] / self.f_ngram[ngram]
            
            if ngram in self.prob_table.keys():
                self.prob_table[ngram].append((target, prob))
            else: 
                self.prob_table[ngram] = [(target, prob),] 
        
        end = time()
        print(f"Elapsed time: {end - start}s. N-gram trained on {len(self.f_ngram.keys())} ngrams\n")
    
    def __call__(self, query):
        ngram = query
        if isinstance(ngram, str):
            ngram = ngram.split(' ')[-self.n:]
        if tuple(ngram) not in self.prob_table.keys():
            return query + " -> " + ' '
        tok_pr = self.prob_table[tuple(ngram)]
        argmax = sorted(tok_pr, key=lambda x: x[1], reverse=True)[0]  # O(mlogm) where m #possible targets after ngram
        return query + " -> " + argmax[0]

if __name__ == "__main__":
    data = pd.read_csv('data/bbc_news.csv')
    texts = [line for line in data['description']]
    # deduplicate dataset
    unique_texts = set(texts)

    all_words = [word for text in unique_texts for word in text.split(' ')]
    unique_words = set(all_words)
    n_texts = len(texts)

    print("Number of sentences :", len(texts),". Number of unique sentences :", len(unique_texts))
    print("Number of words :", len(all_words),". Number of unique words :", len(unique_words), "\n")

    trigram = NGram(n=4)
    trigram.fit(unique_texts)

    print(trigram("on the Moon's"))
    # truncation starting from the left 
    print(trigram("the Moon's south"))
    print(trigram("some jibberish words"))
    print(trigram("two words"))