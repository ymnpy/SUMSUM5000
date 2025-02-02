import numpy as np
import networkx as nx
import pandas as pd
import nltk
import re
import string
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import shelve
"""extraction based text summarization using TextRank algorithm"""

                
def pdf2text(path):
    with open(path,"rb") as fin:
        pdf_reader=PyPDF2.PdfReader(path)
        
        text=""
        for page_num in range(len(pdf_reader.pages)):
            page=pdf_reader.pages[page_num]
            text+=page.extract_text()
    return text


def do_tokenize(text):
    sentences=nltk.sent_tokenize(text)
    tokenized_sentences=pd.Series(sentences).str.replace("[^a-zA-Z]"," ")
    return tokenized_sentences


def get_embeddings(path):
    with shelve.open("file_cache.db") as cache:
        if "word_embeddings" in cache:
            word_embeddings=cache["word_embeddings"]
        else:
            word_embeddings={}
            with open(path,"r",encoding="utf-8") as fin:
                for w in fin.readlines():
                    try: word_embeddings[w.split()[0]]=np.array(w.split()[1:],dtype="float32")
                    except: pass
                cache["word_embeddings"]=word_embeddings
                
    return word_embeddings


def do_vectorization(sentences,word_embeddings):
    vectors=[]
    for sentence in sentences: #vectorization done after clean and stopwords
        if len(sentence)!=0: 
            vector=sum([word_embeddings.get(word, np.zeros(dim,)) for word in sentence.split()])/(len(sentence.split())+1e-3) #this one is an if/else in list comprehension
        else:
            vector=np.zeros(dim,) #if the word is not found, just 0 vector
        vectors.append(vector)
    return vectors


def rem_ascii(s):
    return "".join([c for c in s if ord(c) < 128 ])


def clean(path,doc):
    """
    Cleaning the text sentences so that punctuation marks, stop words and digits are removed.
    """
    with shelve.open("file_cache.db") as cache:
        if "stopwords" in cache:
            stopwords=cache["stopwords"]
        else:
            stopwords=[]
            with open(path,"r") as fin:
                for line in fin.readlines():
                    stopwords.append(line.rstrip())
                cache["stopwords"]=stopwords
                
    stop=set(stopwords)
    exclude = set(string.punctuation)

    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    processed = re.sub(r"\d+","",punc_free)
    return processed


def get_ranked(sentences,vectors,percentage):
    """
    *make similarity matrix
    *apply textrank algorithm
    *sort sentences as tuples with scores and return it
    """
    sim_matrix=np.zeros([len(sentences), len(sentences)])

    for row in range(len(sentences)):
        for col in range(len(sentences)):
            if row!=col: #higher values means more similarity
                sim_matrix[row][col]=cosine_similarity(vectors[row].reshape(1,dim),
                                                    vectors[col].reshape(1,dim))[0,0]
     
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    
    ranked_sentences = sorted(((scores[i],i) for i,s in enumerate(tokenized_text)), reverse=True)
    arranged_sentences = sorted(ranked_sentences[0:int(len(tokenized_text)*percentage)], key=lambda x:x[1])

    return arranged_sentences

if __name__=="__main__":
    with open("example_text.txt","r",encoding="utf-8") as fin:
        text=fin.read()
    # text=pdf2text(path)
    
    dim=100
    glove_path=r"C:\Users\User\Desktop\VSCODE\12_textsum\glove.6B\glove.6B.100d.txt"
    stop_path=r"C:\Users\User\Desktop\VSCODE\12_textsum\corpora\stopwords\english"
    
    tokenized_text=do_tokenize(text)
    clean_text = [rem_ascii(clean(stop_path,sentence)) for sentence in tokenized_text]

    word_embeddings=get_embeddings(glove_path)
    vectors=do_vectorization(clean_text,word_embeddings)

    final_sentences=get_ranked(clean_text,vectors,0.2)
    
    for s in final_sentences:
        print("* "+tokenized_text[s[1]],end="\n\n") #s[1] gives id of the sentence in whole text


