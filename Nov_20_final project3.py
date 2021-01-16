#!/usr/bin/env python
# coding: utf-8

# In[4]:


import csv
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from collections import defaultdict, Counter
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D, BatchNormalization
#from tensorflow.keras.layers.normalization import BatchNormalization

import import_ipynb
import stanza_nlp as nlp 
    
# turns .tsv file into list of lists
def tsv2mat(fname) :
    with open(fname) as f:
        wss = csv.reader(f, delimiter='\t')
        return list(wss)

class Data :
  '''
  builds dataset from dependency edges in .tsv file associating
  <from,link,to> edges and sentences in which they occur;
  links are of the form POS_deprelPOS with POS and deprel
  tags concatenated
  '''
  def __init__(self,fname='texts/english') :
        doc_len = 0
        wss = tsv2mat("out/"+fname+".tsv")
        self.sents=tsv2mat("out/"+fname+"_sents.tsv")
        occs=defaultdict(set)
        sids=set()
        for f,r,t,id in wss:
            id=int(id)
            occs[(f,r,t)].add(id)
            sids.add(id)
            doc_len += 1
        self.occs=occs # dict where edges occur
        self.doc_len = doc_len
        X,Y=list(zip(*list(occs.items())))
        X = np.array(X)
        y0 = np.array(sorted(map(lambda x:[x],sids)))
    
    # make OneHot encoders for X and y
        enc_X = OneHotEncoder(handle_unknown='ignore')
        enc_y = OneHotEncoder(handle_unknown='ignore')
        enc_X.fit(X)
        enc_y.fit(y0)
        hot_X = enc_X.transform(X).toarray()
        self.enc_X = enc_X
        self.enc_y = enc_y
        self.X=X
    # encode y as logical_or of sentence encodings it occurs in
        ms=[]
        for ys in Y :
            m = np.array([[0]],dtype=np.float32)
            for v in ys :
                m0=enc_y.transform(np.array([[v]])).toarray()
                m = np.logical_or(m,m0)
                m=np.array(np.logical_or(m,m0),dtype=np.float32)
            ms.append(m[0])
        hot_y=np.array(ms)

        self.hot_X=hot_X
        self.hot_y =hot_y

        print('\nFINAL DTATA SHAPES','X',hot_X.shape,'y',hot_y.shape,'\n')

class Query(Data) :
  '''
  builds <from,link,to> dependency links form a given
  text query and matches it against data to retrive
  sentences in which most of those edges occur
  '''
  def __init__(self,fname='texts/english'):
        super().__init__(fname=fname)
        self.nlp_engine=nlp.NLP()

  def ask(self,text=None):
        if not text: text = input("Query:")
        else: print("Query:",text)

        self.nlp_engine.from_text(text)
        sids=[]
        for f,r,t,_ in self.nlp_engine.facts() :
            sids.extend(self.occs.get((f,r,t),[]))
        self.show_answers(sids)

  def show_answers(self, sids, k=3):
        c = Counter(sids)
        print('\nHIT COUNTS:',c,"\n")
        best = c.most_common(k)
        for sid, _ in best:
            id, sent = self.sents[sid]
            print(id, ':', sent)
        print("")

class Inferencer(Query) :
  '''
  loads model trained on associating dependency
  edges to sentences in which they occur
  '''
  def __init__(self,fname='texts/english'):
      super().__init__(fname=fname)
      self.model = load_model(fname+"_model")

  def query(self,text=None):
      if not text: text = input("Query:")
      else: print("Query:", text)
      self.nlp_engine.from_text(text)
      X=[]
      for f, r, t, _ in self.nlp_engine.facts():
          X.append([f,r,t])
      X = np.array(X)
      hot_X = self.enc_X.transform(X).toarray()
      y=self.model.predict(hot_X)
      m=self.enc_y.inverse_transform(y)
      sids=m.flatten().tolist()
      self.show_answers(sids)

class Trainer(Data) :
  '''
  neural network trainer and model builder
  '''
  def __init__(self,fname='texts/english'):
      super().__init__(fname=fname)
      '''
      model = keras.Sequential()
      model.add(layers.Dense(128, input_dim=self.hot_X.shape[1], activation='relu'))
      model.add(layers.Dense(self.hot_y.shape[1], activation='sigmoid'))
      model.summary()
      model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
      history = model.fit(self.hot_X, self.hot_y, epochs=100, batch_size=16) 
      model.save(fname+"_model")
      ''' 
      
      model = keras.Sequential()
      model.add(Embedding(input_dim = self.hot_X.shape[1], output_dim = 256, 
                         input_length = self.hot_X.shape[1]))   
     # add dropout to prevent overfitting
      model.add(SpatialDropout1D(0.3))
     #add first LSTM
     # model.add(BatchNormalization())
      model.add(LSTM(256, dropout = 0.25, recurrent_dropout = 0.3, return_sequences=True))
     # add second LSTM to improve accuracy
      
      model.add(LSTM(256))
      #model.add(BatchNormalization())
      model.add(Dense(256, activation = 'relu'))
      model.add(Dropout(0.25))
      model.add(Dense(self.hot_y.shape[1], activation = 'softmax'))
      model.summary()
     # improve its accuracy 
      opt = keras.optimizers.Adam(learning_rate=0.05)  
        
      model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['acc'])
      history = model.fit(self.hot_X, self.hot_y, epochs=8, batch_size=16)
      model.save(fname+"_model")   
        
    # visualize and inform about accuracy and loss
      plot_graphs(fname + "_loss", history, 'loss')
      plot_graphs(fname + "_acc", history, 'acc')

      loss, accuracy = model.evaluate(self.hot_X, self.hot_y)
      print('Accuracy:', round(100 * accuracy, 2), ', % Loss:', round(100 * loss, 2), '%')


# VISUALS

import matplotlib.pyplot as plt

def plot_graphs(fname,history, metric):
    plt.plot(history.history[metric])
  #plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.savefig("pics/"+fname + '.pdf',format="pdf",bbox_inches='tight')
  #plt.show()
    plt.close()


### TESTS ###

def qtest() :
    q=Query()
    q.ask(text="When will vaccine available?")
    q.ask(text="Who is producing vaccine?")
    q.ask(text="How long does it take to bring a vaccine to market?")
    q.ask(text="Which states plan to independently review the data for any vaccine?")
    q.ask(text="who is director of the National Institute of Allergy and Infectious Diseases?")
    q.ask(text="Which company has signed a nearly 2 billion contract with the U.S. government?")
    q.ask(text="What primary goal Pfizer has met?")
    q.ask(text="How effective is BNT162b2?")
    q.ask(text="What is BNT162b2?")
    q.ask(text="What is CDC director Robert Redfield's attitude about developing vaccine?")

def dtest() :
    d=Data()
    print("X",d.hot_X.shape)
    print(d.hot_X)
    print("y",d.hot_y.shape)
    print(d.hot_y)

def dtests():
    dtest('out/texts/english.tsv')
  #dtest('out/texts/const.tsv')
  #dtest('out/texts/spanish.tsv')
    dtest('out/texts/chinese.tsv')
  #dtest('out/texts/russian.tsv')

def ntest() :
    t=Trainer()
    i=Inferencer()
    print("\n\n")
    print("Here's the latest on Covid-19 vaccines:\n")
    i.ask("When will vaccine available?")
    i.ask(text="Who is producing vaccine?")
    i.ask(text="How long does it take to bring a vaccine to market?")
    i.ask(text="Which states plan to independently review the data for any vaccine?")
    i.ask(text="who is director of the National Institute of Allergy and Infectious Diseases?")
    i.ask(text="Which company has signed a nearly 2 billion contract with the U.S. government?")
    i.ask(text="What primary goal Pfizer has met?")
    i.ask(text="How effective is BNT162b2?")
    i.ask(text="What is BNT162b2?")
    i.ask(text="What is CDC director Robert Redfield's attitude about developing vaccine?")
    print("\n")
    print("Here's the latest on Covid-19 vaccines:\n")
    i.query("When will vaccine available?")
    i.query(text="Who is producing vaccine?")
    i.query(text="How long does it take to bring a vaccine to market?")
    i.query(text="Which states plan to independently review the data for any vaccine?")
    i.query(text="who is director of the National Institute of Allergy and Infectious Diseases?")
    i.query(text="Which company has signed a nearly 2 billion contract with the U.S. government?")
    i.query(text="What primary goal Pfizer has met?")
    i.query(text="How effective is BNT162b2?")
    i.query(text="What is BNT162b2?")
    i.query(text="What is CDC director Robert Redfield's attitude about developing vaccine?")

if __name__=="__main__" :
  ntest()


# In[ ]:




