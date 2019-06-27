# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:39:47 2019

@author: TAPAN
"""
import pandas as pd
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import pickle

def get_prediction(input_comment):
    
    with open("saved models/cv.pkl",'rb') as f:
        cv=pickle.load(f)
    with open("saved models/ie.pkl",'rb') as f:
        IEB=pickle.load(f)
    with open("saved models/ns.pkl",'rb') as f:
        NSB=pickle.load(f)
    with open("saved models/tf.pkl",'rb') as f:
        TFB=pickle.load(f)
    with open("saved models/jp.pkl",'rb') as f:
        JPB=pickle.load(f)
    data = re.sub('[^a-zA-Z]', ' ', input_comment).lower().split()
    check=[]
    ps = PorterStemmer()
    review = [word for word in data if not word in set(stopwords.words('english'))]
    review = [ps.stem(word) for word in review]
    review = ' '.join(review)
    check.append(review)
            
            
    data=' '.join(check)
    data=pd.Series(data)
    data1=cv.transform(data).toarray()
    result=[]
    IET=IEB.predict(data1)
    if IET==0:
        result.append('I')
    else:
        result.append('E')  
        
    NST=NSB.predict(data1)
    if NST==0:
        result.append('N')
    else:
        result.append('S')  
       
    TFT=TFB.predict(data1)
    if TFT==0:
        result.append('T')
    else:
        result.append('F')  
        
    JPT=JPB.predict(data1)
    if JPT==0:
        result.append('J')
    else:
        result.append('P')  
    #personality
    pesonality=str(''.join(result))
    return(pesonality) 


#input_comment=input("Enter Your Comments to be Profilled::\n")
#pesonality=get_prediction(input_comment)