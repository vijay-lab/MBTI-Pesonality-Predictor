# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:38:12 2019

@author: TAPAN
"""

import pandas as pd
import numpy as np
import re

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier


from nltk.corpus import stopwords
stop = stopwords.words('english')

from nltk.stem.porter import PorterStemmer
mbti_df1= pd.read_csv('mbti_1.csv')


mbti_df= pd.read_csv('mbti_1.csv')

# Labels that need to be removed from posts
lbl_rmv=list(mbti_df['type'].unique())
lbl_rmv = [item.lower() for item in lbl_rmv]


#Data Preprocessing begins , removing url ,symbols,hashtags and emojis from posts
for i in range(0,8675) :  
    mbti_df['posts'][i] = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', mbti_df['posts'][i])
    mbti_df['posts'][i] = re.sub("[^a-zA-Z]", " ", mbti_df['posts'][i])
    mbti_df['posts'][i] = re.sub(' +', ' ', mbti_df['posts'][i]).lower()
    for j in range(0,16):
        mbti_df['posts'][i]=re.sub(lbl_rmv[j], ' ', mbti_df['posts'][i])
        
mbti_df['posts'] = mbti_df['posts'].str.strip()

#corpus creation and stopwords and porterstemming 
def pre_process(post):
    posts = re.sub('\s+', ' ', post)
    posts = posts.lower()
    posts = posts.split()
    posts = [word for word in posts if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    posts = [ps.stem(word) for word in posts]
    posts = ' '.join(posts)
    return posts
    
corpus = mbti_df["posts"].apply(pre_process)

#we already saved our corpus in tha dataset
# mbti_df['posts']=list(svd_corpus) 

# converting the personality types to 8 respective binary identifiers(I-E,N-S,T-F,J-P)
map1 = {"I": 0, "E": 1}
map2 = {"N": 0, "S": 1}
map3 = {"T": 0, "F": 1}
map4 = {"J": 0, "P": 1}
mbti_df['I-E'] = mbti_df['type'].astype(str).str[0]
mbti_df['I-E'] = mbti_df['I-E'].map(map1)
mbti_df['N-S'] = mbti_df['type'].astype(str).str[1]
mbti_df['N-S'] = mbti_df['N-S'].map(map2)
mbti_df['T-F'] = mbti_df['type'].astype(str).str[2]
mbti_df['T-F'] = mbti_df['T-F'].map(map3)
mbti_df['J-P'] = mbti_df['type'].astype(str).str[3]
mbti_df['J-P'] = mbti_df['J-P'].map(map4)



#bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
features = cv.fit_transform(mbti_df['posts']).toarray()
IE= mbti_df.iloc[:, 2].values
NS= mbti_df.iloc[:, 3].values
TF=mbti_df.iloc[:, 4].values
JP=mbti_df.iloc[:, 5].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, IE_train, IE_test, NS_train, NS_test, TF_train, TF_test, JP_train, JP_test = train_test_split(features, IE,NS,TF,JP, test_size = 0.20, random_state = 0)
###########################################################################################
## Stoic gradient descent ##########################################################################################
sgdd = SGDClassifier(max_iter=5, tol=None)
sgdd.fit(features_train,IE_train)

acc_sgdd_train = round(sgdd.score(features_train,IE_train) * 100, 4)
acc_sgdd_test = round(sgdd.score(features_test,IE_test) * 100, 4)
print('SGD score train',acc_sgdd_train, "%")
print('SGD score test',acc_sgdd_test, "%")

## RAndom Forest ###########################################################################
random_forestt = RandomForestClassifier(n_estimators=20)
random_forestt.fit(features_train,IE_train)

acc_rff_train = round(random_forestt.score(features_train,IE_train) * 100, 4)
acc_rff_test = round(random_forestt.score(features_test,IE_test) * 100, 4)
print('RF score train',acc_rff_train, "%")
print('RF score test',acc_rff_test, "%")

# Logistic Regression ########################################################################
logregg = LogisticRegression()
logregg.fit(features_train,IE_train)

acc_logg = round(logregg.score(features_train,IE_train) * 100, 2)
print("Logisitic Regression Prediction Accuracy",round(acc_logg,2,), "%")

acc_logg = round(logregg.score(features_test,IE_test) * 100, 2)
print("Logisitic Regression Prediction Accuracy on test data",round(acc_logg,2,), "%")

# KNN ############################################################################################
knnn = KNeighborsClassifier(n_neighbors = 3)
knnn.fit(features_train,IE_train)

acc_knnn = round(knnn.score(features_train,IE_train) * 100, 2)
print("Knn neighbor prediction value",round(acc_knnn,2,), "%")

acc_knnn = round(knnn.score(features_test,IE_test) * 100, 2)
print("Knn neighbor prediction value on test data",round(acc_knnn,2,), "%")


## Support vector machine ########################################################################
from sklearn.svm import SVC
svc=SVC()

svc.fit(features_train,IE_train)

acc_svc_train = round(svc.score(features_train,IE_train) * 100, 4)
acc_svc_test = round(svc.score(features_test,IE_test) * 100, 4)
print('SVC score train',acc_svc_train, "%")
print('SVC score test',acc_svc_test, "%")

## Linear Support vector machine ##################################################################

from sklearn.svm import LinearSVC
lsvc=LinearSVC()
lsvc.fit(features_train,IE_train)
acc_lsvc_train = round(lsvc.score(features_train,IE_train) * 100, 4)
acc_lsvc_test = round(lsvc.score(features_test,IE_test) * 100, 4)
print('LSVC score train',acc_lsvc_train, "%")
print('LSVC score test',acc_lsvc_test, "%")

###############################################################################################3

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnb.fit(features_train,IE_train)
gnb_train=gnb.score(features_train,IE_train)*100
gnb_test=gnb.score(features_test,IE_test)*100
print('GNB score train',gnb_train, "%")
print('GNB score test',gnb_test, "%")

mnb.fit(features_train,IE_train)
mnb_train=mnb.score(features_train,IE_train)*100
mnb_test=mnb.score(features_test,IE_test)*100
print('MNB score train',mnb_train, "%")
print('MNB score test',mnb_test, "%")

bnb.fit(features_train,IE_train)
bnb_train=bnb.score(features_train,IE_train)*100
bnb_test=bnb.score(features_test,IE_test)*100
print('BNB score train',bnb_train, "%")
print('BNB score test',bnb_test, "%")
###XGBoost
############################################################################################
#training the model
from xgboost import XGBClassifier

# fit model on training data
IEB = XGBClassifier()
IEB.fit(features_train, IE_train)
ieb_train=IEB.score(features_train,IE_train)
ieb_test=IEB.score(features_test,IE_test)

NSB = XGBClassifier()
NSB.fit(features_train, NS_train)
nsb_train=NSB.score(features_train,NS_train)
nsb_test=NSB.score(features_test,NS_test)


TFB = XGBClassifier()
TFB.fit(features_train, TF_train)
tfb_train=TFB.score(features_train,TF_train)
tfb_test=TFB.score(features_test,TF_test)

JPB = XGBClassifier()
JPB.fit(features_train, JP_train)
jpb_train=JPB.score(features_train,JP_train)
jpb_test=JPB.score(features_test,JP_test)
####################################################################################################
#saving the models  in pickle file
import pickle
cv_obj=open('cv.pkl','wb')
pickle.dump(cv,cv_obj)
cv_obj.close()

ie_obj=open('ie.pkl','wb')
pickle.dump(IEB,ie_obj)
ie_obj.close()

ns_obj=open('ns.pkl','wb')
pickle.dump(NSB,ns_obj)
ns_obj.close()

tf_obj=open('tf.pkl','wb')
pickle.dump(TFB,tf_obj)
tf_obj.close()

jp_obj=open('jp.pkl','wb')
pickle.dump(JPB,jp_obj)
jp_obj.close()


#prediction begins for the random comment

#frist loading the models objects 
with open("cv.pkl",'rb') as f:
    cv=pickle.load(f)
with open("ie.pkl",'rb') as f:
    IEB=pickle.load(f)
with open("ns.pkl",'rb') as f:
    NSB=pickle.load(f)
with open("tf.pkl",'rb') as f:
    TFB=pickle.load(f)
with open("jp.pkl",'rb') as f:
    JPB=pickle.load(f)
    


def get_prediction(input_comment):
    
    with open("cv.pkl",'rb') as f:
        cv=pickle.load(f)
    with open("ie.pkl",'rb') as f:
        IEB=pickle.load(f)
    with open("ns.pkl",'rb') as f:
        NSB=pickle.load(f)
    with open("tf.pkl",'rb') as f:
        TFB=pickle.load(f)
    with open("jp.pkl",'rb') as f:
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

    
#cheching the score on testing data
IEB.score(features_test,IE_test)
NSB.score(features_test,NS_test)
TFB.score(features_test,TF_test)
JPB.score(features_test,JP_test)
