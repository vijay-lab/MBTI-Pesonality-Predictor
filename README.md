# MBTI-Pesonality-Predictor
This project aims toward predicting personality of a person using Machine Learning.
_____________________This documents contains some crucial info about this project.______________________
############################################################################################################
1.In this project cleaning and stemming the data will take more than an hour,so to save time I have exported clean and porter stemmed posts into "svd_corpus.npy" as an numpy array to google drive. file link(https://drive.google.com/open?id=1_HVnGQkZ9fsZVQpnmP1mGuS_x2vbWcRQ).
To use this just import this file using insert button in spyder's variable explorer as then convert it into a list or append directly into your dataframe.

2.MBTI.py contains code for cleaning the data,trainig the model and saving the model onto disk using pickle file.

3.mbti_pred.py contains a function the does prediction using trained and saved pickle files and it's core purpose is to be called by server.py and return a prediction on the given comment.

4.Folder saved models contains trained models and trained countvectorizer.
