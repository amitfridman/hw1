import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import training
# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()

# Reading input TSV
#data = pd.read_csv(args.tsv_path, sep="\t")
x,y=training.preprocessing(args.tsv_path)
#####
# TODO - your prediction code here

# Example:
prediction_df = pd.DataFrame(columns=['id', 'revenue'])
prediction_df['id'] = x['id']
model=pickle.load(open('RFregressor.pkl','rb'))
preds=model.predict(x)
for i,pred in enumerate(preds):
    if pred<0:
        preds[i]=0

prediction_df['revenue'] = preds
####

# TODO - How to export prediction results
prediction_df.to_csv("prediction.csv", index=False, header=False)


#  Utility function to calculate RMSLE
def rmsle(y_true, y_pred):
 """
 Calculates Root Mean Squared Logarithmic Error between two input vectors
 :param y_true: 1-d array, ground truth vector
 :param y_pred: 1-d array, prediction vector
 :return: float, RMSLE score between two input vectors
 """
 assert y_true.shape == y_pred.shape, \
     ValueError("Mismatched dimensions between input vectors: {}, {}".format(y_true.shape, y_pred.shape))
 return np.sqrt((1/len(y_true)) * np.sum(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))


#Example - Calculating RMSLE
res = rmsle(y, prediction_df['revenue'])
print("RMSLE is: {:.6f}".format(res))


