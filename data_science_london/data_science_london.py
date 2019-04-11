import pandas as pd


# Getting data
CWD = '/home/jhofman/PycharmProjects/kaggle/data_science_london/'
train = pd.read_csv(CWD+'train.csv',header=None)
test = pd.read_csv(CWD+'test.csv',header=None)
trainLabels = pd.read_csv(CWD+'trainLabels.csv',header=None)

# Most naive way possible
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression().fit(X=train, y=trainLabels)
predictions = lr.predict(test)

pd.DataFrame(
    data=list(zip([x for x in range(1,9001)], predictions.tolist()))
).to_csv('solution_1.csv', index=False, header=['Id', 'Solution'])

# Evaluation
from sklearn.metrics import roc_auc_score
roc_auc_score(trainLabels, lr.predict_proba(train)[:, 1])



