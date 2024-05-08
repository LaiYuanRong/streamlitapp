################################       保险产品推荐预测       ###############################
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression #Logistic回归模型
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score,recall_score,accuracy_score,roc_curve,auc,f1_score,confusion_matrix,roc_auc_score

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df = df.drop(['Region_Code', 'Policy_Sales_Channel'], axis=1)

labelEncoder= LabelEncoder()
df['Gender'] = labelEncoder.fit_transform(df['Gender'])
df['Vehicle_Damage'] = labelEncoder.fit_transform(df['Vehicle_Damage'])

df['Vehicle_Age'] = df['Vehicle_Age'].map({'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2})

# 截断法处理异常值
f_max = df['Annual_Premium'].mean() + 3*df['Annual_Premium'].std()
f_min = df['Annual_Premium'].mean() - 3*df['Annual_Premium'].std()
df.loc[df['Annual_Premium'] > f_max, 'Annual_Premium'] = f_max
df.loc[df['Annual_Premium'] < f_min, 'Annual_Premium'] = f_min

x=df.drop(['Response','id'],axis=1) #contain all  independent variable
y=df['Response']                    #dependent variable

model = ExtraTreesClassifier()
model.fit(x,y)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based clas

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(11).plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()

#根据变量重要性分数删除变量Driving_License、Gender
x=x.drop(['Driving_License','Gender'],axis=1)

xtrain,xtest,ytrain,ytest=train_test_split(x, y, test_size=0.30, random_state=0)

labelEncoder= LabelEncoder()
scaler=StandardScaler()

xtrain=scaler.fit_transform(xtrain)
xtest=scaler.transform(xtest)

model=LogisticRegression()
model=model.fit(xtrain,ytrain)
lr_pred=model.predict(xtest)

lr_probability =model.predict_proba(xtest)[:,1]
fpr, tpr, _ = roc_curve(ytest, lr_probability)

acc_lr=accuracy_score(ytest,lr_pred)
recall_lr=recall_score(ytest,lr_pred)
precision_lr=precision_score(ytest,lr_pred)
f1score_lr=f1_score(ytest,lr_pred)
AUC_LR=auc(fpr,tpr)

print(classification_report(lr_pred,ytest))

pickle.dump(model, open("逻辑回归模型.pkl", "wb"))