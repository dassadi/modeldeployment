import pandas as pd
data={'experience':['','','five','two','seven','three','ten','eleven'],
     'test_score':[8,8,6,10,9,'',7,7], 'interview_score':[9,6,7,10,6,10,7,8],'salary':[50000,45000,60000,65000,70000,62000,72000,80000]}


df=pd.DataFrame(data )
print(df)

df['experience']=df['experience'].map({'five':5,'two':2,'seven':7,'three':3,'ten':10,'eleven':11})
df['experience']=df['experience'].fillna(0)
df['test_score']=pd.to_numeric(df['test_score'])
df['test_score'].fillna(df['test_score'].mean(),inplace=True)
X=df.drop(['salary'], axis=1)
y=df.salary
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train,y_test= train_test_split(X,y, test_size=0.2, random_state=42)
lr=LinearRegression()
lr.fit(X_train, y_train)
y_predicted=lr.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, adjusted_rand_score
import pickle
pickle.dump(lr, open('model.pkl','wb'))