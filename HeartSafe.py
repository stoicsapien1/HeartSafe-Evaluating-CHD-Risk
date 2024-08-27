#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV



# In[3]:


import warnings
warnings.filterwarnings("ignore")


# In[4]:


df=pd.read_csv(r"C:\Users\lucius seneca\CampusX\Project\Cardiovascular_Risk_Prediction\Dataset\data_cardiovascular_risk.csv",index_col="id")


# In[5]:


df


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


# Shape of the Dataset
df.shape


# In[9]:


#Dataset info
df.info()


# In[10]:


#Duplicate data
df.duplicated()


# In[11]:


# Missing values in the dataset
df.isnull().sum()


# In[12]:


#Missing vlaues percentage
round(df.isnull().sum()/len(df) * 100,2)


# In[13]:


#Visualizing missing value
import missingno as msno
msno.bar(df,color="red",sort="ascending")


# In[14]:


msno.dendrogram(df)


# In[15]:


sns.heatmap(df.isnull(),cmap="coolwarm")


# In[16]:


#Dataset columns
df.columns


# In[17]:


#Summary Statistics
df.describe().T


# In[18]:


#Numeric and Categorical Features
numeric_features=[]
categorical_features=[]

for col in df.columns:
    if df[col].nunique() > 10:
        numeric_features.append(col)
    else:
        categorical_features.append(col)
    
print(f"numeric features: {numeric_features}")

print(f"categorical features:{categorical_features}")


# In[19]:


#Univariate Analysis
plt.figure(figsize=(15,5))
for i,col in enumerate(numeric_features):
    plt.subplot(2,4,i+1)
    sns.histplot(df[col],kde=True)
    plt.xlabel(col)
    plt.tight_layout()


# ## Observation: We can say that glucose has many outliers because it heavely right skewed

# In[20]:


#Outlier Analysis
import plotly.express as px
for col in numeric_features:
    fig=px.box(df,y=col,title=f"Box Plot of {col}")
    fig.show()


# In[21]:


#Univariate Analysis of categorical features
for i,col in enumerate(categorical_features):
    plt.subplot(2,4,i+1)
    sns.countplot(x=df[col])
    plt.xlabel(col)
    plt.tight_layout()


# ## Data Cleaning

# In[22]:


df.duplicated().sum()


# In[23]:


# Missing value

df.isnull().sum()


# In[24]:


# Missing value percentage
round(df.isnull().sum()/len(df)*100,2)


# In[25]:


# features which has less than 5% missing value
nan_col=["education","cigsPerDay","BPMeds","totChol","BMI","heartRate"]
#dropping null values
df.dropna(subset=nan_col,inplace=True)


# In[26]:


df.isnull().sum()


# In[27]:


sns.histplot(df["glucose"])


# In[28]:


df.isnull().sum()


# In[29]:


df["glucose"]=df["glucose"].fillna(value=df["glucose"].median())


# In[30]:


df.isna().sum()


# In[31]:


#Treating Outlier
plt.figure(figsize=(15,5))

sns.boxplot(data=df[numeric_features])
plt.show()


# In[32]:


# capping outlier
def clip_outlier(df):
    for col in df[numeric_features]:
        q1=df[col].quantile(0.25)
        q3=df[col].quantile(0.75)
        iqr=q3-q1
        lower_bound=q1-1.5*iqr
        upper_bound=q3+1.5*iqr
        df[col]=df[col].clip(lower_bound,upper_bound)
    return df


# In[33]:


df=clip_outlier(df)


# In[34]:


#Checking boxplot
plt.figure(figsize=(15,5))

sns.boxplot(data=df[numeric_features])


# ## Feature Engineering

# In[35]:


#Label Encoding

df["sex"]=df["sex"].map({"M":1,"F":0})
df["is_smoking"]=df["is_smoking"].map({"YES":1,"NO":0})


# In[36]:


df.dtypes


# In[37]:


df.head()


# In[38]:


df["education"].value_counts()


# In[39]:


#Onehot Encoding the "education" feature

education_onehot=pd.get_dummies(df["education"],prefix="education",drop_first=True)
df.drop("education",axis=1,inplace=True)

df=pd.concat([df,education_onehot],axis=1)
df.head(3)


# ## Feature Manipulation
# 

# In[40]:


#Plotting correlation coefficient heatmap
plt.figure(figsize=(15,5))
sns.heatmap(df.corr(),annot=True,cmap="coolwarm")


# In[41]:


#adding new column PulsePresure
df["pulse_pressure"]=df["sysBP"]-df["diaBP"]
#dropping the sysBp and diaBP columns
df.drop(columns=["sysBP","diaBP"],inplace=True)


# In[42]:


# dropping is_smoking dute to high multicollinearity 
df.drop("is_smoking",axis=1,inplace=True)


# In[43]:


df.head(2)


# In[44]:


#plotting correlation heatmap
plt.figure(figsize=(15,5))
sns.heatmap(df.corr(),annot=True,cmap="coolwarm")


# In[45]:


X=df.drop(columns="TenYearCHD")


# In[46]:


y=df["TenYearCHD"]


# In[47]:


X


# In[48]:


#Handling Imbalance Target variable
df["TenYearCHD"].value_counts().plot(kind="bar")


# In[49]:


#SMOTE implementation
from collections import Counter
from imblearn.over_sampling import SMOTE

print(f'Before Handling Imbalanced Class {Counter(y)}')
smote=SMOTE(random_state=42)

X,y=smote.fit_resample(X,y)
print(f'After Handling Imbalanced Class {Counter(y)}')


# ## Model Building

# In[50]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=33)

print(X_train.shape)
print(X_test.shape)


# In[51]:


# Standard Scaler

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)


# In[52]:


X_train


# Model Training

# In[53]:


model_result=[]

def predict(ml_model,model_name):

    model=ml_model.fit(X_train,y_train)

    #prediction

    y_pred=model.predict(X_test)

    '''Performance Metrics'''
    test_accuracy=accuracy_score(y_test,y_pred)
    print(f"test accuracy:{test_accuracy}")

    test_precision=precision_score(y_test,y_pred)
    print(f"precision score:{test_precision}")

    test_recall=recall_score(y_test,y_pred)
    print(f"recall score:{test_recall}")

    test_f1=f1_score(y_test,y_pred)
    print(f"test f1 score:{test_f1}")


# In[54]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


# In[55]:


from sklearn.linear_model import LogisticRegression


# In[56]:


predict(LogisticRegression(),"Logistic Regression")


# In[57]:


from sklearn.tree import DecisionTreeClassifier


# In[58]:


predict(DecisionTreeClassifier(),"Decision Tree Classifier")


# In[59]:


from sklearn.ensemble import RandomForestClassifier
predict(RandomForestClassifier(),"Random Forest classifier")


# In[ ]:




