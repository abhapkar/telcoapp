# Last amended: 2nd October, 2021
#              11th Oct, 2021
# My folder: /home/ashok/Documents/churnapp
# VM: lubuntu_healthcare
#            D:\data\OneDrive\Documents\streamlit
# Objectives:
#           i)  Develop a churn model using Pipeline
#           ii) Use it in a webapp

# Ref: Github: https://github.com/spierre91/medium_code/tree/master/streamlit_builtin

# 1.0 Call libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA     # optional. OHE shoud then put sparse = False
                                          # PCA does not support sparse data.
                                          # TruncatedSVD does
import pandas as pd
import numpy as np
import pickle
import os,gc


# 2.0 Read data and select just few columns for our model
path = "/home/ashok/Documents/churnapp-sklearn/"
df = pd.read_csv(path+'telco_churn.csv')

# 2.1 Show few rows
df.head(3)


### A. Perform bare-minimum essential processing:

# 3.0 Drop not nneded columns
df = df.drop(labels = ['customerID'], axis =1)

# 3.1
df.shape    # (7043, 20)

# 3.2 Map Churn to 1 or 0
#     Can use df.map() also
df['Churn'] = np.where(df['Churn']=='Yes', 1, 0)   #df['Churn'].apply(target_encode)


# 3.3 Field, 'TotalCharges' is object type:
df.dtypes

# 3.4 There is 'space' value inserted
#     in some rows instead of NaN:
(df['TotalCharges']== " ").sum()     # 11


# 3.5 Replace 'space' with NaN
df.loc[df['TotalCharges']== " " , 'TotalCharges'] = np.nan


# 3.6 Now change datatype of 'TotalCharges'
df['TotalCharges']= df[['TotalCharges']].astype('float64')

#3.6.1
df['TotalCharges'].min()
df['TotalCharges'].max()

df['MonthlyCharges'].min()
df['MonthlyCharges'].max()

df.memory_usage().sum()

np.iinfo('uint8')

df.select_dtypes(include = ['int64']).head(2)

df['SeniorCitizen']=df['SeniorCitizen'].astype('uint8')

df['tenure']=df['tenure'].astype('uint8')
df['Churn']=df['Churn'].astype('uint8')

df['MonthlyCharges']=df['MonthlyCharges'].astype('float16')
df['TotalCharges']=df['TotalCharges'].astype('float16')


df.memory_usage().sum()

df['Partner']=df['Partner'].astype('category')

# 3.7 And check again
df.dtypes


# 3.8 Shuffle data:
df = df.sample(frac= 1.0)

# 3.9 Separate into predictors and target
X = df.drop('Churn', axis=1)
y = df['Churn']


### B. Prepare to create pipes:

# 4.0 List of categorical and numeric columns
cat_cols = list(X.select_dtypes(include = ['object','category']).columns)
num_cols = list(X.select_dtypes(include = ['int64', 'float64','uint8','float']).columns)
len(cat_cols)      # 15
len(num_cols)      # 4
len(X.columns)     # 19


### C. Pipes now:



# 5.0 Pipe to process categorical features:
pipe_cat = Pipeline(
                    steps = [
                              ('si', SimpleImputer( strategy='most_frequent')),
                              ('ohe',   OneHotEncoder())
                            ]
                       )

# 5.1 Pipe to process numerical features:
pipe_num = Pipeline(
                    steps = [
                              ('si', SimpleImputer( strategy='median')),
                               ('scale', StandardScaler())
                            ]
                       )


# 5.2 Column transformer to process both cat and num features:
ct_transformer = ColumnTransformer(
                                    [
                                      ('cat_cols',   pipe_cat,   cat_cols),
                                      ('scaler', pipe_num, num_cols),
                                    ]
                                  )

# 5.3 Just test if ColumnTransformer is working:
abc = ct_transformer.fit_transform(X)
abc.shape      # (7043, 45)
del abc
gc.collect()


### D. Model now:

# 6.1 Final pipeline for transformation and modeling
outer_pipe = Pipeline(
                        [
                            ('ct', ct_transformer),        # Column transformer object
                            #('pca', PCA(n_components = 0.95)),
                            #('abc',SelectKBest(k=10)),
                            ('rf', RandomForestClassifier()) # Estimator
                        ]
                     )

# 6.2 Train the outer_pipe:
outer_pipe.fit(X,y)

# 6.3 Save the model to disk:
pickle.dump(outer_pipe, open(path+'churn_pipe.pkl', 'wb'))

########################## Done ####################
