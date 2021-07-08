# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 05:55:13 2019

@author: minh
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import gc
from datetime import timedelta, datetime
import seaborn as sns
import matplotlib.pyplot as plt
import json
import re
from numpy import random
import nltk
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

os.chdir('E:\SAFETY\SSE')
sns.set_style('whitegrid')
sns.set_context('talk')
sns.set(style="whitegrid")
dash_folder = 'dash_app'
stop_words = set(stopwords.words('english'))
#%% Settings
server = 'ADC-MINH-SEA\SQLEXPRESS'
db = 'AQD_Anonymized_20190429'
driver = 'ODBC Driver 17 for SQL Server'  #remember to install on new computer
eng = create_engine('mssql+pyodbc://{}/{}?driver={}?trusted_connection=yes'
    .format(server, db, driver))

def get_upper_IDs(s):
    p = dict_child_parent.get(s, 'nan')
    gp = dict_child_parent.get(p, 'nan')
    return [x for x in [p, gp] if x != 'nan']

def flag_upper(r):
    child_IDs = r[r == 1].index.tolist()
    for c in child_IDs:
        for p in get_upper_IDs(c):
            r[p] = 1
    return r

def full_desc(kid):
    p = dict_child_parent[kid]
    if p == 'nan':
        return dict_ID_Text[kid].replace(',', '')
    return (full_desc(p) + ' > ' + dict_ID_Text[kid]).replace(',', '')

#occurrence
sql = '''select a.Occurrence_No, a.Occurrence_Description, a.Occurrence_Title, 
    a.Aircraft_ID, b.Occurrence_Type_Code
    from AQD_Anonymized_20190429.AQD.oc_Occurrence a
    inner join 
    (select * from AQD_Anonymized_20190429.AQD.oc_Occurrence_Type_Classificat
    where occurrence_type_code in ('SQCHFIR', 'SQAFIR','SQAGIR','SQFOVR','SQGAIR','SQHZR','SQMDR','SQSIF','SQTCIF')) b
    on a.Occurrence_No = b.Occurrence_No
    '''
df_occurrence = pd.read_sql_query(sql, eng)

def replace_special(s):
    return s.replace('"', '\'').replace('\r', ' ').replace('\n', ' ') if s else s

for col in ['Occurrence_Description', 'Occurrence_Title']:
    df_occurrence[col] = df_occurrence[col].apply(replace_special)

# fleet
sql = '''select a.Aircraft_ID, a.Aircraft_Model_ID, b.Friendly_Name as 'Fleet Variant'
from AQD_Anonymized_20190429.AQD.gn_aircraft a
inner join AQD_Anonymized_20190429.AQD.gn_AircraftModel b
on a.Aircraft_Model_ID = b.Aircraft_Model_ID'''
df_fleet = pd.read_sql_query(sql, eng)
df_fleet.drop('Aircraft_Model_ID', axis=1, inplace=True)

df_occurrence = df_occurrence.merge(df_fleet, how='left', on='Aircraft_ID')

#event descriptor
sql = '''select *
from AQD_Anonymized_20190429.AQD.oc_caa_Event_Descriptors c'''
df_event_descriptors = pd.read_sql_query(sql, eng)
df_event_descriptors['Item_ID'] = df_event_descriptors['Item_ID']\
    .apply(str)
df_event_descriptors['Parent_ID'] = df_event_descriptors['Parent_ID']\
    .apply(lambda f: str(int(f)) if not pd.isnull(f) else 'nan')

dict_ID_Text = dict(zip(df_event_descriptors.Item_ID\
    , df_event_descriptors.Item_Text))
dict_child_parent = dict(zip(df_event_descriptors.Item_ID\
    , df_event_descriptors.Parent_ID))
dict_ID_level = dict(zip(df_event_descriptors.Item_ID\
    , df_event_descriptors.Item_Level))
causalfactor_IDs = set(df_event_descriptors[df_event_descriptors.Cause == -1].Item_ID)
noncf_IDs = set(df_event_descriptors[df_event_descriptors.Cause == 0].Item_ID)

category = (['Cabin', 'Ground', 'Maintenance', 'Operational', 'Technical'
    , 'Security', 'Occupational'])
category_subject_IDs = (df_event_descriptors[df_event_descriptors
    .Item_Text.isin(category) & (df_event_descriptors.Item_Level == 'D')]
    .Descriptor_Subject_ID.tolist())
category_IDs = set(df_event_descriptors[df_event_descriptors
    .Descriptor_Subject_ID.isin(category_subject_IDs)]
    .Item_ID.tolist())
    
#occurrence_event
sql = '''select *
from AQD_Anonymized_20190429.AQD.oc_Occurrence_Events b
where occurrence_no in 
(select occurrence_no from AQD_Anonymized_20190429.AQD.oc_Occurrence_Type_Classificat 
where  occurrence_type_code in ('SQCHFIR', 'SQAFIR','SQAGIR','SQFOVR','SQGAIR','SQHZR','SQMDR','SQSIF','SQTCIF'))'''
df_occurrence_events = pd.read_sql_query(sql, eng)
df_occurrence_events['Item_ID'] = df_occurrence_events['Item_ID'].apply(str)
df_occurrence_events['Value'] = 1
df_occurrence_events_pivot = df_occurrence_events.pivot_table(index='Occurrence_No'\
    , columns='Item_ID', values='Value')
df_occurrence_events_pivot = df_occurrence_events_pivot.fillna(0)
df_occurrence_events_pivot.reset_index(inplace=True)

# find other upper IDs from current IDs but not flagged yet
all_IDs = set(df_event_descriptors.Item_ID.tolist())
cur_IDs = set(df_occurrence_events_pivot.columns.tolist())
missing_upp_IDs = list(set([x for s in cur_IDs for x in get_upper_IDs(s) if x not in cur_IDs]))

# flag those IDs
df_others = pd.DataFrame(np.zeros((df_occurrence_events_pivot.shape[0], len(missing_upp_IDs)))\
    , columns=missing_upp_IDs)
df_occurrence_events_pivot = pd.concat([df_occurrence_events_pivot, df_others]\
    , axis=1, ignore_index=False)
df_occurrence_events_pivot = df_occurrence_events_pivot.apply(lambda r: 
    flag_upper(r), axis=1)

# =============================================================================
# df_occurrence_events_pivot.set_index('Occurrence_No', inplace=True)
# 
# a = df_occurrence_events_pivot.stack()
# a = a[a > 0].reset_index()
# a.columns = ['Occurrence_No', 'Item_ID', 'Value']
# df_occurrence_events_upflag = a
# 
# =============================================================================
#%% Process data and label
# =============================================================================
# X = df_occurrence[['Occurrence_No', 'Occurrence_Title', 'Occurrence_Description', 
#     'Occurrence_Type_Code']]
# X['Occurrence_Description'] = X['Occurrence_Description'].apply(lambda s:
#     re.sub(r'[^a-zA-Z0-9-_/]+', ' ', s.lower()) if s is not None else '')
# X['Occurrence_Description'] = (X[['Occurrence_Description', 'Occurrence_Type_Code']]
#     .apply(lambda x: ' '.join([w for w in x['Occurrence_Description'].split() 
#     if (len(w)>1 or w == '.')]), axis=1))
# =============================================================================

# prepare consistent data
X = df_occurrence[['Occurrence_No', 'Occurrence_Title', 'Occurrence_Description', 
    'Occurrence_Type_Code']]
X['Occurrence_Description'] = X['Occurrence_Description'].apply(lambda s:
    re.sub(r'[^,.a-zA-Z0-9-_/]+', ' ', s) if s is not None else '')
X['Occurrence_Description'] = (X[['Occurrence_Description', 'Occurrence_Type_Code']]
    .apply(lambda x: ' '.join([w for w in x['Occurrence_Description'].split() 
    if (len(w)>1 or w == '.')]), axis=1))
    
X['Occurrence_Title'] = X['Occurrence_Title'].apply(lambda s:
    re.sub(r'[^a-zA-Z0-9-_/]+', ' ', s) if s is not None else '')
X['Occurrence_Title'] = (X['Occurrence_Title'].apply(lambda x: 
    ' '.join([w for w in x.split() if ((len(w)>1 and ('_' not in w)) or w == '.')])
    .lower()))
#%%
y = df_occurrence_events_pivot.copy()

cat_lv1 = {k:full_desc(k) for k,v in dict_ID_Text.items() 
    if (dict_ID_level[k] == 'D') and (k in category_IDs)}
tagged_cat = set(cat_lv1.keys()).intersection(set(y.columns))
y_lv1 = y[['Occurrence_No'] + list(tagged_cat)]
z_lv1 = y_lv1[list(tagged_cat)].sum(axis=1)
y_lv1 = y_lv1[z_lv1 > 0].reset_index(drop=True)

cat_lv2 = {k:full_desc(k) for k,v in dict_ID_Text.items() if (dict_ID_level[k] == 'C') 
    and (k in category_IDs)}
tagged_cat = set(cat_lv2.keys()).intersection(set(y.columns))
y_lv2 = y[['Occurrence_No'] + list(tagged_cat)]
z_lv2 = y_lv2[list(tagged_cat)].sum(axis=1)
y_lv2 = y_lv2[z_lv2 > 0].reset_index(drop=True)

cat_lv3 = {k:full_desc(k) for k,v in dict_ID_Text.items() if (dict_ID_level[k] == 'S') 
    and (k in category_IDs)}
tagged_cat = set(cat_lv3.keys()).intersection(set(y.columns))
y_lv3 = y[['Occurrence_No'] + list(tagged_cat)]
z_lv3 = y_lv3[list(tagged_cat)].sum(axis=1)
y_lv3 = y_lv3[z_lv3 > 0].reset_index(drop=True)


cat_const = {k:full_desc(k) for k,v in dict_ID_Text.items() if (dict_ID_level[k] in ('D','C'))
    and (k in category_IDs)}
tagged_cat = set(cat_const.keys()).intersection(set(y.columns))
y_const = y[['Occurrence_No'] + list(tagged_cat)]
h_const = y_const[list(tagged_cat)].sum(axis=1)
v_const = y_const[list(tagged_cat)].sum(axis=0)
sel_col = v_const[v_const >= 5].index.tolist()
y_const = y_const.loc[h_const > 0, ['Occurrence_No'] + sel_col].reset_index(drop=True)
#y_const.rename(dict_ID_Text, axis=1, inplace=True)
#%% Change different level of cat here
y_lv = y_const.copy()
#%% Explore
#Number of occurrences in each category

y_lv.drop('Occurrence_No', axis=1).sum(axis=0).plot(x='category', y='number_of_comments', kind='bar', 
    legend=False, grid=True, figsize=(20, 5))
plt.title("Number of occurrences per category")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('category', fontsize=12)

#How many occurrences have multi labels?
z_lv = y_lv.drop('Occurrence_No', axis=1).sum(axis=1).value_counts()
plt.figure(figsize=(8,5))
ax = sns.barplot(z_lv.index, z_lv.values)
plt.title("Multiple categories per occurrence")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('# of categories', fontsize=12)

#distribution of the number of words in comment texts
lens = X['Occurrence_Description'].str.len()
lens.hist(bins = np.arange(0,2000,50))
plt.title("Distribution of the number of words in descriptions")

#%% Join data
#train_data = pd.read_csv('tagprediction_L2_atleast5occs.csv')
train_data = X.merge(y_lv, how='inner', on='Occurrence_No').drop(
    ['Occurrence_No'], axis=1)
train_data = train_data[sorted(train_data.columns)]
train_data = train_data[~pd.isnull(train_data['Occurrence_Description'])]

df_invg_orgn = train_data.copy()
categories = {k:dict_ID_Text[k] for k in train_data.columns if k not in 
    ['Occurrence_Description', 'Occurrence_Title', 'Occurrence_Type_Code']}
#train_data = train_data.rename(categories, axis='columns')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

train_data['Occurrence_Description'] = train_data['Occurrence_Description'].map(
    lambda com : clean_text(com))
train_data['Occurrence_Description'][0]
train_data.to_csv('tagprediction_L2_5occs_title_type.csv', index=False)

#%%
# duplicate classes
# =============================================================================
# z = train_data[list(categories.keys())].sum()
# sgl_inst_classes = z[z==1].index.tolist()
# dup_inst = train_data[train_data[sgl_inst_classes].sum(axis=1) == 1]
# train_data = pd.concat([train_data, dup_inst]).reset_index(drop=True)
# 
# train, test = train_test_split(train_data,  
#     random_state=12, test_size=0.1, shuffle=True)
# X_train = train.Occurrence_Description
# X_test = test.Occurrence_Description
# print(X_train.shape)
# print(X_test.shape)
# =============================================================================

#%%
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

mskf = MultilabelStratifiedKFold(n_splits=5, random_state=0)
folds = []
c_fold = 0
X = train_data[['Occurrence_Title', 'Occurrence_Description',
    'Occurrence_Type_Code']]
y = train_data[list(categories.keys())]

for train_index, test_index in mskf.split(X, y):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    #X_train = pd.Series(np.asarray(X_train).reshape(-1), name='Occurrence_Description')
    #X_test = pd.Series(np.asarray(X_test).reshape(-1), name='Occurrence_Description')
    #y_train = pd.DataFrame(y_train, columns=categories.keys())
    #y_test = pd.DataFrame(y_test, columns=categories.keys())

    c_fold += 1
    print('Training fold ', c_fold)
    model_svc = {}
    prediction_svc = {}
    f1score_svc = {}
    train_size_svc = {}
    for category in list(categories.keys()):
        print('... Processing {}'.format(category))
        train_size_svc.update({category: 
            [y_train[category].sum(), y_test[category].sum()]})
        # train the model            
        SVC_pipeline = Pipeline(steps=[
            ('tfm', 
                ColumnTransformer(
                    transformers=[
                        ('type', OneHotEncoder(), ['Occurrence_Type_Code']),
                        ('title', TfidfVectorizer(), 'Occurrence_Title'),
                        ('desc', TfidfVectorizer(stop_words=stop_words), 'Occurrence_Description')
                    ],
                    n_jobs=-1,
                    #transformer_weights={'type':0.4, 'title':0.4, 'desc':0.2}  # F1: 0.581
                    #no weight  # F1: 0.594
                )
            ),
            ('clf', 
                OneVsRestClassifier(
                    LinearSVC(C=1, class_weight='balanced', random_state=12), 
                    n_jobs=-1
                )
            )
        ])
        
        SVC_pipeline.fit(X_train, y_train[category])
        model_svc.update({category: SVC_pipeline})
        # compute the testing accuracy
        pred = SVC_pipeline.predict(X_test)
        prediction_svc.update({category: pred})
        
        if (y_test[category] == pred).sum() == 0:
            f1score_svc.update({category: 1.0})
        elif pred.sum() == 0:
            f1score_svc.update({category: 0.0})
        else:
            f1score_svc.update({category: f1_score(y_test[category], pred)})
        #print('Test accuracy is {}'.format(accuracy_score(y_test[category], 
        #    prediction)))
        #print('F1 score is {}'.format(f1_score(y_test[category], 
        #    prediction)))
    res = {'model_svc': model_svc,
        'prediction_svc': prediction_svc,
        'f1score_svc': f1score_svc,
        'train_size_svc': train_size_svc
    }
    folds.append(res)

list_dict = [f['f1score_svc'] for f in folds]
size = {k:int(sum(v)) for k,v in folds[0]['train_size_svc'].items()}
df = pd.DataFrame(list_dict)

#df = df.append(df.mean(numeric_only=True), ignore_index=True)
long_df = pd.melt(df)
long_df['text'] = long_df.variable.apply(lambda s: full_desc(s))
long_df['size'] = long_df.variable.apply(lambda s: size[s])
order = (long_df.groupby(['text', 'size'])['value']
    .aggregate({'mean':np.mean, 'std':np.std}).reset_index()
    .sort_values('mean', ascending=False))
long_df.to_csv('long_df_all_weight442.csv', index=False)
long_df = pd.read_csv('long_df_all_noweight.csv')
# =============================================================================
# plt.figure(figsize=(20, 10))
# ax = sns.barplot(x='text', y='value', data=long_df, order=order.text)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# =============================================================================

#long_df['variable'] = long_df['variable'].astype('category')
plt.figure(figsize=(10, 40))
ax = sns.barplot(x='value', y='text', data=long_df, order=order['text'])
for i in range(len(order)): 
    x_txt = order['mean'].values[i] + 0.1
    y_txt = i + 0.1
    ax.text(x_txt, y_txt, order['size'].values[i], color='blue')

order.plot.scatter(x='size', y='mean')
plt.xscale('log')
plt.xlabel('Number of occurrences per category')
plt.ylabel('F1 performance vs category')

order.plot.scatter(x='size', y='std')
plt.xscale('log')
plt.xlabel('Number of occurrences per category')
plt.title('Std Dev of F1 performance vs category')

plt.hist(order['mean'].values)

#%% Confusion matrix
from sklearn.metrics import multilabel_confusion_matrix

#%% Investigation
ID = [k for k,v in dict_ID_Text.items() if full_desc(k)=='Cabin > Communications (cabin)'][0]
df_invg = df_invg_orgn.loc[train_data[ID]==1, [ID, 'Occurrence_Description']]
df_invg_not = (df_invg_orgn.loc[
    (df_invg_orgn['Occurrence_Description'].str.contains('(?i)fire|flame'))
    , [ID, 'Occurrence_Description']])


#%%
RFC_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('rfc', RandomForestClassifier(n_estimators=100,
        class_weight='balanced', random_state=12
        , n_jobs=-1)),
])
RFC_pipeline.fit(X_train, y_train)
prediction = RFC_pipeline.predict(X_test)
prediction = pd.DataFrame(prediction, columns=y_test.columns)
labels = y_test.columns.tolist()
def func(r):
    lbl = y_test.columns[np.where(r)[0]].tolist()
    lbls = [dict_ID_Text[l] for l in lbl]
    return ', '.join(lbls)

y_test_labels = y_test.apply(func, axis=1)
pred_labels = prediction.apply(func, axis=1)
test_results = pd.DataFrame(X_test)
test_results['Truth'] = y_test_labels
test_results['Suggest'] = pred_labels
test_results = test_results[test_results.Suggest!='']

f1score_rfc = {}
for category in categories:
    f1score_rfc.update({category: f1_score(y_test[category], prediction[category])})
    
    
    
    
#%%
category = '9672'
print('... Processing {}'.format(category))
train_size.update({category: y_train[category].sum()})
# prepare features
vec = TfidfVectorizer(stop_words=stop_words)
x = vec.fit_transform(X_train)




#%%
SVC_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('clf', OneVsRestClassifier(
        RandomForestClassifier(n_estimators=100,
            class_weight='balanced', 
            random_state=12), 
        n_jobs=-1)
    ),
])
model = {}
prediction = {}
f1score = {}
train_size = {}
for category in categories:
    print('... Processing {}'.format(category))
    train_size.update({category: y_train[category].sum()})
    # train the model using X_dtm & y
    SVC_pipeline.fit(X_train, y_train[category])
    model.update({category: SVC_pipeline})
    # compute the testing accuracy
    pred = SVC_pipeline.predict(X_test)
    prediction.update({category: prediction})
    if (y_test[category] == pred).sum() == 0:
        f1score.update({category: np.nan})
    else:
        f1score.update({category: f1_score(y_test[category], pred)})
plt.plot(f1score.values())
#%%
category = '10213'
pred = model[category].predict(X_test)
pred = col_res[category]
lbl = y_test[category]








