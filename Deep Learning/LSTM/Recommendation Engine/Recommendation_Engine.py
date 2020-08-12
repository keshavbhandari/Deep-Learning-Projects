# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 17:07:24 2019

@author: kbhandari
"""

import pandas as pd
import numpy as np
from itertools import accumulate
import math
import pickle
import gc
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import BatchNormalization
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Concatenate
from keras.layers import GRU
from keras.layers import TimeDistributed
from keras.models import Model
from keras.layers import SpatialDropout1D
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Bidirectional
from keras.layers import Average
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,Callback
from sklearn.model_selection import KFold
from keras.models import load_model
#from tensorflow.keras import backend as K
from keras import backend as K
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

wd = "C:/Users/kbhandari/OneDrive - Epsilon/Desktop/Guitar Center Raw Data/ARRAYS/"
RunType = 'Score'
num_cols = ['AVG_MARKDOWN_IN_CATEGORY', 'AVG_UNIT_RETAIL_IN_CATEGORY', 'NUMBER_ITEMS_FROM_CATEGORY', 'RETURNS_IN_CATEGORY', 'AVG_MARKDOWN_ALL', 'AVG_UNIT_RETAIL_ALL', 'NUMBER_ITEMS_ALL', 'RETURN_COUNT_ALL', 'AVG_DOW_ALL', 'AVG_MONTH_ALL', 'VISIT_COUNT_ALL', 'MARKDOWN_IN_CATEGORY_NORM1', 'AUR_IN_CATEGORY_NORM1']
cat_cols = ['CATEGORY3_CD']

def get_max_length(list_of_list):
    max_value = 0
    for l in list_of_list:
        if len(l) > max_value:
            max_value = len(l)
    return max_value

def get_padding(numerical_cols, categorical_cols, wd, RunType = 'Model',
                tokenizer = None, max_padding_length = None):
    if RunType == 'Model':
        data_folder = 'Data/'
        word_index, max_length, padded_docs, word_tokenizer = {},{},{},{}
    else:
        data_folder = 'Scoring_Data/'
        padded_docs = {} 
    
    for col in num_cols:
        print(col)
        array = pd.read_csv(wd + data_folder + "{0}.csv".format(col))
        array = array.sort_values(['SHIPTO_ACCT_SRC_NBR', 'CATEGORY3_CD'], ascending=[True, True])
        array.reset_index(drop=True, inplace=True)
        if RunType == 'Model':
            array = np.array(array.iloc[:,2:])
            array = array.astype(np.float)   
            max_length_value = get_max_length(array)
            max_length[col] = max_length_value if max_length_value < 50 else 50
            #    padded_docs[col] = pad_sequences(array, maxlen=max_length[col], padding='post')
        else:
            array = np.array(array.iloc[:,2:])
            array = array.astype(np.float) 
        padded_docs[col] = array
        del array
        gc.collect()
    
    for col in cat_cols:
        print(col)
        data = pd.read_csv(wd + data_folder + "{0}.csv".format(col))
        data.columns = ['SHIPTO_ACCT_SRC_NBR', col]
        data = data.sort_values(['SHIPTO_ACCT_SRC_NBR', 'CATEGORY3_CD'], ascending=[True, True])
        data.reset_index(drop=True, inplace=True)
        
        if RunType == 'Model':
            t = Tokenizer()
            t.fit_on_texts(data[col].astype(str))
            word_index[col] = t.word_index
            txt_to_seq = t.texts_to_sequences(data[col].astype(str))
            max_length_value = get_max_length(txt_to_seq)
            max_length[col] = max_length_value if max_length_value < 50 else 50
            padded_docs[col] = pad_sequences(txt_to_seq, maxlen=max_length[col], padding='post')
            word_tokenizer[col] = t
        else:
            t = tokenizer[col]
            txt_to_seq = t.texts_to_sequences(data[col].astype(str))
            padded_docs[col] = pad_sequences(txt_to_seq, maxlen=max_padding_length[col], padding='post')
        del data, txt_to_seq
        gc.collect()
        
    if RunType == 'Model':
        return word_index, max_length, padded_docs, word_tokenizer
    else:
        return padded_docs

if RunType == 'Model':
    word_index, max_length, padded_docs, tokenizer = get_padding(num_cols, cat_cols, wd, RunType = 'Model')
    pickle.dump(word_index, open(wd+"word_index.p", "wb"))
    pickle.dump(max_length, open(wd+"max_length.p", "wb"))
    pickle.dump(tokenizer, open(wd+"tokenizer.p", "wb"))
else:
    word_index = pickle.load(open(wd+"word_index.p", "rb"))
    max_length = pickle.load(open(wd+"max_length.p", "rb"))
    tokenizer = pickle.load(open(wd+"tokenizer.p", "rb"))
    padded_docs = get_padding(num_cols, cat_cols, wd, RunType = 'Score', tokenizer = tokenizer, max_padding_length = max_length)

if RunType == 'Model':
    data = pd.read_csv(wd+'Data/'+"DV.csv")
    data = data.sort_values(['SHIPTO_ACCT_SRC_NBR', 'CATEGORY3_CD'], ascending=[True, True])
    data.reset_index(drop=True, inplace=True)
    cid_cat_label = data.copy()
    data.drop(['SHIPTO_ACCT_SRC_NBR', 'CATEGORY3_CD'], axis = 1, inplace = True)
else:
    cid_cat_label = pd.read_csv(wd+'Scoring_Data/'+"CATEGORY3_CD.csv")
    cid_cat_label = cid_cat_label.sort_values(['SHIPTO_ACCT_SRC_NBR', 'CATEGORY3_CD'], ascending=[True, True])
    cid_cat_label.reset_index(drop=True, inplace=True)

if RunType == 'Model':
    # Train-Test Split
    train_y, validation_y = train_test_split(data, test_size = 0.5, random_state = 0, shuffle=False)
    del data
    gc.collect()
    
    validation_idx = validation_y.index  
    train_idx = train_y.index
    
    train_iv_list, validation_iv_list = [], []
    for i in cat_cols: 
        print(i)
        train_iv_list.append(padded_docs[i][train_idx])
        validation_iv_list.append(padded_docs[i][validation_idx])
    for i in num_cols:
        print(i)
        array = padded_docs[i][train_idx]
        array = array.reshape(array.shape[0],array.shape[1],1)
        train_iv_list.append(array)
        
        array = padded_docs[i][validation_idx]
        array = array.reshape(array.shape[0],array.shape[1],1)
        validation_iv_list.append(array)
else:
    test_iv_list = []
    for i in cat_cols: 
        print(i)
        test_iv_list.append(padded_docs[i])
    for i in num_cols:
        print(i)
        array = padded_docs[i]
        array = array.reshape(array.shape[0],array.shape[1],1)
        test_iv_list.append(array)
        
del padded_docs
gc.collect()

# Loss Function  
gamma = 2.0
epsilon = K.epsilon()
def focal_loss(y_true, y_pred):
    pt = y_pred * y_true + (1-y_pred) * (1-y_true)
    pt = K.clip(pt, epsilon, 1-epsilon)
    CE = -K.log(pt)
    FL = K.pow(1-pt, gamma) * CE
    loss = K.sum(FL, axis=1)
    return loss

import keras.backend.tensorflow_backend as tfb

POS_WEIGHT = 10  # multiplier for positive targets, needs to be tuned

def weighted_binary_crossentropy(target, output):
    """
    Weighted binary crossentropy between an output tensor 
    and a target tensor. POS_WEIGHT is used as a multiplier 
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    # transform back to logits
    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.math.log(output / (1 - output))
    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(labels=target,
                                                    logits=output,
                                                    pos_weight=POS_WEIGHT)
    return tf.reduce_mean(loss, axis=-1)

# Model
def create_model(categorical_cols, numerical_cols, word_index, max_length, final_layer):
    inputs = []
    sequence_embeddings = []
    
    for col in categorical_cols:
        input_array = Input(shape=(max_length[col],))
        inputs.append(input_array)
        vocab_size = len(word_index[col]) + 1
        embed_size = int(min(np.ceil((vocab_size)/2), 50))
#        embed_size = int(min(np.ceil((vocab_size)/2), 30)) #25
        embedding = Embedding(vocab_size, embed_size, input_length=max_length[col], name='{0}_embed'.format(col), trainable=True)(input_array)
        embedding = SpatialDropout1D(0.15)(embedding)
        embedding = Reshape(target_shape=(embed_size*max_length[col],))(embedding)
#        lstm = LSTM(50, dropout=0.2, recurrent_dropout=0.2)(embedding)
        sequence_embeddings.append(embedding)
        
    for col in numerical_cols:
        input_array = Input(shape=(max_length[col],1))
        inputs.append(input_array)
        RNN = LSTM(max_length[col], dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(input_array)
        RNN = TimeDistributed(Dense(max_length[col], activation='relu'))(RNN)
        RNN = Flatten()(RNN)
        sequence_embeddings.append(RNN)
                
    x = Concatenate()(sequence_embeddings)
    
    x = BatchNormalization()(x)
    x = Dense(500, activation='relu')(x)
    x = Dropout(.2)(x)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)
    x = Dropout(.2)(x)
    x = BatchNormalization()(x)
    x = Dense(150, activation='relu')(x)
    x = Dropout(.2)(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.2)(x)
    x = BatchNormalization()(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(.2)(x)
    x = BatchNormalization()(x)
    x = Dense(25, activation='relu')(x)
    output = Dense(final_layer, activation='sigmoid', name='model')(x)
    
    model = Model(inputs, output)
    model.compile(loss = focal_loss , optimizer = "adam", metrics=['accuracy'])    
    
    return model

if RunType == 'Model':
    final_layer = train_y.shape[1]
    model = create_model(cat_cols, num_cols, word_index, max_length, final_layer)
    print(model.summary())
    file_path = wd + 'Model/'
    filepath=file_path+"model.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_learning_rate = ReduceLROnPlateau(monitor = "val_loss", factor = 0.6, patience=1, min_lr=1e-6,  verbose=1, mode = 'min')
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1)
    callbacks_list = [checkpoint,reduce_learning_rate,early_stopping]
    entity_embedding_model = model.fit(train_iv_list,train_y, 
                             validation_data=(validation_iv_list,validation_y), 
                             epochs=100,#100
                             callbacks=callbacks_list, 
                             shuffle=False, 
                             batch_size=2048, #256
                             verbose=1)

    filepath = file_path + "/model.best.hdf5"
    #new_model = load_model(filepath)
    new_model = load_model(filepath, custom_objects={'focal_loss': focal_loss})
    
    predictions = new_model.predict(validation_iv_list,verbose=1, batch_size=int(0.05*len(validation_iv_list[0])))
    #predictions.columns = validation_y.columns
    from sklearn.metrics import roc_auc_score
    print("AUC:",roc_auc_score(validation_y, predictions))
    
    # Hit Rate Calculation
    cid_cat_label_validation = cid_cat_label.iloc[validation_idx]
    cid_cat_label_validation['Resell_Probability'] = predictions
    cid_cat_label_validation['Rank'] = cid_cat_label_validation.groupby(['SHIPTO_ACCT_SRC_NBR'])['Resell_Probability'].rank(method='first',ascending=False).astype(int)
    
    # Filter customers who have at least one transaction in DV period
    cid_cat_label_validation = cid_cat_label_validation.groupby(['SHIPTO_ACCT_SRC_NBR']).filter(lambda x: x['1'].max() > 0)
    
    # Model Top K Recommendation Calculation
    k = 1
    hit_rate = cid_cat_label_validation[cid_cat_label_validation['Rank'] <= k]
    hits = sum(hit_rate.groupby(['SHIPTO_ACCT_SRC_NBR'])['1'].max())
    unique_customers = hit_rate['SHIPTO_ACCT_SRC_NBR'].nunique()
    hits/unique_customers
    
    # Global Popularity Top K Recommendation Calculation
    k = 1
    counts = cid_cat_label['CATEGORY3_CD'].value_counts().reset_index(drop=False)
    counts = counts.iloc[0:k,0].values.tolist()
    hit_rate = cid_cat_label_validation[(cid_cat_label_validation['CATEGORY3_CD'].isin(counts)) & (cid_cat_label_validation['1'] == 1)]
    hits = len(hit_rate)
    unique_customers = cid_cat_label_validation['SHIPTO_ACCT_SRC_NBR'].nunique()
    hits/unique_customers
    
    # Customer Most Popular Top K Recommendation Calculation
    col = 'NUMBER_ITEMS_FROM_CATEGORY'
    quantity = pd.read_csv(wd+'Data/'+"{0}.csv".format(col))
    col_list= list(quantity)
    col_list = list(set(col_list) - set(['SHIPTO_ACCT_SRC_NBR','CATEGORY3_CD']))
    quantity['Quantity'] = quantity[col_list].sum(axis=1)
    k=1
    hit_rate = pd.merge(cid_cat_label_validation, quantity[['SHIPTO_ACCT_SRC_NBR','CATEGORY3_CD','Quantity']], on = ['SHIPTO_ACCT_SRC_NBR','CATEGORY3_CD'], how='left')
    hit_rate['Rank'] = hit_rate.groupby(['SHIPTO_ACCT_SRC_NBR'])['Quantity'].rank(method='first',ascending=False).astype(int)
    hit_rate = hit_rate[hit_rate['Rank'] <= k]
    hits = sum(hit_rate.groupby(['SHIPTO_ACCT_SRC_NBR'])['1'].max())
    unique_customers = hit_rate['SHIPTO_ACCT_SRC_NBR'].nunique()
    hits/unique_customers

else:
    file_path = wd + 'Model/'
    filepath = file_path + "/model.best.hdf5"
    #new_model = load_model(filepath)
    new_model = load_model(filepath, custom_objects={'focal_loss': focal_loss})
    
    predictions = new_model.predict(test_iv_list,verbose=1, batch_size=int(0.05*len(test_iv_list[0])))
    del test_iv_list
    gc.collect()
    cid_cat_label['Resell_Probability'] = predictions
    cid_cat_label['Rank'] = cid_cat_label.groupby(['SHIPTO_ACCT_SRC_NBR'])['Resell_Probability'].rank(method='first',ascending=False).astype(int)
    k = 10
    cid_cat_label = cid_cat_label[cid_cat_label['Rank'] <= k]
    cid_cat_label.to_csv(wd+'Resell_Scores_Top_10.csv', index=False)








pred_max = np.argmax(predictions, axis=1)
pred_max = predictions.argsort()[:,::-1][:,:2]

hit = 0
no_hit = 0
for i in range(len(validation_y)):
    if validation_y.iloc[i,pred_max[i][0]] == 1: #or validation_y.iloc[i,pred_max[i][1]] == 1:
        hit += 1
    else:
        no_hit += 1

hit/len(validation_y)


global_popular = validation_y.sum(axis = 0).sort_values(ascending = False)
global_popular[0] / len(validation_y)











#data = pd.read_csv(wd+"part-00000-012a6f20-a346-4eaf-ab5a-a8cbf080a353-c000.csv", nrows=300000)
#print(data.shape)
#data = data[~((data['NUMBER_ITEMS_FROM_CATEGORY'] == 0) & (data['RETURNS_IN_CATEGORY'] == 0))]
#print(data.shape)

#Delete single visit customers
data = data[data.groupby(['SHIPTO_ACCT_SRC_NBR'])['WEEK_NUMBER'].transform(len) > 2].reset_index(drop = True)
print(data.shape)

def replace_recency(data):
    columns = data.columns
    cols_to_replace = {}
    for col in columns:
        if 'CATEGORY2_CD' in col or 'CATEGORY3_CD' in col:
            cols_to_replace[col] = 'MISSING'
        else:
            cols_to_replace[col] = -999
    data = data.fillna(cols_to_replace)
    return data

data = replace_recency(data)
data.isna().sum()

DV = data['CATEGORY3_CD'].unique().tolist()  
IV = list(set(data.columns.tolist()) - set(['SHIPTO_ACCT_SRC_NBR','WEEK_NUMBER']))

agg_df = data.groupby(['SHIPTO_ACCT_SRC_NBR','WEEK_NUMBER']).agg({
        'CATEGORY3_CD': lambda s: ', '.join(s)
        }).reset_index()
    
# Create DV
def f(row, dv_label):
    if dv_label in row['CATEGORY3_CD']:
        val = 1
    else:
        val = 0
    return val  
DV_df = agg_df[['SHIPTO_ACCT_SRC_NBR','CATEGORY3_CD']]
for i in DV:
    print(i)
    DV_df[i] = agg_df.apply(f, dv_label = i, axis=1)
    DV_df[i] = DV_df.groupby(['SHIPTO_ACCT_SRC_NBR'])[i].shift(-1)
DV_df = DV_df.drop(['SHIPTO_ACCT_SRC_NBR','CATEGORY3_CD'], axis=1)

# Cumulative IV
agg_df = data.groupby(['SHIPTO_ACCT_SRC_NBR','WEEK_NUMBER']).agg({
        'CATEGORY2_CD': lambda s: list(s),
        'CATEGORY3_CD': lambda s: list(s),
        'AVG_MARKDOWN_IN_CATEGORY': lambda s: list(s),
        'AVG_UNIT_RETAIL_IN_CATEGORY': lambda s: list(s),
        'NUMBER_ITEMS_FROM_CATEGORY': lambda s: list(s),
        'RETURNS_IN_CATEGORY': lambda s: list(s),
        'AVG_MARKDOWN_ALL': lambda s: list(s),
        'AVG_UNIT_RETAIL_ALL': lambda s: list(s),
        'NUMBER_ITEMS_ALL': lambda s: list(s),
        'RETURN_COUNT_ALL': lambda s: list(s),
        'AVG_DOW_ALL': lambda s: list(s),
        'AVG_MONTH_ALL': lambda s: list(s),
        'VISIT_COUNT_ALL': lambda s: list(s),
        'MARKDOWN_IN_CATEGORY_NORM1': lambda s: list(s),
        'AUR_IN_CATEGORY_NORM1': lambda s: list(s)
        }).reset_index()

def cum_concat(x):
    return list(accumulate(x))
f = lambda x: [item[-50:] for item in cum_concat([i for i in x])] #cum_concat([i for i in x])
for i in IV:
    print(i)
    b = agg_df.groupby(['SHIPTO_ACCT_SRC_NBR'])[i].apply(f)
    agg_df[i]=[item for sublist in b for item in sublist]

# Padding
agg_df['CATEGORY2_CD'] = agg_df['CATEGORY2_CD'].apply(lambda x: ','.join(map(str, x)))
agg_df['CATEGORY3_CD'] = agg_df['CATEGORY3_CD'].apply(lambda x: ','.join(map(str, x)))

def get_padding(data, categorical_cols, tokenizer = None, max_padding_length = None):
    
    def get_max_length(list_of_list):
        max_value = 0
        for l in list_of_list:
            if len(l) > max_value:
                max_value = len(l)
        return max_value
    
    # Tokenize Sentences
    word_index, max_length, padded_docs, word_tokenizer = {},{},{},{}
    if tokenizer is None:    
        for col in categorical_cols:            
            print("Processing column:", col)        
            t = Tokenizer()
            t.fit_on_texts(data[col].astype(str))
            word_index[col] = t.word_index
            txt_to_seq = t.texts_to_sequences(data[col].astype(str))
            max_length[col] = get_max_length(txt_to_seq) #len(max(txt_to_seq, key = lambda x: len(x)))
            padded_docs[col] = pad_sequences(txt_to_seq, maxlen=max_length[col], padding='post')
            word_tokenizer[col] = t
        return word_index, max_length, padded_docs, word_tokenizer
        
    else:
        for col in categorical_cols:
            print("Processing column:", col)
            t = tokenizer[col]
            txt_to_seq = t.texts_to_sequences(data[col].astype(str))
            padded_docs[col] = pad_sequences(txt_to_seq, maxlen=max_padding_length[col], padding='post')        
        return padded_docs

word_index, max_length, padded_docs, tokenizer = get_padding(agg_df, ['CATEGORY2_CD', 'CATEGORY3_CD'])

#Remove Nan from DV/IV
DV_df.isna().sum()
DV_df = DV_df.dropna()
agg_df = agg_df.loc[DV_df.index]
agg_df.reset_index(drop = True, inplace = True)
DV_df.reset_index(drop = True, inplace = True)

validation = agg_df.sort_values('WEEK_NUMBER', ascending=False).drop_duplicates(['SHIPTO_ACCT_SRC_NBR'])
validation_idx = validation.index
validation_y = DV_df.loc[validation_idx]
validation_y.reset_index(drop = True, inplace = True)
train = agg_df.drop(validation_idx, axis=0)  
train_idx = train.index
train_y = DV_df.loc[train_idx]
train_y.reset_index(drop = True, inplace = True)

# IV Array
#train_iv_list, validation_iv_list = [], []
#for i in IV: 
#    if i == 'CATEGORY2_CD' or i == 'CATEGORY3_CD':
#        arr = pd.DataFrame(agg_df[i].values.tolist()).fillna('<EOS>').to_numpy()
#    else:
#        arr = pd.DataFrame(agg_df[i].values.tolist()).fillna(-999).to_numpy()
#    print(i, arr.shape[1])
#    train_iv_list.append(arr[train_idx])
#    validation_iv_list.append(arr[validation_idx])
    
train_iv_list, validation_iv_list = [], []
cat_cols = ['CATEGORY2_CD', 'CATEGORY3_CD']
num_cols = ['AVG_MARKDOWN_IN_CATEGORY', 'AVG_UNIT_RETAIL_IN_CATEGORY', 'NUMBER_ITEMS_FROM_CATEGORY', 'RETURNS_IN_CATEGORY', 'AVG_MARKDOWN_ALL', 'AVG_UNIT_RETAIL_ALL', 'NUMBER_ITEMS_ALL', 'RETURN_COUNT_ALL', 'AVG_DOW_ALL', 'AVG_MONTH_ALL', 'VISIT_COUNT_ALL', 'MARKDOWN_IN_CATEGORY_NORM1', 'AUR_IN_CATEGORY_NORM1']

for i in cat_cols: 
    if i == 'CATEGORY2_CD' or i == 'CATEGORY3_CD':
        print(i)
        train_iv_list.append(padded_docs[i][train_idx])
        validation_iv_list.append(padded_docs[i][validation_idx])
first_time = True
for i in num_cols:
    arr = pd.DataFrame(agg_df[i].values.tolist()).fillna(-999).to_numpy()
    arr = arr.reshape(arr.shape[0],arr.shape[1],1)
    if first_time:
        arr_combined = arr
        first_time = False
    else:
        arr_combined = np.concatenate([arr_combined, arr], -1)
    
    print(i, arr_combined.shape)
#    max_length[i] = arr.shape[1]
#    train_iv_list.append(arr[train_idx])
#    validation_iv_list.append(arr[validation_idx])

train_iv_list.append(arr_combined[train_idx])
validation_iv_list.append(arr_combined[validation_idx])

# Model
def create_model(categorical_cols, numerical_cols, word_index, max_length, final_layer):
    inputs = []
    sequence_embeddings = []
    
    for col in categorical_cols:
        input_array = Input(shape=(max_length[col],))
        inputs.append(input_array)
        vocab_size = len(word_index[col]) + 1
#        embed_size = int(min(np.ceil((vocab_size)/2), 30)) #25
        embedding = Embedding(vocab_size, 50, input_length=max_length[col], name='{0}_embed'.format(col), trainable=True)(input_array)
        embedding = SpatialDropout1D(0.15)(embedding)
        lstm = LSTM(50, dropout=0.2, recurrent_dropout=0.2)(embedding)
        sequence_embeddings.append(lstm)
        
#    for col in numerical_cols:
#        input_array = Input(shape=(max_length[col],1))
#        inputs.append(input_array)
#        lstm = LSTM(max_length[col], dropout=0.2, recurrent_dropout=0.2)(input_array)
#        sequence_embeddings.append(lstm)
    
    input_array = Input(shape=(50,13))
    inputs.append(input_array)
    lstm = LSTM(50, dropout=0.2, recurrent_dropout=0.2)(input_array)
    sequence_embeddings.append(lstm)
                
    x = Concatenate()(sequence_embeddings)
    x = BatchNormalization()(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(.2)(x)
    x = BatchNormalization()(x)
    output = Dense(final_layer, activation='sigmoid', name='model')(x)
    
    model = Model(inputs, output)
    model.compile(loss = 'binary_crossentropy', optimizer = "adam", metrics=['accuracy'])    
    
    return model

def create_class_weight(labels_dict, mu=0.15, default=False):
    total = sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()

    if default == False:
        for key in keys:
            score = math.log(mu*total/float(labels_dict[key]))
            class_weight[key] = score if score > 1.0 else 1.0
    else:
        for key in keys:
            class_weight[key] = 1.0
            
    return class_weight

#labels_dict = {0: 2813, 1: 78, 2: 2814, 3: 78, 4: 7914, 5: 248, 6: 7914, 7: 248}
labels_dict = dict(train_y.sum().reset_index(drop=True))
for key, value in labels_dict.items():
    if labels_dict[key] == 0:
        labels_dict[key] = 1
class_weights = create_class_weight(labels_dict, default=True)

final_layer = DV_df.shape[1]
model = create_model(cat_cols, num_cols, word_index, max_length, final_layer)
print(model.summary())
file_path = "C:/Users/kbhandari/OneDrive - Epsilon/Desktop/Not Just OLS/Recommendation_Engine/"
filepath=file_path+"/model.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
reduce_learning_rate = ReduceLROnPlateau(monitor = "val_loss", factor = 0.6, patience=1, min_lr=1e-6,  verbose=1, mode = 'min')
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
callbacks_list = [checkpoint,reduce_learning_rate,early_stopping]
entity_embedding_model = model.fit(train_iv_list,train_y, 
                         validation_data=(validation_iv_list,validation_y), 
                         epochs=100,#100
                         callbacks=callbacks_list, 
                         shuffle=False, 
                         batch_size=64, #64
                         verbose=1,
                         class_weight=class_weights)


filepath = file_path + "/model.best.hdf5"
new_model = load_model(filepath)

predictions = new_model.predict(validation_iv_list,verbose=1, batch_size=int(0.05*len(validation_iv_list[0])))


pred_max = np.argmax(predictions, axis=1)
pred_max = predictions.argsort()[:,::-1][:,:2]
hit = 0
no_hit = 0
for i in range(len(validation_y)):
    if validation_y.iloc[i,pred_max[i][0]] == 1: #or validation_y.iloc[i,pred_max[i][1]] == 1:
        hit += 1
    else:
        no_hit += 1

hit/len(validation_y)


#Trials
from numpy import array
n = array([array([1]), array([2,3]), array([4,5,6])], dtype=object)
np.array(list(accumulate(n, lambda x, y: np.concatenate([y,x])[0:4])))

agg_df2 = agg_df.copy()
def cum_concat(x):
    return list(accumulate(x))
f = lambda x: [item[-5:] for item in cum_concat([i for i in x])]
for i in IV:
    print(i)
    b = agg_df2.groupby(['SHIPTO_ACCT_SRC_NBR'])[i].apply(f)
    agg_df2[i]=[item for sublist in b for item in sublist]
    break


import numpy_indexed as npi
from numpy import array
a = array([
       [1, 1, 275],
       [1, 1, 441],
       [1, 2, 494],
       [1, 3, 593],
       [2, 1, 679],
       [2, 1, 533],
       [2, 2, 686],
       [3, 1, 559],
       [3, 2, 219],
       [3, 2, 455],
       [4, 1, 605],
       [4, 1, 468],
       [4, 1, 692],
       [4, 1, 613]])
b = npi.group_by(a[:, [0,1]]).split(a[:, 2])
npi.group_by(a[:, [0,1]]).split(a[:, 2])

import bcolz
#Save to memory
bcolz_array = bcolz.carray(a, mode='w', cparams=bcolz.cparams(quantize=1))
bcolz_array.flush()
#bcolz.carray(data[0:training_rows], mode='w',cparams=bcolz.cparams(quantize=QUANTIZE_LEVEL),expectedlen=EXPECTED_ROWS)

#Write to disk
wd = "C:/Users/kbhandari/OneDrive - Epsilon/Desktop/Guitar Center Raw Data/ARRAYS/"
variable = 'bcolz_test'
bcolz_array = bcolz.carray(a ,rootdir= wd + variable + '.dat', mode='w')
bcolz_array.flush()

#Open file
c = bcolz.open(wd + variable + '.dat')

#Open as numpy array
d = bcolz.open(wd + variable + '.dat')[0:5]

#Append to bcolz
rows = a[:,:]+1000
c.append(rows)
c.shape
