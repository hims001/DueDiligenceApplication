import numpy as np
from os import path
import os
import pandas as pd
import sqlite3
from eventregistry import *
from datetime import date, timedelta, datetime
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import string
import warnings
import ConfigurationManager as cfg
import sys
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn import metrics

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    from keras.models import load_model
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, GlobalMaxPool1D
    from keras.layers import Flatten
    from keras.layers import Embedding
    from keras.callbacks import EarlyStopping,CSVLogger
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#print('CONFIG VALUE :' + cfg.read_config('sql_db_path'))

model_path = cfg.read_config('model_path')
sql_db_path = cfg.read_config('sql_db_path')
max_length = int(cfg.read_config('max_length'))
embedding_dim = int(cfg.read_config('embedding_dim'))
sentence_buffer = int(cfg.read_config('sentence_buffer'))

def train_sequential_model(X, Y, is_split_data):
    print('Number of records for training the model : ', str(len(X)))
    # Data pre-processing
    for index, value in enumerate(X):
        #print(value)
        #tokens = unique_list(clean_article(value))
        tokens = clean_article(value)
        #print('\n-------------------------------------------\n')
        # Slice out max length characters from the article
        # if len(tokens) > max_length:
        #     tokens = tokens[:max_length]

        cleaned_article = " ".join(tokens)
        #print(cleaned_article)
        # Grab Before and After 2 sentences of sentence with ENTITY word
        sentences = sent_tokenize(cleaned_article)
        indices = [idx for idx, sent in enumerate(sentences) if 'ENTITY' in sent]
        #print(indices)
        extended_indices = []
        for i, sentence in enumerate(sentences):
            extended_indices.extend(list(set([i for index in indices if abs(index-i) <= sentence_buffer])))
        #print(extended_indices)
        #print(len(sentences))
        #print(cleaned_article)
        desired_list = list(itemgetter(*extended_indices)(sentences))

        #print(" ".join(desired_list))

        X[index] = " ".join(desired_list)
        #print('\n++++++++++++++++++++++++++++++++++++++++++++++\n')
        #print(X[index])#.encode('utf-8'))
        #print('\n++++++++++++++++++++++++++++++++++++++++++++++\n')

    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(X)
    vocab_size = len(t.word_index) + 1
    print('vocab_size : ' + str(vocab_size))

    if path.exists(model_path + 'trained_model.h5'):
        #model = load_model(model_path + 'trained_model.h5')
        os.remove(model_path + 'trained_model.h5')

        # max_length = int(cfg.read_config('max_length'))
        #
        # for index, value in enumerate(X):
        #     tokens = unique_list(clean_article(value))
        #     # Slice out max length characters from the article
        #     if len(tokens) > max_length:
        #         tokens = tokens[:max_length]
        #     X[index] = " ".join(tokens)
    #else:
    # pad documents to a max length of words
    max_length = max([len(sentence.split()) for sentence in X])
    print('max_length : ' + str(max_length))
    # Write Max length to config file
    cfg.write_config('max_length', str(max_length))

    # load the whole embedding into memory
    embedding_matrix = create_embedding_matrix(model_path + 'glove.6B.50d.txt', t.word_index, embedding_dim)

    # define model
    model = Sequential()
    e = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(GlobalMaxPool1D())
    #model.add(Dropout(0.2))
    #model.add(Flatten())
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    # integer encode the documents
    encoded_docs = t.texts_to_sequences(X)
    #print(encoded_docs)

    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    #print(padded_docs)

    # summarize the model
    print(model.summary())

    X_train,X_test,y_train,y_test = train_test_split(padded_docs,Y, random_state=0)
    #if is_split_data:
    # fit the model
    #X_train = padded_docs[:len(padded_docs)-100]
    #y_train = Y[:len(padded_docs) - 100]
    print('X_train, y_train', str(len(X_train)), str(len(y_train)))
    #X_test = padded_docs[-100:]
    #y_test = Y[-100:]
    print('X_test, y_test', str(len(X_test)), str(len(y_test)))

    # X_val=X_train[-50:]
    # y_val=y_train[-50:]
    # print('X_val, y_val', str(len(X_val)), str(len(y_val)))

    es_callback = EarlyStopping(monitor='val_acc', patience=3)
    logger_callback = CSVLogger('training.log', separator=',', append=True)
    model.fit(X_train, y_train, batch_size=10, epochs=20, validation_data=(X_test, y_test), shuffle=True, verbose=1, callbacks=[logger_callback])#, es_callback])
    #, validation_freq=5,
    #, validation_split = 0.2

    # evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    # else:
    #     model.fit(padded_docs, Y, epochs=30, shuffle=True, verbose=1)
    #     loss, accuracy = model.evaluate(padded_docs, Y, verbose=1)

    y_pred = model.predict_classes(x=X_test, batch_size=5, verbose=2)

    print(metrics.confusion_matrix(y_test, y_pred))

    print('Accuracy: %f' % (accuracy * 100))
    print('Loss: ', loss)

    # Save the model to disk
    model.save(model_path + 'trained_model.h5')
    print("Model saved to disk.")

def fetch_data_db():
    conn = sqlite3.connect(sql_db_path)
    c = conn.cursor()
    myquery = ("SELECT Id,ArticleText,Outcome FROM DueDiligenceUI_trainingmodel WHERE IsTrained=1 and Outcome IS NOT NULL")
    c.execute(myquery)
    result = c.fetchall()
    X = []
    Y = []
    Ids = []
    for record in result:
        Ids.append(str(record[0]))
        X.append(record[1])
        Y.append(record[2])
    #print("UPDATE DueDiligenceUI_trainingmodel SET IsTrained=1 WHERE Id IN (" + ','.join(Ids) + ")")
    myquery = ("UPDATE DueDiligenceUI_trainingmodel SET IsTrained=1 WHERE Id IN (" + ','.join(Ids) + ")")
    c.execute(myquery)
    conn.commit()
    c.close()
    conn.close()

    return X, Y, False

def fetch_data_CSV():
    training_data = pd.read_csv(model_path + 'TrainingData.csv', encoding='latin1')
    training_data = training_data.drop('Name', axis=1)

    conn = sqlite3.connect(sql_db_path)
    c = conn.cursor()
    cursorObj = conn.cursor()

    for ind in training_data.index:
        content = training_data["Data"][ind]
        outcome = training_data["Pattern"][ind]
        articleDateTime = datetime.today()
        # Insert article into database
        sql = "INSERT INTO DueDiligenceUI_trainingmodel (ArticleText, Outcome, TrainingDate, SearchModel_id, IsTrained, Url) " \
              "VALUES(?,'" + str(outcome) + "','" + str(articleDateTime) + "',0,1,'') "
        #print(sql)
        cursorObj.execute(sql, [content])
        conn.commit()

    cursorObj.close()
    conn.close()
    return training_data['Data'], training_data['Pattern'], True

    # turn a doc into clean tokens

def clean_article(article_content):
    tokens = [word.replace('\\n', ' ').replace('\n', ' ').replace('  ', ' ') for word in article_content.split()]
    #print(tokens)
    # split again in order to accomodate replaced spaces into tokens by white space
    article_content = " ".join(tokens)
    tokens = article_content.split()
    #print(tokens)
    # remove punctuation from each token except Full Stop(.)
    table = str.maketrans('', '', '!"#$%&\'()*+,/:;<=>?@[\\]^_`{|}~')
    tokens = [w.translate(table) for w in tokens]
    #print(tokens)
    # remove remaining tokens that are not alphabetic
    tokens = [word.strip() for word in tokens if word.strip().isalpha() or '.' in word or '-' in word]
    #print(tokens)
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    #print(tokens)
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1 or '.' in word or '-' in word]
    #print(tokens)
    return tokens

def unique_list(list):
    uniquelist = []
    [uniquelist.append(word) for word in list if word not in uniquelist or word == 'ENTITY']
    return uniquelist

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath, encoding='utf-8') as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

def fetch_data_eventregistry(entityname):
    er = EventRegistry(apiKey="f4a005ab-a24f-487e-bff4-f39b1b2ba6c2")
    # q = QueryArticlesIter(
    #     keywords=entityname,
    #     dataType=["news"],  # , "blog"],
    #     lang="eng",
    #     dateStart=date.today() - timedelta(days=30),
    #     dateEnd=date.today())
    cq = ComplexArticleQuery(
        query=CombinedQuery.AND(
            [
                BaseQuery(keyword=QueryItems.AND(["narendra", "modi"]),
                          #sourceLocationUri=er.getLocationUri("United States"),
                          lang="eng",
                          dateStart=date.today() - timedelta(days=45),
                          dateEnd=date.today()
                          ),
                BaseQuery(keyword=QueryItems.OR(
                    ["sanction", "bribery", "laundering", "corruption", "blacklist", "crime", "scam", "fraud"]))
                # "drugs","trafficking","gambling","illegal","smuggling","terrorism",
                # "extortion","forgery","tax evasion","SDN","burglary","robbery","murder"]))
            ])
    )
    q = QueryArticles.initWithComplexQuery(cq)
    q.setRequestedResult(RequestArticlesInfo(page=1,
                                             count=10,
                                             sortBy="date", sortByAsc=False,
                                             returnInfo=ReturnInfo()))
    res = er.execQuery(q)

    X = []

    conn = sqlite3.connect(sql_db_path)
    c = conn.cursor()
    cursorObj = conn.cursor()
    # obtain at most 500 newest articles or blog posts
    for article in res['articles']['results']:
        content = article["body"]
        articleDateTime = article["dateTime"].replace('T', ' ', 1).replace('Z', '', 1)
        # Insert article into database
        sql = "INSERT INTO DueDiligenceUI_trainingmodel (ArticleText, TrainingDate, SearchModel_id, IsTrained) " \
              "VALUES(?,'" + articleDateTime + "',0,0) "
        print(sql)
        cursorObj.execute(sql, [content])
        conn.commit()
        # print('---------------------------------Article Body---------------------------------')
        # print(content + os.linesep)
        # printn('---------------------------------Tokens---------------------------------')
        #tokens = unique_list(clean_article(content))
        tokens = clean_article(content)
        token_sentence = " ".join(tokens)
        # print(token_sentence)
        X.append(token_sentence)
    cursorObj.close()
    conn.close()
    return X

print('#######################################################################################')
print('Select an operation : ')
print('1. Train the Model')
print('2. Feedback to the Model')
print('#######################################################################################')
input = input('Enter the number : ')
if input == '1':
    X, Y, is_split_data = fetch_data_CSV()
elif input == '2':
    X, Y, is_split_data = fetch_data_db()
else:
    print('Please select a valid option')
    sys.exit()

train_sequential_model(X,Y, is_split_data)