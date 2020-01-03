from numpy import asarray
from numpy import zeros
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from os import path
import pandas as pd
import sqlite3
from eventregistry import *
from datetime import date, timedelta, datetime
from nltk.corpus import stopwords
import string
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    from keras.models import load_model
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def train_sequential_model(X, Y):
    modelspath = 'DueDiligenceUI/Models/'
    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(X)
    vocab_size = len(t.word_index) + 1
    print('vocab_size' + str(vocab_size))
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(X)
    #print(encoded_docs)

    # pad documents to a max length of words
    max_length = max([len(word.split()) for word in X])
    print('max_length' + str(max_length))
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    #print(padded_docs)

    if path.exists(modelspath + 'trained_model.h5'):
        model = load_model(modelspath + 'trained_model.h5')
    else:
        # load the whole embedding into memory
        embeddings_index = dict()
        f = open(modelspath + 'glove.6B.100d.txt', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))

        # create a weight matrix for words in training docs
        embedding_matrix = zeros((vocab_size, 100))
        for word, i in t.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        # define model
        model = Sequential()
        e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
        model.add(e)
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        # compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    # summarize the model
    print(model.summary())

    # fit the model
    model.fit(padded_docs[:150], Y[:150], epochs=50, shuffle=True, verbose=1)

    # evaluate the model
    loss, accuracy = model.evaluate(padded_docs[30:], Y[30:], verbose=1)
    print('Accuracy: %f' % (accuracy * 100))

    # Save the model to disk
    model.save(modelspath + 'trained_model.h5')
    print("Model saved to disk.")

def fetch_data_db():
    conn = sqlite3.connect('C:\\Dev\\Work\\DueDiligence_Material\\DueDil_Final\\DueDiligenceRepo\\db.sqlite3')
    c = conn.cursor()
    myquery = ("SELECT ArticleText,Outcome FROM DueDiligenceUI_trainingmodel WHERE IsTrained=0 and Outcome IS NOT NULL")
    c.execute(myquery)
    result = c.fetchall()
    c.close()
    conn.close()
    return result

def fetch_data_CSV():
    training_data = pd.read_csv('C:\\Dev\\Work\\DueDiligence_Material\\DueDil_Final\\DueDiligenceRepo\\DueDiligenceUI\\Models\\TrainingData.csv', encoding='latin1')
    training_data = training_data.drop('Name', axis=1)

    conn = sqlite3.connect('C:\\Dev\\Work\\DueDiligence_Material\\DueDil_Final\\DueDiligenceRepo\\db.sqlite3')
    c = conn.cursor()
    cursorObj = conn.cursor()

    for ind in training_data.index:
        content = training_data["Data"][ind]
        outcome = training_data["Pattern"][ind]
        articleDateTime = datetime.today()
        # Insert article into database
        sql = "INSERT INTO DueDiligenceUI_trainingmodel (ArticleText, Outcome, TrainingDate, SearchModel_id, IsTrained) " \
              "VALUES(?,'" + str(outcome) + "','" + str(articleDateTime) + "',0,1) "
        #print(sql)
        cursorObj.execute(sql, [content])
        conn.commit()

    cursorObj.close()
    conn.close()
    return training_data['Data'], training_data['Pattern'];

    # turn a doc into clean tokens
def clean_article(article_content):
    # split into tokens by white space
    tokens = article_content.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

def unique_list(list):
    uniquelist = []
    [uniquelist.append(word) for word in list if word not in uniquelist]
    return uniquelist

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
                          sourceLocationUri=er.getLocationUri("United States"),
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

    conn = sqlite3.connect('C:\\Dev\\Work\\DueDiligence_Material\\DueDil_Final\\DueDiligenceRepo\\db.sqlite3')
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
        tokens = unique_list(clean_article(content))
        token_sentence = " ".join(tokens)
        # print(token_sentence)
        X.append(token_sentence)
    cursorObj.close()
    conn.close()
    return X

#Y = [1,1,1,1,1,0,0,0,0,0]
#X = fetch_data_eventregistry('Vijay Mallya')
#X = fetch_data_db()
X,Y = fetch_data_CSV()
#for row in lstRecords:
    #X.append(row[0])
    #Y.append(row[1])

train_sequential_model(X,Y)