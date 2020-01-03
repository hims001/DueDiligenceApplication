from eventregistry import *
from datetime import date, timedelta
from os import listdir
from os.path import isfile, join
import io, os, re, json
from os import path
# import pdfminer
# from pdfminer.converter import TextConverter
# from pdfminer.pdfinterp import PDFPageInterpreter
# from pdfminer.pdfinterp import PDFResourceManager
# from pdfminer.pdfpage import PDFPage
import shutil
import random
from nltk.corpus import stopwords
import string
import sqlite3
import numpy as np
import DueDiligenceUI.BusinessLogic.ConfigurationManager as cfg
from nltk.tokenize import sent_tokenize
from operator import itemgetter
from sklearn.preprocessing import binarize
from difflib import SequenceMatcher

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    from keras.models import load_model
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras import backend as K
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class SearchProcess:
    #print('CONFIG VALUE :' + cfg.read_config('sql_db_path'))
    model_path = cfg.read_config('model_path')
    sql_db_path = cfg.read_config('sql_db_path')
    max_length = int(cfg.read_config('max_length'))
    sentence_buffer = int(cfg.read_config('sentence_buffer'))
    news_fetch_count = int(cfg.read_config('news_fetch_count'))

    def extract_text_from_pdf(self, pdf_path, laparams):
        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
        converter = TextConverter(resource_manager, fake_file_handle, laparams=laparams)
        page_interpreter = PDFPageInterpreter(resource_manager, converter)

        with open(pdf_path, 'rb') as fh:
            for page in PDFPage.get_pages(fh,
                                          caching=True,
                                          check_extractable=True):
                page_interpreter.process_page(page)

            text = fake_file_handle.getvalue()

        # close open handles
        converter.close()
        fake_file_handle.close()

        if text:
            return text

    # turn a doc into clean tokens
    def clean_article(self, article_content):
        tokens = [word.replace('\\n', ' ').replace('\n', ' ').replace('  ', ' ') for word in article_content.split()]
        # print(tokens)
        # split again in order to accomodate replaced spaces into tokens by white space
        article_content = " ".join(tokens)
        tokens = article_content.split()
        # print(tokens)
        # remove punctuation from each token except Full Stop(.)
        table = str.maketrans('', '', '!"#$%&\'()*+,/:;<=>?@[\\]^_`{|}~')
        tokens = [w.translate(table) for w in tokens]
        # print(tokens)
        # remove remaining tokens that are not alphabetic
        tokens = [word.strip() for word in tokens if word.strip().isalpha() or '.' in word or '-' in word]
        # print(tokens)
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        # print(tokens)
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1 or '.' in word or '-' in word]
        # print(tokens)
        return tokens

    def unique_list(self, list):
        uniquelist = []
        [uniquelist.append(word) for word in list if word not in uniquelist]
        return uniquelist

    def process_request_pdf(self, entityname):
        print(entityname)
        # Perform layout analysis for all text
        laparams = pdfminer.layout.LAParams()
        setattr(laparams, 'all_texts', True)

        # Fetch all articles from the folder
        dirpath = 'DueDiligenceUI/Articles_Repo'
        archived_dirpath = 'DueDiligenceUI/Articles_Repo_Archives'
        articles = [f for f in listdir(dirpath) if isfile(join(dirpath, f))]

        print('Reading PDF files...')
        for article in articles:
            fullpath = join(dirpath, article)
            print(fullpath + os.linesep)
            content = self.extract_text_from_pdf(fullpath, laparams)
            # print(content + os.linesep)
            print('Tokens:')
            tokens = self.unique_list(self.clean_article(content))
            print(tokens)
            print(len(tokens))
            print(" ".join(tokens))

            # Move a file from the directory d1 to d2
            randompath = join(archived_dirpath, entityname, str(random.randint(0, 1000000)))
            os.makedirs(randompath)
            shutil.copy(fullpath, join(randompath, article))
            # shutil.move(fullpath, join(randompath, article))

            # Clear repos directory
            # filelist = [ f for f in os.listdir(dirpath) ]
            # for f in filelist:
            #     os.remove(join(dirpath, f))

    def sortSecond(self, val):
        return val[1];

    def process_request(self, entityname, model_id):
        table = str.maketrans('', '', string.punctuation.replace('&',''))
        entityname = entityname.translate(table)
        print('Entity being searched : ' + entityname)
        entityList = []
        entityList.append(entityname)
        entityArr = re.split('\W+', entityname)
        if len(entityArr) == 2:
            entityList.append(entityArr[1] + " " + entityArr[0])

        print(entityList)
        er = EventRegistry(apiKey="f4a005ab-a24f-487e-bff4-f39b1b2ba6c2")
        cq = ComplexArticleQuery(
            query=CombinedQuery.AND(
                [
                    BaseQuery(keyword=QueryItems.OR(entityList),
                              #sourceLocationUri=er.getLocationUri("United States"),
                              lang="eng",
                              dateStart=date.today() - timedelta(days=365),
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
                                                 count=self.news_fetch_count,
                                                 sortBy="date", sortByAsc=False,
                                                 returnInfo=ReturnInfo()))
        res = er.execQuery(q)

        #sql_db_path = cfg.read_config('sql_db_path')
        con = sqlite3.connect(self.sql_db_path)
        cursorObj = con.cursor()

        #Remove similar redundant news articles
        # article_list = []
        # match_list = []
        # for article1 in res['articles']['results']:
        #     similarity_flag = False
        #     for article2 in res['articles']['results']:
        #         val = SequenceMatcher(a=article1['body'], b=article2['body']).ratio()
        #         match_list.append(val)
        #         if article1 != article2 and val > 0.8:
        #             similarity_flag = True
        #             print(val)
        #     print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        #     if not similarity_flag:
        #         article_list.append(article1)
        #
        # print(match_list)
        articles=[]
        print('Number of articles found: ' + str(len(res['articles']['results'])))
        for article in res['articles']['results']:
            content = article["body"]
            #print(content.encode("utf-8"))
            title = article["title"]
            url = article["url"]
            articleDateTime = article["dateTime"].replace('T', ' ', 1).replace('Z', '', 1)
            # Replace name of the entity with word ENTITY
            if entityname.lower() in content.lower():
                re_replace = re.compile(re.escape(entityname), re.IGNORECASE)
                content = re_replace.sub('ENTITY', content)
                replaced_entityname = entityname
            if len(entityArr) == 2:
                entity_temp = entityArr[1] + " " + entityArr[0]
                if entity_temp.lower() in content.lower():
                    re_replace = re.compile(re.escape(entity_temp), re.IGNORECASE)
                    content = re_replace.sub('ENTITY', content)
                    replaced_entityname = entity_temp
            if 'ENTITY' in content:
                lst = [title, content, articleDateTime, url]
                articles.append(lst)

        # obtain at most 10 newest articles or blog posts
        X = []
        print('remaining length : ' + str(len(articles)))
        for article in articles:
            content = article[1]
            articleDateTime = article[2]
            url = article[3]
            #print(url)
            # Insert article into database
            sql = "INSERT INTO DueDiligenceUI_trainingmodel (ArticleText, TrainingDate, SearchModel_id, IsTrained, Url) " \
                  "VALUES(?,'" + articleDateTime + "'," + str(model_id) + ",0,'" + url + "') "
            #print(sql)
            cursorObj.execute(sql, [content])
            con.commit()

            #print('---------------------------------Article Body---------------------------------')
            # print(content.encode("utf-8"))
            #print('---------------------------------Tokens---------------------------------')
            #tokens = self.unique_list(self.clean_article(content))
            tokens = self.clean_article(content)

            #Slice out max length characters from the article
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]

            # Grab Before and After 2 sentences of sentence with ENTITY word
            sentences = sent_tokenize(" ".join(tokens))
            indices = [idx for idx, sent in enumerate(sentences) if 'ENTITY' in sent]
            #print(indices)
            extended_indices = []
            for i, sentence in enumerate(sentences):
                extended_indices.extend(list(set([i for index in indices if abs(index - i) <= self.sentence_buffer])))
            #print(extended_indices)
            #print(len(sentences))

            desired_list = list(itemgetter(*extended_indices)(sentences))

            token_sentence = " ".join(desired_list)
            #print(token_sentence)
            X.append(token_sentence)

        # Before prediction
        K.clear_session()

        if path.exists(self.model_path + 'trained_model.h5'):
            if len(X) > 0:
                # Load the model
                model = load_model(self.model_path + 'trained_model.h5')

                # prepare tokenizer
                t = Tokenizer()
                t.fit_on_texts(X)

                # integer encode the documents
                encoded_docs = t.texts_to_sequences(X)
                #print(encoded_docs)

                # pad documents to a max length of words
                #max_length = max([len(word.split()) for word in X])
                padded_docs = pad_sequences(encoded_docs, maxlen=self.max_length, padding='post')
                #print(padded_docs)

                # Predict on searched articles
                probabilities = model.predict(x=padded_docs, batch_size=5, verbose=2)
                #classes = model.predict_classes(x=padded_docs, batch_size=5, verbose=2)
                classes = binarize(probabilities, 0.6)
                #print(classes)
                print(probabilities)

                prediction=[]
                for idx in range(len(classes)):
                    prediction.append((int(classes[idx][0]), round(probabilities[idx][0]*100, 2)))

                # Sort in descending order of probability
                prediction.sort(key = self.sortSecond, reverse = True)

                print(prediction)
                # After prediction
                K.clear_session()

                # for i in range(len(prediction)):
                #     if prediction[i][0] > 0.5:
                #max_prediction = max(prediction)[1];
                #print('MAX Prediction: ' + str(max_prediction))
                #i, j = np.where(prediction == max_prediction)
                #print('MAX Prediction Index: ' + str(i[0]))
                #print('Target article: ' + str(res['articles']['results'][i[0]]))
                #if max_prediction > 0.6:

                # Replace ENTITY word with original entity name
                for article in articles:
                    re_replace = re.compile(re.escape('ENTITY'), re.IGNORECASE)
                    article[1] = re_replace.sub(replaced_entityname, article[1])
                #print(articles)
                return [1, prediction, articles]
                #return [0, round(max_prediction*100, 2)]
            else:
                return [-1]
        else:
            raise Exception('Model is not yet ready to predict')