from DueDiligenceUI.models import TrainingModel
from nltk.corpus import stopwords
import string

class TrainingProcess:
    # turn a doc into clean tokens
    def clean_article(self, article_content):
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

    def unique_list(self, list):
        uniquelist = []
        [uniquelist.append(word) for word in list if word not in uniquelist]
        return uniquelist

    def process_request(self):
        trainingArticles = TrainingModel.objects.filter(IsTrained=0);

        for article in trainingArticles:
            content = article.ArticleText
            # print(content + os.linesep)
            #print('Tokens:')
            tokens = self.unique_list(self.clean_article(content))
            #print(tokens)


TrainingProcess.process_request()