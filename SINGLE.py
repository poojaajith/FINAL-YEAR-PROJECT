from tweetscrape.search_tweets import TweetScrapperSearch
import pandas as pd
from pandas import DataFrame
import twitter
import numpy as np
import re
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from numpy import loadtxt
from xgboost import XGBClassifier
# plotting
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE


#------------------------------------PERSONALITY PREDICTION--------------------------------------

check="INFJ"
data = pd.read_csv("mbti_1.csv")
#print(data.head(10))
def get_types(row):
    t = row['type']

    I = 0;
    N = 0
    T = 0;
    J = 0

    if t[0] == 'I':
        I = 1
    elif t[0] == 'E':
        I = 0
    else:
        print('I-E incorrect')

    if t[1] == 'N':
        N = 1
    elif t[1] == 'S':
        N = 0
    else:
        print('N-S incorrect')

    if t[2] == 'T':
        T = 1
    elif t[2] == 'F':
        T = 0
    else:
        print('T-F incorrect')

    if t[3] == 'J':
        J = 1
    elif t[3] == 'P':
        J = 0
    else:
        print('J-P incorrect')
    return pd.Series({'IE': I, 'NS': N, 'TF': T, 'JP': J})


data = data.join(data.apply(lambda row: get_types(row), axis=1))
#print(data.head(5))

#print ("Introversion (I) /  Extroversion (E):\t", data['IE'].value_counts()[0], " / ", data['IE'].value_counts()[1])
#print ("Intuition (N) – Sensing (S):\t\t", data['NS'].value_counts()[0], " / ", data['NS'].value_counts()[1])
#print ("Thinking (T) – Feeling (F):\t\t", data['TF'].value_counts()[0], " / ", data['TF'].value_counts()[1])
#print ("Judging (J) – Perceiving (P):\t\t", data['JP'].value_counts()[0], " / ", data['JP'].value_counts()[1])

b_Pers = {'I': 0, 'E': 1, 'N': 0, 'S': 1, 'F': 0, 'T': 1, 'J': 0, 'P': 1}
b_Pers_list = [{0: 'I', 1: 'E'}, {0: 'N', 1: 'S'}, {0: 'F', 1: 'T'}, {0: 'J', 1: 'P'}]


def translate_personality(personality):
    # transform mbti to binary vector

    return [b_Pers[l] for l in personality]


def translate_back(personality):
    # transform binary vector to mbti personality

    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s


# Check ...
d = data.head(4)
list_personality_bin = np.array([translate_personality(p) for p in d.type])
#print("Binarize MBTI list: \n%s" % list_personality_bin)

unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
                    'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']

unique_type_list = [x.lower() for x in unique_type_list]

# Lemmatize
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()

# Cache the stop words for speed
cachedStopWords = stopwords.words("english")


def pre_process_data(data, remove_stop_words=True, remove_mbti_profiles=True):
    list_personality = []
    list_posts = []
    len_data = len(data)
    i = 0

    for row in data.iterrows():
        i += 1
        #if (i % 500 == 0 or i == 1 or i == len_data):
            #print("%s of %s rows" % (i, len_data))

        ##### Remove and clean comments
        posts = row[1].posts
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)
        temp = re.sub("[^a-zA-Z]", " ", temp)
        temp = re.sub(' +', ' ', temp).lower()
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])

        if remove_mbti_profiles:
            for t in unique_type_list:
                temp = temp.replace(t, "")

        type_labelized = translate_personality(row[1].type)
        list_personality.append(type_labelized)
        list_posts.append(temp)

    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality
list_posts, list_personality  = pre_process_data(data, remove_stop_words=True)
print("---------------Training Data Set-----Binarized Personality List ----------------\n",list_personality)



# Posts to a matrix of token counts
cntizer = CountVectorizer(analyzer="word",
                             max_features=1500,
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_df=0.7,
                             min_df=0.1)

# Learn the vocabulary dictionary and return term-document matrix
#print("CountVectorizer...")
#test2=""
#with open('in.csv', encoding='ISO-8859-2') as f:
 #   for text in f.read().split("\n"):
#	    test2+=str((word_tokenize(text)))
X_cnt = cntizer.fit_transform(list_posts)

# Transform the count matrix to a normalized tf or tf-idf representation
tfizer = TfidfTransformer()

#print("Tf-idf...")
# Learn the idf vector (fit) and transform a count matrix to a tf-idf representation
X_tfidf =  tfizer.fit_transform(X_cnt).toarray()
#feature_names = list(enumerate(cntizer.get_feature_names()))
#print(feature_names)




#----------------------------------------------------------------LOOP-----------------------------------------------
f=open("id.txt", "r")
f1=f.readlines()
r=[]
i=0
for x in f1:
    use = x.rstrip('\n')
    tweet_scrapper = TweetScrapperSearch(search_from_accounts=use, search_till_date="2020-03-10", search_since_date="2019-10-19", num_tweets=40, tweet_dump_path=use+'1.csv', tweet_dump_format='csv')
    tweet_count, tweet_id, tweet_time, dump_path = tweet_scrapper.get_search_tweets()
#print(tweet_scrapper.text())
#print("Extracted {0} tweets till {1} at {2}".format(tweet_count, tweet_time, dump_path))


    df1 = pd.read_csv(use+"1.csv") # read csv file and store it in a dataframe
#df.to_csv('hrdata_modified.csv')
    my_columns = DataFrame(df1["text"])
#add in.csv instead of x.txt
#my_columns.drop([0])
    my_columns.to_csv(use+".csv")


    my_posts=""
    with open(use+'.csv', encoding='ISO-8859-2') as f:
	    for text in f.read().split("\n"):
		    my_posts+=str((word_tokenize(text)))



#my_posts  = "Getting started with data science and applying machine learning has never been as simple as it is now. There are many free and paid online tutorials and courses out there to help you to get started. I’ve recently started to learn, play, and work on Data Science & Machine Learning on Kaggle.com. In this brief post, I’d like to share my experience with the Kaggle Python Docker image, which simplifies the Data Scientist’s life."
    mydata = pd.DataFrame(data={'type': ['INFJ'], 'posts': [my_posts]})

    my_posts, dummy  = pre_process_data(mydata, remove_stop_words=True)

    my_X_cnt = cntizer.transform(my_posts)
    my_X_tfidf =  tfizer.transform(my_X_cnt).toarray()

    param = {}
    param['n_estimators'] = 200
    param['max_depth'] = 2
    param['nthread'] = 8
    param['learning_rate'] = 0.2

    result = []
    type_indicators = [ "IE: Introversion (I) / Extroversion (E)", "NS: Intuition (N) – Sensing (S)",
                   "FT: Feeling (F) - Thinking (T)", "JP: Judging (J) – Perceiving (P)"  ]

# Let's train type indicator individually
    X = X_tfidf
    for l in range(len(type_indicators)):
        #print("l value",l)
        print("-------------------------------%s---------------------------------" % (type_indicators[l]))
        Y = list_personality[:, l]
        print("Training dataset-Binarized personality list for type indicator-------> ",Y)
    # split data into train and test sets
        seed = 7
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model on training data
        model = XGBClassifier(**param)
        model.fit(X_train, y_train)

    # make predictions for my  data
        y_pred = model.predict(my_X_tfidf)
        print("Predicted value-----> ",y_pred)
        result.append(y_pred[0])
    # print("* %s prediction: %s" % (type_indicators[l], y_pred))
    print("RESULT IS----",result)
    print("TRANSLATED RESULT IS------", translate_back(result))

#     x= translate_back(result)
#     score=0
#     for p in check:
#         if p in x:
#             score=score+10
#     print("MY SCORE----------",score)
#
#     r+=[(score,use)]
# print("SORTED TWITTER IDs BASED ON THE SCORE")
# def Sort_Tuple(tup):
#     # reverse = None (Sorts in Ascending order)
#     # key is set to sort using second element of
#     # sublist lambda has been used
#     return (sorted(tup, key=lambda x: x[0],reverse=True))
#
# p=Sort_Tuple(r)
# for x in p:
#     print(x[1])
#


#print(x[0])
