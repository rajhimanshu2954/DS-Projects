#Importing Libs
import pandas as pd
import numpy as np
from wordcloud import WordCloud,ImageColorGenerator
import nltk
nltk.download("stopwords")
from nltk import SnowballStemmer
import re
import matplotlib.pyplot as plt
from PIL import Image
import requests
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
import scipy
from sklearn import preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import ensemble
import pandas, numpy
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import keras
#import keras.preprocessing.text as kpt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


## Loadig the file

def loadFile(file):
    data_raw = pd.read_csv(file)
    data_raw.head()
    return data_raw    

## Data Preprocessing    
    
def dataPrep(data):
    print(data.describe())
    print(data.dtypes)
    print(data.isnull().sum())                                                                                # There are 4733 empty values in tweet_location and 4820 empty value in user_timezone
    data['word_count'] = data['text'].apply(lambda x: len(str(x).split(" ")))                                 # Calculating No of words
    data['char_count'] = data['text'].str.len()                                                               # Calculating no of charcter including spaces
    stopwords = nltk.corpus.stopwords.words('english')
    newFile = open(r"D:\Test\Eli_Lilly_Assesment\NLP_Case_Study\MyStopwords.txt", 'r')
    myStopwords = newFile.read()
    myStopwords = myStopwords.lower().split('\n')
    stopwords.extend(myStopwords)
    data['stopwords_count'] = data['text'].apply(
        lambda x: len([x for x in x.split() if x in stopwords]))                                              # Calcuating No of stopwords
    data['hastags_count'] = data['text'].apply(
        lambda x: len([x for x in x.split() if x.startswith('#')]))                                           # Calcualting Hastags and Special Chahracters
    data['numerics'] = data['text'].apply(
        lambda x: len([x for x in x.split() if x.isdigit()]))                                                 # Calculating numeric digits 
    data['cleanedText'] = data['text'].str.lower()                                                            # Converting to lowercase
    data['cleanedText'] = data['cleanedText'].str.replace('rt', '')                                           #Replace rt indicating that was a retweet
    data['cleanedText'] = data['cleanedText'].replace(r'@\w+', '', regex=True)                                #Replace occurences of mentioning @UserNames
    data['cleanedText'] = data['cleanedText'].replace(r'http\S+', '', regex=True)                             #Replace http links contained in the tweet
    data['cleanedText'] = data['cleanedText'].replace(r'www.[^ ]+', '', regex=True)                           #Replace www links contained in the tweet
    data['cleanedText'] = data['cleanedText'].replace(r'[0-9]+', '', regex=True)                              #remove numbers
    data['cleanedText'] = data['cleanedText'].replace(r'[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~]', '', regex=True)   #replace special characters and puntuation marks
    stemmer = SnowballStemmer("english")
    # Removing stopwords and stemming the texts
    data['cleanedText'] = data['cleanedText'].apply(lambda x: " ".join([stemmer.stem(i) for i in x.split() if i not in (stopwords)]))
    return data

'''
Furthur to this we can also work on detecting and replacing elongated words
and handling negations.
'''

## Extration of certain features foranalyzing the tweets and contents

def featureEngg(data):
    # Extracting date-timefeatures
    data['tweet_created']   = pd.to_datetime(data['tweet_created'])
    data['hour']   = pd.DatetimeIndex(data['tweet_created']).hour
    data['month']  = pd.DatetimeIndex(data['tweet_created']).month
    data['day']    = pd.DatetimeIndex(data['tweet_created']).day
    data['Year']    = pd.DatetimeIndex(data['tweet_created']).year
    data['month_f']  = data['month'].map({1:"JAN",2:"FEB",3:"MAR",
                                        4:"APR",5:"MAY",6:"JUN",
                                        7:"JUL",8:"AUG",9:"SEP", 
                                        10:"Oct"})
    # Find out what's trending Hashtags(#)
    data['hashtag']  = data['text'].str.findall(r'#.*?(?=\s|$)')
    data['hashtag'] = data['hashtag'].apply(lambda x: str(x).replace('[','').replace(']',''))  
    # Extracting twitter account references
    data['accounts'] = data['text'].str.findall(r'@.*?(?=\s|$)')
    data['accounts'] = data['accounts'].apply(lambda x: str(x).replace('[','').replace(']',''))
    # Finding which is are-tweet
    data['is_retweted'] = data['text'].str.startswith('RT')
    return data

# Drawing the wordcloud for different scenario
    
def visualsWordCloud(words, titleText):
    mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))
    image_colors = ImageColorGenerator(mask)
    wc = WordCloud(background_color='black', height=1500, width=4000,mask=mask).generate(words)
    plt.figure(figsize=(10,20))
    plt.imshow(wc.recolor(color_func=image_colors),interpolation="hamming")
    plt.title(titleText)
    plt.axis('off')
    plt.show()
    
# Funcion for barplots 
    
def visualbarPlot(words, titleText):
    freq_positive = nltk.FreqDist(words)
    df_positive = pd.DataFrame({'Hashtags':list(freq_positive.keys()),'Count':list(freq_positive.values())})
    df_positive_plot = df_positive.nlargest(20,columns='Count')
    sns.barplot(data=df_positive_plot,y='Hashtags',x='Count').set_title(titleText)
    sns.despine()
    plt.show()        

# Function for plotting basic plots

def basicPlots(modData):
    # Seeing the count of realted classes
    sns.catplot(x="sentiment", kind="count", palette="ch:.25", data=modData)
    plt.show()
    
    # Seeing the variation of tweets on daily basis
    sns.catplot(x="day", hue="sentiment", kind="count", palette="ch:.25", data=modData)
    plt.show()
    
    ax = sns.distplot(modData['day'], kde= False)
    ax.set_title('Variation of Tweets every day')
    ax.set_ylabel('Count')
    plt.show()
    
    ax = sns.distplot(modData['hour'], kde= False)
    ax.set_title('Variation of Tweets every hour')
    ax.set_ylabel('Count')
    plt.show()
    
# Function for exploring the data basic plots    
    
def dataExploration(data):
    basicPlots(data)
    positive_words = ' '.join(text for text in data['cleanedText'][data['sentiment']=="positive"])
    negative_words = ' '.join(text for text in data['cleanedText'][data['sentiment']=="negative"])
    neutral_words = ' '.join(text for text in data['cleanedText'][data['sentiment']=="neutral"])
    # WordCloud for top positive tweets
    visualsWordCloud(positive_words, "Top Positive Words Wordcloud")
    # Wordcloud for top neutral tweets
    visualsWordCloud(neutral_words, "Top Neutral Words Wordcloud")
    # Wordcloud for top negative tweets
    visualsWordCloud(negative_words, "Top Negative Words Wordcloud")
    # Let's see what are the popular hastags associated with positive tweets
    positive_hastags = ' '.join(text for text in data['hashtag'][data['sentiment']=="positive"])
    visualsWordCloud(positive_hastags, "Popular hastags associated with positive tweets")
    # And now those who are associated with negative tweets
    negative_hastags = ' '.join(text for text in data['hashtag'][data['sentiment']=="negative"])
    visualsWordCloud(negative_hastags, "Popular hastags associated with negative tweets")
    # PLotting top 20 positive hashtags
    positive_hastags = positive_hastags.replace('#', '')
    positive_hastags = positive_hastags.replace('.', '')
    positive_hastags = positive_hastags.replace("'", '').split()
    visualbarPlot(positive_hastags, "Top 20 Positive Hastags")
    # PLotting top 20 negative hashtags
    negative_hastags = negative_hastags.replace('#', '')
    negative_hastags = negative_hastags.replace('.', '')
    negative_hastags = negative_hastags.replace("'", '').split()
    visualbarPlot(negative_hastags, "Top 20 Negative Hastags")
    
    # Let's see what are the popular accounts associated with positive tweets
    positive_accounts = ' '.join(text for text in data['accounts'][data['sentiment']=="positive"])
    visualsWordCloud(positive_accounts, "Popular accounts associated with positive tweets")
    # And now those accounts who are associated with negative tweets
    negative_accounts = ' '.join(text for text in data['accounts'][data['sentiment']=="negative"])
    visualsWordCloud(negative_accounts, "Popular accounts associated with negative tweets")


## Preparing Data for modelling
    
def dataPrepModel(data):
    print("Preaparing the data and creating new features for the model......")
    data['hashtag'] = data['hashtag'].apply(lambda x: re.sub('[^A-Za-z0-9]+', '', x))
    data['accounts'] = data['accounts'].apply(lambda x: re.sub('[^A-Za-z0-9]+', '', x))
    data['ctext'] = data[['cleanedText', 'name', 'hashtag', 'accounts']].agg(' '.join, axis=1)
    data['sentiment'] = data["sentiment"].map({"neutral" : 1,"positive" : 2,
                                          "negative" : 3})
    data['is_retweted'] = data["is_retweted"].map({True : 1,False : 2})  
    
    #dependent and independent variables
    predictors = ['ctext', 'word_count', 'hastags_count', 'is_retweted']
    target     = "sentiment"
    
    print("Splitting the data into train and test.......")
    #splitting the dataset
    train,test = train_test_split(data,test_size = .25,
                                  stratify = data[["sentiment"]],
                                  random_state  = 123)
    train_X = train[predictors]
    train_Y = train[target]
    test_X  = test[predictors]
    test_Y  = test[target]
    #label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    train_Y = encoder.fit_transform(train_Y)
    test_Y = encoder.fit_transform(test_Y)
    return train_X, train_Y, test_X, test_Y

## Preparing Data Features for modelling
    
def dataFeatureModel(train_X, test_X, feature):
    print("Selecting the feature model........")
    # For Count Vector Approach
    if(feature == 'tfidf'):
        print("Using TF-IDF approach......")
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=2500)
        tfidf_vect.fit(train_X['ctext'])
        xtrain_count =  tfidf_vect.transform(train_X['ctext'])
        xvalid_count =  tfidf_vect.transform(test_X['ctext'])
        # Adding created features into the list of term attributes
        train_df = pd.DataFrame(xtrain_count.toarray(), columns=tfidf_vect.get_feature_names())
        train_df.index = train_X.index
        trainData = pd.concat([train_df, train_X[['word_count', 'word_count', 'hastags_count', 'is_retweted']]],
                              axis=1)
        trainDataX = scipy.sparse.csr_matrix(trainData.values)  # Training X Sparse Matrix
        
        # Adding created features into the list of term attributes
        valid_df = pd.DataFrame(xvalid_count.toarray(), columns=tfidf_vect.get_feature_names())
        valid_df.index = test_X.index
        validData = pd.concat([valid_df, test_X[['word_count', 'word_count', 'hastags_count', 'is_retweted']]],
                              axis=1)
        testDataX = scipy.sparse.csr_matrix(validData.values)
        
    elif (feature == 'tfidf2'):
        print("Using TF-IDF n-gram approach......")
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=1500)
        tfidf_vect_ngram.fit(train_X['ctext'])
        xtrain_count =  tfidf_vect_ngram.transform(train_X['ctext'])
        xvalid_count =  tfidf_vect_ngram.transform(test_X['ctext'])
        train_df = pd.DataFrame(xtrain_count.toarray(), columns=tfidf_vect_ngram.get_feature_names())
        train_df.index = train_X.index
        trainData = pd.concat([train_df, train_X[['word_count', 'word_count', 'hastags_count', 'is_retweted']]],
                              axis=1)
        trainDataX = scipy.sparse.csr_matrix(trainData.values)  # Training X Sparse Matrix
        
        # Adding created features into the list of term attributes
        valid_df = pd.DataFrame(xvalid_count.toarray(), columns=tfidf_vect_ngram.get_feature_names())
        valid_df.index = test_X.index
        validData = pd.concat([valid_df, test_X[['word_count', 'word_count', 'hastags_count', 'is_retweted']]],
                              axis=1)
        testDataX = scipy.sparse.csr_matrix(validData.values)  # Training X Sparse Matrix
        
    else:
        # create a count vectorizer object
        print("Using count-vector approach......")
        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        count_vect.fit(train_X['ctext'])
        # transform the training and validation data using count vectorizer object
        xtrain_count =  count_vect.transform(train_X['ctext'])
        xvalid_count =  count_vect.transform(test_X['ctext'])
        # Adding created features into the list of term attributes
        train_df = pd.DataFrame(xtrain_count.toarray(), columns=count_vect.get_feature_names())
        train_df.index = train_X.index
        trainData = pd.concat([train_df, train_X[['word_count', 'word_count', 'hastags_count', 'is_retweted']]],
                              axis=1)
        trainDataX = scipy.sparse.csr_matrix(trainData.values)  # Training X Sparse Matrix
        
        # Adding created features into the list of term attributes
        valid_df = pd.DataFrame(xvalid_count.toarray(), columns=count_vect.get_feature_names())
        valid_df.index = test_X.index
        validData = pd.concat([valid_df, test_X[['word_count', 'word_count', 'hastags_count', 'is_retweted']]],
                              axis=1)
        testDataX = scipy.sparse.csr_matrix(validData.values)  # Training X Sparse Matrix
    return trainDataX, testDataX

def train_model(classifier, train_X, train_Y, test_X, test_Y):
    # fit the training dataset
    print("Training the model with selected classifier......")
    classifier.fit(train_X, train_Y)
    # predict the labels
    predictions = classifier.predict(test_X)
    print(metrics.classification_report(predictions, test_Y, digits=3))
    return metrics.accuracy_score(predictions, test_Y)

def callModel(train_X, train_Y, test_X, test_Y, classifier):
    
    if(classifier== 'NB'):
        #Naive Bayes
        accuracy = train_model(naive_bayes.MultinomialNB(), train_X, train_Y, test_X, test_Y)
        print("Naive Bayes Accuracy: ", accuracy)
        
    
    if(classifier== 'SVM'):
        # Support Vector
        accuracy = train_model(svm.SVC(), train_X, train_Y, test_X,test_Y )
        print("SVM Accuracy: ", accuracy)
    
    if(classifier== 'RF'):
        # Random Forest
        accuracy = train_model(ensemble.RandomForestClassifier(), train_X, train_Y, test_X,test_Y)
        print("RF Accuracy: ", accuracy)
        
    if(classifier== 'LOG'): 
        # Linear Classifier 
        accuracy = train_model(linear_model.LogisticRegression(C=2.7825594022071245, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='warn', n_jobs=None, penalty='l2', random_state=None,
          solver='warn', tol=0.0001, verbose=0, warm_start=False), train_X, train_Y, test_X, test_Y)
        print("Logistic Accuracy: ", accuracy)

#Create a Neural Network
#Create the model
def train(trainDataX, train_Y, features, shuffle, drop, layer1, layer2, epoch, lr, epsilon, validation):
    model_nn = Sequential()
    model_nn.add(Dense(layer1, input_shape=(features,), activation='relu'))
    model_nn.add(Dropout(drop))
    model_nn.add(Dense(layer2, activation='sigmoid'))
    model_nn.add(Dropout(drop))
    model_nn.add(Dense(3, activation='softmax'))
    
    optimizer = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=epsilon, decay=0.0, amsgrad=False)
    model_nn.compile(loss='sparse_categorical_crossentropy',
                 optimizer=optimizer,
                 metrics=['accuracy'])
    model_nn.fit(trainDataX, train_Y,
                 batch_size=32,
                 epochs=epoch,
                 verbose=1,
                 validation_split=validation,
                 shuffle=shuffle)
    return model_nn

def test(X_test, model_nn):
    prediction = model_nn.predict_classes(X_test)
    return prediction

def model1(trainDataX, train_Y):   
    features = 2504
    shuffle = True
    drop = 0.5
    layer1 = 512
    layer2 = 256
    epoch = 10
    lr = 0.001
    epsilon = None
    validation = 0.2
    model = train(trainDataX, train_Y, features, shuffle, drop, layer1, layer2, epoch, lr, epsilon, validation)
    return model;

## Main Function
    
def main():    
    path = r"D:\Test\Eli_Lilly_Assesment\NLP_Case_Study\Data.csv"
    data = loadFile(path)
    prepData = dataPrep(data)
    modData = featureEngg(prepData)
    dataExploration(modData)
    train_X, train_Y, test_X, test_Y = dataPrepModel(modData)
    trainDataX, testDataX = dataFeatureModel(train_X, test_X, 'tfidf')
    callModel(trainDataX,train_Y,testDataX,test_Y, 'LOG')
    '''
    pipe = Pipeline([('classifier', LogisticRegression())])
    search_space = [{'classifier': [LogisticRegression()],
                 'classifier__penalty': ['l1', 'l2'],
                 'classifier__C': np.logspace(0, 4, 10)},
                {'classifier': [svm.SVC()],
                 'classifier__C': [0.1, 1, 10, 100, 1000],
                 'classifier__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                 'classifier__kernel': ['rbf']}]
    
    clf = GridSearchCV(pipe, search_space, cv=5, n_jobs = -1, verbose = 2)
    best_model = clf.fit(trainDataX, train_Y)
    print(best_model.best_estimator_.get_params()['classifier'])
    predictions = best_model.predict(testDataX)
    '''
    nnmodel= model1(trainDataX, train_Y)
    predictions = test(testDataX, nnmodel)
    metrics.accuracy_score(predictions, test_Y)
    
## Main call    
if __name__ == "__main__":
    main()