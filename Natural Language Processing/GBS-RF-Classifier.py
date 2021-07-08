# -*- coding: utf-8 -*-
"""
Updated on Mon Aug 24 07:38:09 2019

@author: Himanshu_Raj
"""

###########################Loading Libs####################################
from pyforest import *
# Visual Libraries
from wordcloud import WordCloud
nltk.download("stopwords")
import nltk
from nltk import SnowballStemmer
nltk.download('wordnet')
import scipy
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
from skmultilearn.problem_transform import LabelPowerset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


###########################Reading Data####################################
def readData(path):
    os.chdir(path)
    cwd = os.getcwd()
    files = os.listdir(cwd + '\data\\')  # Get all the files in that directory
    print("Files present is  %s" % (files[0]))
    # load the dataset
    data = pd.read_csv(cwd + '\data\\' + files[0])
    data.isnull().sum()
    data.replace('', np.nan, inplace=True)
    data.replace(' ', np.nan, inplace=True)
    data.replace(' \n', np.nan, inplace=True)
    data.dropna(inplace=True)
    data.isnull().sum()
    finalData = data.copy()
    stopwords = nltk.corpus.stopwords.words('english')
    stemmer = SnowballStemmer("english")
    newFile = open(r'C:\Users\adcdevadmin\Documents\Projects\SSE_Txt_Classfication\MyStopwords.txt', 'r')
    myStopwords = newFile.read()
    myStopwords = myStopwords.split('\n')
    stopwords.extend(myStopwords)
    # Cleaning Data
    finalData['Occurrence_Description'] = finalData['Occurrence_Description'].astype(str)
    # Calculating No of words
    finalData['word_count'] = finalData['Occurrence_Description'].apply(lambda x: len(str(x).split(" ")))
    # Calculating no of charcter including spaces
    finalData['char_count'] = finalData['Occurrence_Description'].str.len()
    # Calcuating No of stopwords
    finalData['stopwords_count'] = finalData['Occurrence_Description'].apply(
        lambda x: len([x for x in x.split() if x in stopwords]))
    # Calcualting Hastags and Special Chahracters
    finalData['hastags_count'] = finalData['Occurrence_Description'].apply(
        lambda x: len([x for x in x.split() if x.startswith('#')]))
    # Calculating numeric digits 
    finalData['numerics'] = finalData['Occurrence_Description'].apply(
        lambda x: len([x for x in x.split() if x.isdigit()]))
    # Converting all the words to lower
    finalData['Occurrence_Description'] = finalData['Occurrence_Description'].apply(
        lambda x: " ".join(x.lower() for x in x.split()))
    # Cleaning data in Description column
    finalData['cleanedDesc'] = finalData['Occurrence_Description'].apply(lambda x: " ".join(
        [stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stopwords]).lower())
    # Removing words less than 3 characters
    finalData['cleanedDesc'] = finalData['cleanedDesc'].apply(lambda x: re.sub(r'\b\w{1,2}\b', '', x))
    finalData['Occurrence_Title'] = finalData['Occurrence_Title'].astype(str)
    finalData['word_count_2'] = finalData['Occurrence_Title'].apply(lambda x: len(str(x).split(" ")))
    # Calculating no of charcter including spaces
    finalData['char_count_2'] = finalData['Occurrence_Title'].str.len()
    # Calcuating No of stopwords
    finalData['stopwords_count_2'] = finalData['Occurrence_Title'].apply(
        lambda x: len([x for x in x.split() if x in stopwords]))
    # Calcualting Hastags and Special Chahracters
    finalData['hastags_count_2'] = finalData['Occurrence_Title'].apply(
        lambda x: len([x for x in x.split() if x.startswith('#')]))
    # Calculating numeric digits 
    finalData['numerics_2'] = finalData['Occurrence_Title'].apply(
        lambda x: len([x for x in x.split() if x.isdigit()]))
    # Converting all the words to lower
    finalData['Occurrence_Title'] = finalData['Occurrence_Title'].apply(
        lambda x: " ".join(x.lower() for x in x.split()))
    finalData['cleanedTitle'] = finalData['Occurrence_Title'].apply(lambda x: " ".join(
        [stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stopwords]).lower())
    
    # Top 50 common words from Occurance Description
    freq = pd.Series(' '.join(finalData['cleanedDesc']).split()).value_counts()[:50]
    print(freq)
    freq.index
    # Generate a word cloud image
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(
        str(finalData['cleanedDesc']))
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    
    # Top 50 common words from Occurance Title
    freq = pd.Series(' '.join(finalData['cleanedTitle']).split()).value_counts()[:50]
    print(freq)
    freq.index
    # Generate a word cloud image
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(
        str(finalData['cleanedTitle']))
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    return finalData

###########################Prepping Model####################################
def modelPrep(finalData):
    finalData.replace(np.NaN, 0, inplace=True)
    traincols = ['cleanedDesc', 'word_count', 'char_count', 'stopwords_count', 'hastags_count', 
                 'numerics', 'Occurrence_Title', 'word_count_2', 'char_count_2', 
                 'stopwords_count_2', 'hastags_count_2','numerics_2', 'cleanedTitle', 'Occurrence_Type_Code']
    
    targetcols = ['10048', '10052', '10054', '10055', '10176', '10213', '10214', '10215',
                  '10235', '10240', '10242', '10252', '10262', '10268', '10269', '10270',
                  '10271', '10272', '10278', '10279', '10287', '10294', '10295', '10321',
                  '10329', '10353', '10375', '10376', '10398', '10423', '10429', '10449',
                  '10459', '10470', '10472', '10507', '10512', '10521', '10525', '10528',
                  '10538', '10557', '10558', '10571', '10580', '10599', '10622', '10644',
                  '10656', '10666', '10701', '10713', '10727', '10741', '10763', '10774',
                  '10775', '10789', '10797', '10813', '10835', '10854', '10874', '10882',
                  '10891', '10911', '10924', '10931', '10940', '10946', '10947', '10982',
                  '11009', '11308', '11309', '11314', '9672', '9673', '9702', '9733',
                  '9757', '9783', '9809', '9917', '9918']
    # Regularizing the values
    finalData[targetcols] = finalData[targetcols].apply(
        lambda x: [y if y <= 1 else 1 for y in x])
    
    # split the dataset into training and validation datasets
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(finalData[traincols], finalData[targetcols])
    
    # create a count vectorizer object for cleanedDesc column
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(train_x['cleanedDesc'])
    
    # transform the training and validation data using count vectorizer object
    xtrain_count = count_vect.transform(train_x['cleanedDesc'])
    xvalid_count1 = count_vect.transform(valid_x['cleanedDesc'])
    uni = pd.DataFrame(xtrain_count.toarray(), columns=count_vect.get_feature_names())
   
    # create a count vectorizer object for cleanedTitle column
    count_vect2 = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect2.fit(train_x['cleanedTitle'])
    
    # transform the training and validation data using count vectorizer object
    xtrain_count = count_vect2.transform(train_x['cleanedTitle'])
    xvalid_count2 = count_vect2.transform(valid_x['cleanedTitle'])
    uni2 = pd.DataFrame(xtrain_count.toarray(), columns=count_vect2.get_feature_names())
    
    count_vect3 = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect3.fit(train_x['Occurrence_Type_Code'])
    
    # transform the training and validation data using count vectorizer object
    xtrain_count = count_vect3.transform(train_x['Occurrence_Type_Code'])
    xvalid_count3 = count_vect3.transform(valid_x['Occurrence_Type_Code'])
    uni3 = pd.DataFrame(xtrain_count.toarray(), columns=count_vect3.get_feature_names())

    # Adding created features into the list of term attributes
    uni.index = train_x.index
    uni2.index = uni.index
    uni3.index = uni2.index
    trainData = pd.concat([uni, uni2, uni3, train_x[['word_count', 'char_count', 'stopwords_count', 'hastags_count', 'numerics', 'word_count_2', 'char_count_2', 
                 'stopwords_count_2', 'hastags_count_2','numerics_2']]],
                          axis=1)
    trainDataX = scipy.sparse.csr_matrix(trainData.values)  # Training X Sparse Matrix
    trainDataY = scipy.sparse.csr_matrix(train_y.values)  # Training Y Sparse Matrix

    # Adding created features into the list of term attributes
    uni4 = pd.DataFrame(xvalid_count1.toarray(), columns=count_vect.get_feature_names())
    uni5 = pd.DataFrame(xvalid_count2.toarray(), columns=count_vect2.get_feature_names())
    uni6 = pd.DataFrame(xvalid_count3.toarray(), columns=count_vect3.get_feature_names())
    
    uni4.index = valid_x.index
    uni5.index = uni4.index
    uni6.index = uni5.index
    validData = pd.concat([uni4, uni5, uni6, valid_x[['word_count', 'char_count', 'stopwords_count', 'hastags_count', 'numerics', 'word_count_2', 'char_count_2', 
                 'stopwords_count_2', 'hastags_count_2','numerics_2']]],
                          axis=1)
    validDataX = scipy.sparse.csr_matrix(validData.values)  # Validation X Sparse Matrix
    validDataY = scipy.sparse.csr_matrix(valid_y.values)  # Validation Y Sparse Matrix
    return [trainDataX, trainDataY, validDataX, validDataY]

###########################  Main Body  ###################################
def main():
    originalData = readData(r'C:\Users\adcdevadmin\Documents\Projects\SSE_Txt_Classfication')
    (trainDataX, trainDataY, validDataX, validDataY) = modelPrep(originalData)

    # initialize LabelPowerset multi-label classifier with a RandomForest
    classifier = LabelPowerset(
        classifier=RandomForestClassifier(n_estimators=100),
        require_dense=[False, True]
    )
    # train
    classifier.fit(trainDataX, trainDataY)
    # predict
    predictions = classifier.predict(validDataX)
    pred_c = predictions.toarray()
    valid_c = validDataY.toarray()
    
    # Calculating subset accurcy for a multilabel classification
    print('Subset Accuracy is: ', accuracy_score(valid_c, pred_c, normalize=True))
    
    # Calculate metrics for each instance, and find their average (only meaningful for multilabel classification where this differs from accuracy_score).
    print('f1 sample average score: ', f1_score(valid_c, pred_c, average='samples'))
    
    targetcols = ['10048', '10052', '10054', '10055', '10176', '10213', '10214', '10215',
                  '10235', '10240', '10242', '10252', '10262', '10268', '10269', '10270',
                  '10271', '10272', '10278', '10279', '10287', '10294', '10295', '10321',
                  '10329', '10353', '10375', '10376', '10398', '10423', '10429', '10449',
                  '10459', '10470', '10472', '10507', '10512', '10521', '10525', '10528',
                  '10538', '10557', '10558', '10571', '10580', '10599', '10622', '10644',
                  '10656', '10666', '10701', '10713', '10727', '10741', '10763', '10774',
                  '10775', '10789', '10797', '10813', '10835', '10854', '10874', '10882',
                  '10891', '10911', '10924', '10931', '10940', '10946', '10947', '10982',
                  '11009', '11308', '11309', '11314', '9672', '9673', '9702', '9733',
                  '9757', '9783', '9809', '9917', '9918']
     
    report = skm.classification_report(valid_c, pred_c, target_names=targetcols, output_dict=True)
    rep = pd.DataFrame.from_dict(report, orient='index')
    rep.to_csv('tag_output_2.0.csv')
   
########################### Call Main ###################################
if __name__ == '__main__':
    main()