# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 09:53:41 2020
@author: rajhi
"""
import pandas as pd, numpy as np, math as mt, re 
from collections import Counter
import nltk
nltk.download('punkt')
from nltk import word_tokenize
import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS

## Loading the text file

def loadFile():
    book = open(r"data\book_scrambled.txt","r", encoding='utf-8')
    content = book.readlines()
    content = [x for x in content if x != []]
    texts = []
    texts = [x.strip(' ') for x in content]
    texts = list(filter(None, texts))
    return texts    
 
## Cleaning the text file  
    
def cleanData(texts):
    tempdf = pd.DataFrame(texts)
    tempdf.columns = {"content"}
    # Cleaning all the empty lines
    tempdf['content'].replace('\n', np.nan, inplace=True)
    tempdf2 = tempdf.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    return tempdf2  

def decodeChapters(df):
    # Getting list of  prime nos between a range
    start = 1
    end = 200
    primesnos = []
    for val in range(start, end + 1):
        if val > 1: 
            for n in range(2, val): 
                if (val % n) == 0: 
                    break
            else:
                
                primesnos.append(val)
    comp= pd.DataFrame(primesnos)
    comp.columns = {"orgChap"}
    comp['sqaured'] = comp['orgChap']*comp['orgChap']
    comp['actual'] = comp.index
    # Finding all the rows with chapter names
    temp = df[df['content'].str.match('Chapter')]
    temp[['Chapter','Number']] = temp.content.str.split(expand=True) 
    temp['OrgChap'] = temp['Number'].apply(lambda x: mt.sqrt(int(x)))
    temp['finalChap'] = temp['OrgChap'].map(comp.set_index('orgChap')['actual'])
    #Appending all the values with 1 as chapter usually starts with 1
    temp['finalChap'] = temp['finalChap']+1
    temp['chapterList'] = temp['finalChap'].apply(lambda x: 'Chapter ' + str(x))
    
    #Replacing the correct chapter list in the content
    df.loc[temp.index] = temp[['chapterList']]
    return df     


def decodeNoise1(df):
    # Finding and replacing foo
    df[df['content'].str.lower().str.find('foo ') >0 ]
    # We found total 17,361 rows added with foo
    # overwriting column with replaced value of age  
    df["content"]= df["content"].str.replace("foo ", ",", case = False)
    return df

def decodeNoise2(df):
    # Finding the long words with space substitued as nos
    temp = df[df['content'].str.split().str.len().lt(2)]
    temp['content'] = temp.replace({'\d+': ' '}, regex=True)
    df.update(temp)
    df['content'] = df['content'].str.strip()
    return df
        
def analytics(df):
    # Counting Total no of words
    df.content.apply(lambda x: pd.value_counts(x.lower().split(" "))).sum(axis = 0)
    texts = df['content'].str.lower()    
    word_counts = Counter(word_tokenize('\n'.join(texts)))
    word_counts.most_common()
    comment_words = ' '
    stopwords = set(STOPWORDS) 
    '''      
    # iterate through the csv file 
    for val in df.content:
        # typecaste each val to string 
        val = str(val) 
        # split the value 
        tokens = val.split() 
        # Converts each token into lowercase 
        for i in range(len(tokens)): 
               tokens[i] = tokens[i].lower() 
              
        for words in tokens: 
             comment_words = comment_words + words + ' '
      
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(comment_words) 
      
    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0)      
    plt.show() 
    '''    
    #Compute the average word-length of a sentence, and the longest one.
    newdf = df.copy()
    newdf = newdf['content'].str.cat(sep= ' ')
    concat = newdf.replace(' ', '')
    concat = newdf.replace(' ', '')
    concat = concat.replace('\n', '')
    pat = ('(?<!The)(?<!Esq)\. +(?=[A-Z])')
    concat = re.sub(pat,'.\n',newdf).strip()
    data_df = pd.DataFrame([sentence for sentence in concat.split('.') if sentence],
                       columns=['sentences'])
    # The dataframe is created by seperating all the sentences with . thefore the no of rows =total no of senences
    print ("The total no of sentences are " + str (data_df.size))
    
    # Compute the average word-length of a sentence, and the longest one.
    data_df['word_legth'] = data_df['sentences'].apply(lambda x: len(x.split()))
    print("The average word lenght is " + str((data_df['word_legth'].mean())))
    print("The maximum word lenght in a sentence is " + str((data_df['word_legth'].max())))
    
      
## Main Function
    
def main():    
    content = loadFile()
    cleanedContent = cleanData(content)
    chapters = decodeChapters(cleanedContent)
    deNoise =  decodeNoise1(chapters)
    deNoise2 = decodeNoise2(deNoise)
    analytics(deNoise2)   
    
    
## Main call    
if __name__ == "__main__":
    main()
