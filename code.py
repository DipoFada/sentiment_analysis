#importing the python libraries
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import spacy
import wordcloud
import gensim
from gensim.utils import simple_preprocess

#reading the data
movie_df = pd.read_csv('IMDB_Datset.csv')
movie_df.head()

#checking for null values
movie_df.describe()
movie_df.isnull().sum()

#viewing the balancing
ax2 = sns.countplot(data = movie_df, x = 'sentiment', stat = 'percent')
for container in ax2.containers:
  ax2.bar_label(container, fmt = '%0.1f%%')
ax = sns.countplot(data = movie_df, x = 'sentiment', hue = 'sentiment')
plt.title ('Sentiment Classification of Movie Reviews')
plt.xlabel ('Sentiment')
plt.ylabel ('Count')

#DATA PREPROCESSING
#preprocessing
from nltk.corpus import stopwords
#reviewing a sample
movie_df['review'][2]

#removing HTML tags using regex library
tags_re = re.compile (r'<[^>]+>')
def remove_tags(text):
  """
  This function removes the HTML tags by replacing anything that has opening and closing braces < > with spaces
  input: text in string format
  output: text with tags replaced with empty spaces
  """
  return tags_re.sub(' ', text)

#more text cleaning
def process_test(textline):
  """
  This function cleans text data leaving 2 or more words composed of letters in lower case
  """
  sentence = textline.lower()
  sentence = remove_tags(sentence) #remove HTML tags from function
  sentence = re.sub('[^a-zA-Z]', ' ', sentence) #remove punctuation and numbers
  sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence) #removes single characters like I, A
  sentence = re.sub(r'\s+', ' ', sentence) 
  #remove stopwords
  pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
  sentence = pattern.sub('', sentence)
  
  return sentence
#calling preprocessed text function on movie_df review
x = []
sentences = list(movie_df['review'])
for sen in sentences:
  x.append (process_text(sen))

#viewing the processed version of movie_df['review'][2]
x[2]

#encoding the sentiment column with 1 for positive and 0 for negative
sentiment_encode = {'positive' : 1, 'negative' : 0}
movie_df['label'] = movie_df['sentiment'].replace(sentiment_encode)

movie_df

#separating the number of words to build N-grams models
def sent_to_words(sentences):
  for sentence in sentences:
    yield(gensim.utils.simple_preprocess(str(sentence), deacc = False))
data = movie_df['review'][0:500].tolist()
data_words = list (sent_to_words(data))
print ('data_words[:1]', data_words[:1])

#build the bigram and trigram models
bigram = gensim.models.Phrases (data_words, min_count = 5, threshold = 100)
trigram = gensim.models.Phrases(bigram[data_words], threshold = 100)

bigram_mod = gensim.models.phrases.Phrases (bigram)
trigram_mod = gensim.models.phrases.Phraser (trigram)

bigram_mod.save('bigram_mod')
trigram_mod.save('trigram_mod')

bigram_mod = gensim.models.Phrases.load('bigram_mod')
trigram_mod = gensim.models.Phrases.load('trigram_mod')

for bigram in bigram_mod.phrasegrams.keys():
  print (bigram)
print (' ')
for trigram in trigram_mod.phrasegrams.keys():
  print (trigram)

from gensim.parsing.preprocessing import STOPWORDS
stopwords = STOPWORDS.union(set(['br']))

def remove_stopwords(texts):
  return [[word for word in simple_preprocess(str(doc)) if word not in stopwords] for doc in texts]
data_words_nostops = remove_stopwords(data_words)

def make_bigrams (texts):
  return [bigram_mod[doc] for doc in texts]
data_words_bigrams = make_bigrams(data_words_nostops)

def make_trigrams(texts):
  return [trigram_mod[bigram_mod[doc]] for doc in texts]

#performing lemmatize tasks
def lemmatize (text, allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']):
  """
  https://spacy.io/api/annotation
  """
  texts_out = []
  for sent in texts:
    doc = nlp(" ".join(sent))
    texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
  return texts_out

nlp = spacy.load('en_core_web_sm', disable = ['parser', 'ner'])
data_lemmatized = lemmatize (data_words_bigrams, allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV'])
print ('Lemmatized Data[:1]', data_lemmatized[:1])

#developing a wordcloud
def getWordCloud(hashtags):
  text = ' '.join(hashtags)
  wordcloud = WordCloud().generate(text) 
  plt.imshow(wordcloud, interpolation = 'bilinear') #display the generated image
  plt.axis('off')

  wordcloud = WordCloud(max_font_size = 40).genreate(text)
  plt.figure()
  plt.imshow(wordcloud, interpolation = 'bilinear')
  plt.axis('off')
  plt.savefig('wordcloud_all.pdf', dpi = 500)
  plt.show()

string = []
for l in dama_lemmatized:
  t = ' '.join(l)
  string.append(t)

getWordCloud(string)

#FEATURE EXTRACTION
#TFID Vectorizer
from sklearn.feature_extraction.text import TfidVectorizer
vectorizer = TfidfVectorizer(use_idf = True, stop_words = 'english', lowercase = True, strip_accents = 'ascii')
y = movie_df['label']
#apply tfidf vectorizer
x_vec = vectorizer.fit_transform (movie_df['review'])
#vectorizer on cleaned text
x_vec2 = vectorizer.fit_transform
