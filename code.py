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
x_vec2 = vectorizer.fit_transform(x)
#viewing the shape of the variables
print (y.shape)
print (x_vec.shape)
print (x_vec2.shape)

#MACHINE LEARNING TECHNIQUES
#Naive Bayes Classification
from sklearn.naive_bayes import MultinomialNB
#initialize Naive Bayes classifier
naive_bayes = MultinomialNB()

#split into test and train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_vec, y, test_size = 0.2, random_state = 42)

#train the classifier
NB_classifier = naive_bayes.fit(x_train, y_train)
#Evaluating the test data
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
pred = NB_classifer.predict(x_test)
#printing the confusion matrix
print (confusion_matrix(y_test, pred))

from sklearn.metrics import ConfusionMatrixDisplay
con_matrix = confusion_matrix(y_test, pred)
con_matrix_display = ConfusionMatrixDisplay(confusion_matrix = con_matrix, display_labels = [0, 1])
con_matrix_display.plot()
plt.title('Naive Bayes Confusion Matrix')
plt.show()

print ('Accuracy Score:', accuracy_score (y_test, pred))
print ('roc_auc_score:', roc_auc_score(y_test, pred))

from sklearn.metrics import classification_report
print (classification_report (y_test, pred))

#Support Vector Machine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm

#data transformation to convert cleaned text into bag of words
vectorizer_svm = CountVectorizer (max_features = 1420)

#storing bag of words representation in x
x_svm = vectorizer_svm.fit_transform(x).toarray()

tfidf = TfidfTransformer()
x_tfidf = tfidf.fit_transform (x_vec)

#splitting the dataset 
x_train, x_test, y_train, y_test = train_test_split(x_svm, y, test_size = 0.20, random_state = 42)
SVM_classifier = svm.SVC(kernel = 'linear', C = 1, degree = 3, gamma = 'auto')
SVM_classifier.fit(x_train, y_train)

y_pred = SVM_classifier.predict(x_test)
confusion_matrix(y_test, y_pred)

con_matrix2 = confusion_matrix(y_test, y_pred)
con_matrix_display2 = ConfusionMatrixDisplay(confusion_matrix = con_matrix2, display_labels = [0, 1])
con_matrix_display2.plot()
plt.title ('SVM Confusion Matrix')
plt.show()

print ('Accuracy Score:', accuracy_score (y_test, y_pred))
print ('roc_auc_score:', roc_auc_score(y_test, y_pred))

from sklearn.metrics import classification_report
print (classification_report (y_test, y_pred))

#DEEP LEARNING TECHNIQUES
import keras
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from keras.layers import Flatten, GlobalMaxPooling1D, Embedding, Conv1D, LSTM

y = movie_df['label']
#split data 
#train set will be used to train deep learning models
#test set will be used to evaluate the performace of the model
a_train, a_test, b_train, b_test = train_test_split(x, y, test_size = 0.20, random_state = 42)

#preparing embedded layer to convert test data to numeric form
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(a_train)
a_train = word_tokenizer.texts_to_sequences(a_train)
a_test = word_tokenizer.texts_to_sequences(a_test)

#we add 1 to store dimensions where no pretrained word embeddings exist
vocab_length = len(word_tokenizer.word_index) + 1
#to know the number of unique words
vocab_length 

#padding all reviews to fixed length of 100
maxlen = 100
a_train = pad_sequences(a_train, padding = 'post', maxlen = maxlen)
a_test = pad_sequences(a_test, padding = 'post', maxlen = maxlen)

#load GloVe word embedding and create an embeddings dict
from numpy import asarray
from numpy import zeros
embeddings_dict = dict()
glove_file = open('glove.6B.100d.txt', encoding = 'utf8')
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype = 'float32')
    embeddings_dict[word] = vector_dimensions
glove_file.close()

#lets create an embedded matrix having 100 columns containing 100-dimensional GloVe word embeddings
embedding_matrix = zeros((vocab_length, 100))
for word, index in word_tokenizer.word_index.items():
    embedding_vec = embeddings_dict.get(word)
    if embedding_vec is not None:
        embedding_matrix[index] = embedding_vec
embedding_matrix.shape

#Simple Neural Network
a_model = Sequential()
embedded_layer = Embedding (vocab_length, 100, weights = [embedding_matrix], input_length = maxlen, trainable = False)
a_model.add(embedded layer)
a_model.add(Flatten())
a_model.add(Dense(1, activation = 'sigmoid'))

a_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
print (a_model.summary())

a_model_history = a_model.fit(a_train, b_train, batch_size = 120, epochs = 6, verbose = 1, validation_split = 0.2)
performance = a_model.evaluate(a_test, b_test, verbose = 1)

print ('Test Score:', performance[0])
print ('Test Accuracy:', performance[1])

#plotting chart of performance
plt.plot (a_model_history.history['acc'])
plt.plot(a_model_history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel ('Accuracy')
plt.xlabel ('Epoch')
plt.legend (['train', 'test'], loc = 'upper left')
plt.show()

plt.plot (a_model_history.history ['Loss'])
plt.plot (a_model_history.history ['val_loss'])
plt.title ('Model Logs')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()
