########################################################################
##  w2v_modules.py
##  A collection of useful w2v modules to be used as plugins in larger code
##
##  by David Curry
##
#######################################################################

import nltk, re, pickle
from nltk.corpus import stopwords
from nltk import FreqDist
import numpy as np
import gensim.models.word2vec as w2v
from gensim.models.word2vec import Word2Vec

import keras
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier



def sent_token(text):
    ''' Helper function for sentence_clean()'''
    text = nltk.sent_tokenize(text)
    return text

def sentence_clean(text):

    ''' Cleans sentences in a text and tokenizes words.
    Input: Text Body
    Output: List of tokenized sentences.
    '''

    new_text = []
    for sentence in text:
        sentence = sentence.lower()
        sentence = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", sentence)
        sentence = re.sub("[^a-z ]", "", sentence)
        sentence = nltk.word_tokenize(sentence)
        sentence = [word for word in sentence if len(word)>1] # exclude 1 letter words
        new_text.append(sentence)
    return new_text

def apply_all(text):
    ''' Helper function for sentence_clean()'''
    return sentence_clean(sent_token(text))



def create_w2v_nltk_sentence(dataframe, hyperparameters, title):

    '''
    Creates and trains a wrod2vec instance on body of texts.
    Input:  Dataframe of text, w2v params, title for storage
    Dataframe must have its text in a column labeled 'text'.
    Output: w2v weight matrix, vocab list, dict of {word:vector}
    '''

    # Tokenize the sentences
    print('\n Tokenizing All Sentences with NLTK...')
    dataframe['sent_tokenized_text'] = dataframe['text'].apply(apply_all)
    all_sentences = list(dataframe['sent_tokenized_text'])
    #all_sentences = [subitem for item in all_sentences for subitem in item]

    print(len(all_sentences))
    
    
    # convert to text and labels to numpy arrays
    train_text = np.array(all_sentences)
    train_y    = dataframe['class'].values
    
    print('Initial Train Dimensions:', train_text.shape)
    print('Initial Label Dimensions:', train_y.shape)

    model = Word2Vec(train_text, size=100, window=5, min_count=5, workers=2)
    w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
    
    # Init the w2v module
    # all2vec = w2v.Word2Vec(
    #     sg=1,
    #     workers=hyperparameters['num_workers'],
    #     size=hyperparameters['num_features'],
    #     min_count=hyperparameters['min_word_count'],
    #     window=hyperparameters['context_size'],
    #     sample=hyperparameters['downsampling']
    # )

    # all2vec.build_vocab(all_sentences)
    # print("Word2Vec vocabulary length:", len(all2vec.wv.vocab))
    # print('\nTraining Word2Vec...')
    # all2vec.train(all_sentences, total_examples=all2vec.corpus_count, epochs=all2vec.iter)
    # all_word_vectors_matrix = all2vec.wv.syn0
    # vocab = all2vec.wv.vocab
    # vocab_index = all2vec.wv.index2word
    # pickle.dump(all_word_vectors_matrix, open('weights/w2v_'+title+'_matrix.p', 'wb'))
    # pickle.dump(vocab, open('weights/w2v_'+title+'_vocab.p', 'wb'))
    # pickle.dump(vocab_index, open('weights/w2v_'+title+'_index.p', 'wb'))

    #return all_word_vectors_matrix, vocab, vocab_index
    return w2v

def create_w2v_cv(cv_text, hyperparameters, title):

    '''
    Creates and trains a wrod2vec instance on sci-kit count vectorized text.
    Input:  CV numpy array, w2v params, title for storage
    Dataframe must have its text in a column labeled 'text'.
    Output: w2v dictionary of wards to context vectors
    '''
    
    print('\nTraining Word2Vec...')

    # Init the w2v module
    all2vec = Word2Vec(
        sg=1,
        workers=hyperparameters['num_workers'],
        size=hyperparameters['num_features'],
        min_count=hyperparameters['min_word_count'],
        window=hyperparameters['context_size'],
        sample=hyperparameters['downsampling']
    )
    
    all2vec.build_vocab(cv_text)
    all2vec.train(cv_text, total_examples=all2vec.corpus_count, epochs=all2vec.iter)
    all_word_vectors_matrix = all2vec.wv.syn0
    vocab = all2vec.wv.vocab
    vocab_index = all2vec.wv.index2word
    
    # Create a dictionary or word index to context vector
    w2v = {w: vec for w, vec in zip(vocab_index, all_word_vectors_matrix)}

    pickle.dump(w2v, open('weights/w2v_'+title+'_dict.p', 'wb'))
    
    return w2v


class MeanEmbeddingVectorizer(object):

    '''
    Class for creating features from word2vec output weight matrix and be used in sci-kit's pipeline.
    Input: word2vec trained object
    Output: New features from word2vec
    '''

    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec) 
    
    def fit(self, X, y):
        return self 

    def transform(self, X):

        print('\nAveraging w2v word vectors for new features...')
        
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    def mean_features():
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    



