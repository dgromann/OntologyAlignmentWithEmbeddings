# -*- coding: utf-8 -*-
#
# Author: Dagmar Gromann
# Description: Methods to retrieve word2vec embeddings from pretrained libraries
# Embeddings: https://github.com/Kyubyong/wordvectors

from __future__ import print_function

import numpy as np
import re

from gensim.models import KeyedVectors, Word2Vec
from semSim import cosine_similarity
from nltk.corpus import stopwords
from time import time
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()
DIGITS = re.compile("[0-9]", re.UNICODE)

PATH_TO_GOOGLE_EMBEDDINGS ="yourPathToGoogleNewsCorpus"
PATH_TO_WIKI_EMBEDDINGS = "yourPath"
PATH_OOV = "yourPathMissingWords"

'''
This methods splits strings of words into individual words and performs some preprocessing
'''
def getWords(label, language):
	label = "".join(x for x in label if x not in ('!', ',', ':', '(',')',';','&'))
	#text = Text(label.strip())
	label = label.strip()
	label = label.replace("  ", " ")
	label = label.split(" ")
	wordlist = [word for word in label if word not in stopwords.words(language)]
	return wordlist

'''
This method does a minimum of normalization and attempts lemmatization to 
increase the retrieval of embeddings from Polyglot in case the string 
is not found as is.
'''
def normalize(word, word_id):
	if not word in word_id:
		word = DIGITS.sub("#", word)
	if not word in word_id:
		if word.lower() in word_id:
			word = word.lower()
		if word.upper() in word_id: 
			word = word.upper()
		if word.title() in word_id:
			word = word.title()
	if not word in word_id: 
		word = wordnet_lemmatizer.lemmatize(word)

	if not word in word_id:
		return None
	return word

def getEmbedding(word, model): 
	if word in model.vocab: 
		return model[word]
	else: 
		word = normalize(word, model.vocab)
		if word in model.vocab:
			return model[word]
		else:
			return None

def getW2VEmbeddingFromExternal(word, language):
	model = KeyedVectors.load_word2vec_format(PATH_TO_WIKI_EMBEDDINGS+language+".tsv", binary=False)
	return getEmbedding(word, model)

def getW2VEmbedDict(gics, icb, language, stopWordLang):
	missing = open(PATH_OOV+language+"notInW2V.txt", "w")
	model = Word2Vec.load(PATH_TO_WIKI_EMBEDDINGS+language+"/"+language+".bin")

	embedDict = {}
	noneValues = 0

	for key, value in icb.items():
		icbWords = getWords(value, stopWordLang)
		for icbWord in icbWords:
			icbEmbedding = getEmbedding(icbWord, model)
			if icbEmbedding is not None: 
				if icbWord not in embedDict.keys():
					embedDict[icbWord] = icbEmbedding
			else:
				noneValues +=1 
				missing.write("ICB "+icbWord+"\n")

	for key, value in gics.items(): 
		gicsWords = getWords(value, stopWordLang)
		for gicsWord in gicsWords:
			gicsEmbedding = getEmbedding(gicsWord, model)
			if gicsEmbedding is not None: 
				if gicsWord not in embedDict.keys():
					embedDict[gicsWord] = gicsEmbedding
			else: 
				noneValues +=1 
				missing.write("GICS "+gicsWord+"\n")

	return embedDict

'''
Loads the word2vec model into memory and creates the model
'''
def getWord2VecModel():
	model = KeyedVectors.load_word2vec_format(PATH_TO_EMBEDDINGS+'GoogleNews-vectors-negative300.bin', binary=True)

	# makes the w2v model faster but you cannot train it anymore
	model.init_sims(replace=True)
	return model

'''
Creates a vector for a word
'''
def wordVector(word, model):
	word.replace(' ', '_')
	try:
		vector = model[word]
		return vector
	except:
		print(word)
	return None