# -*- coding: utf-8 -*-
#
# Author: Dagmar Gromann
# Description: Method to retrieve Polyglot embeddings for individual input words from a pretrained embedding repository
# Embedding source: https://sites.google.com/site/rmyeid/projects/polyglot

import polyglot
import pickle

from polyglot.mapping import Embedding
from polyglot.text import Text, Word
from normalizer import normalize

PATH_TO_EMBEDDINGS = "yourPath"
PATH_OOV = "yourPathMissingWords"

'''
This method retrieves the embeddings for each word in the corresponding language from the local 
polyglot repository. The path below needs to be set to the corresponding local respository
'''
def getEmbedding(word, language):
	with open(PATH_TO_EMBEDDINGS+language+'/words_embeddings_32.pkl', 'rb') as f:
		words, embeddings = pickle.load(f, encoding='latin1')
	word_id = {w:i for (i, w) in enumerate(words)}

	#Try to find the most frequent alternative if word is not in embeddings
	if not word in words: 
		#print(word, " not in polyglot")
		word = normalize(word, word_id)
		#print(word, " alternative word in polyglot")
		
	if word in words:
		word_index = word_id[word]
		vector = embeddings[word_index]
		return vector


'''
Retrieve all Polyglot embeddings for all words in both GICS and ICB
'''
def getPolyglotEmbeddingDict(gics, icb, language, stopWordLang): 
	missing = open(PATH_OOV+language+"_notInPolyglot.txt", "w")
	embedDict = {}
	noneValues = 0

	for key, value in icb.items():
		icbWords = getWords(value, stopWordLang)
		for icbWord in icbWords: 
			icbEmbedding = getEmbedding(icbWord, language)
			if icbEmbedding is not None: 
				if icbWord not in embedDict.keys(): 
					embedDict[icbWord] = icbEmbedding
			else:
				noneValues +=1 
				missing.write("ICB "+icbWord+"\n")

	for key, value in gics.items(): 
		gicsWords = getWords(value, stopWordLang)
		for gicsWord in gicsWords:
			gicsEmbedding = getEmbedding(gicsWord, language)
			if gicsEmbedding is not None: 
				if gicsWord not in embedDict.keys():
					embedDict[gicsWord] = gicsEmbedding 
			else: 
				noneValues +=1 
				missing.write("GICS "+gicsWord+"\n")
	
	print (len(embedDict), noneValues)
	return embedDict