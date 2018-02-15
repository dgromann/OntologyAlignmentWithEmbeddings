# -*- coding: utf-8 -*-
#
# Author: Dagmar Gromann 
# Description: Program for some basic string normalizations to reduce number of out of vocabulary words 

import re
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()
DIGITS = re.compile("[0-9]", re.UNICODE)

'''
This method tries various case representations of the input word in order to improve the retrieval rate 
of embeddings from Polyglot.
dictionary: words in Polyglot for each language 
'''
def case_normalizer(word, dictionary):
	w = word
	lower = (dictionary.get(w.lower(), 1e12), w.lower())
	upper = (dictionary.get(w.upper(), 1e12), w.upper())
	title = (dictionary.get(w.title(), 1e12), w.title())
	results = [lower, upper, title]
	results.sort()
	index, w = results[0]
	if index != 1e12:
		return w
	return word

'''
This method does a minimum of normalization and attempts lemmatization to 
increase the retrieval of embeddings from Polyglot in case the string 
is not found as is.
'''
def normalize(word, word_id):
    """ Find the closest alternative in case the word is OOV."""
    if not word in word_id:
        word = DIGITS.sub("#", word)
    if not word in word_id:
        word = case_normalizer(word, word_id)
    if not word in word_id: 
    	word = wordnet_lemmatizer.lemmatize(word)
    
    if not word in word_id:
        return None
    return word