# -*- coding: utf-8 -*-
#
# Author: Dagmar Gromann <dagmar.gromann@gmail.com>
# Description: This program aligns two industry clasification standards using word embeddings from Polyglot. The embeddings and the 
# input data are pre-stored in a pkl provided in the "data" folder to speed up the processing, but can be retrieved
# with this code by uncommenting the corresponding lines. 

#Sources for embeddings: 
#Multilingual word2vec: https://github.com/Kyubyong/wordvectors
#English word2Vec Google News embeddings: https://code.google.com/archive/p/word2vec/
#Multilingual fastText embeddings: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
#Polyglot: https://sites.google.com/site/rmyeid/projects/polyglot

import pickle
import numpy as np
import polyglot

from scipy.optimize import linear_sum_assignment
from numpy import ndarray, array, float32 as REAL
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from polyglot.text import Text, Word

from semSim import getEditDistance, jaccard_similarity, getJensenShanon, JSD, l2norm, cosine_similarity, readEvalFile
from polyglotEmbeddings import getEmbedding, getPolyglotEmbeddingDict
from fastTextEmbeddings import getFastTextEmbeddingDict, getFastTextEmbeddingFromExternal
from word2vecEmbeddings import getW2VEmbedDict, getWord2VecModel, wordVector
from collections import Counter


INPUTPATH = "pathOfPickledIndustryClassifications"
#An example of a decomposition file is provided in the folder "decomposition file"
PATH_GERMAN_COMPOUNDS = "decomposedCompoundsFile"
EVALUATIONFILE = "evaluationFilePath"
OUTPUTPATH = "resultsPath"
EMBEDDINGNAME="outputSubFolder"

'''
This methods splits strings of words into individual words and performs some preprocessing
'''
def getWords(label, language):
	label = "".join(x for x in label if x not in ('!', ',', ':', '(',')',';','&'))
	label = label.strip()
	label = label.replace("  ", " ")
	label = label.split(" ")
	wordlist = [word for word in label if word not in stopwords.words(language)]
	return wordlist

'''
Compare the generated alignment to the manual evaluation and calculate accuraccy metrics
'''
def getResults(simDict, language):
	print("Write alignment results to file and compare against manual alignment")
	'''evaluation - retrieves the manual Alignments and compares them to the results outputing 
	precision, recall, f-measure'''
	evalFile = open(EVALUATIONFILE, "r")
	results = open(OUTPUTPATH+EMBEDDINGNAME+"/results_"+language+"_"+EMBEDDINGNAME+".txt", "w")
	counter = 0
	allAlignments = 0
	recallCounter = 0
	icbrecord =  []
	value = 0
	for row in evalFile:
		allAlignments += 1
		ids = row.split(" ")
		gicID = int(ids[0].strip())
		icbID = ids[1].strip()

		if gicID in simDict.keys(): 
			value = simDict[gicID]
			recallCounter += 1

		if icbID == value:
			counter += 1

	print("recall", recallCounter)
	print("counter", counter, "all alignments", allAlignments)
	precision = counter/recallCounter
	recall = recallCounter/allAlignments
	print(language)
	print("Precision: ",round(precision, 3))
	print("Recall: ", round(recall, 3))
	print("FMeasure: ", round((2*precision*recall)/(precision+recall), 3))
	results.write("Precision: "+str(round(precision, 3))+"\n"+"Recall: "+str(round(recall, 3))+"\n"+"FMeasure: "+str(round((2*precision*recall)/(precision+recall), 3)))
	evalFile.close()
	results.close()

'''
Replace the previously and externally decomposed German compounds in the current and local data structure
'''
def replaceGermanCompounds(gics, icb):
	with open(PATH_GERMAN_COMPOUNDS) as f:
		decomposed = list(x.strip().split('=') for x in f)

	for key, value in gics.items():
		if str(key)[0:4] != "4040" and key != 25502010:
			wordList = value.split(" ")
			for word in wordList:
				if word[-1:] == "-": 
					gics[key] = value.replace("-", "") 
				for st in decomposed: 
					if "".join(st) == word.lower():
						newValue = "" 
						replaceValue = list()
						replaceValue += [w.capitalize() for w in st if w != "s"]
						newValue = value.replace(word, " ".join(replaceValue))
						newValue = newValue.replace("-", " ")
						gics[key] = newValue

	for key, value in icb.items():
		wordList = value.split(" ")
		for word in wordList:
			if word[-1:] == "-": 
				icb[key] = value.replace("-", "")
			for st in decomposed: 
				if "".join(st) == word.lower(): 
					newValue = ""
					replaceValue = list()
					replaceValue += [w.capitalize() for w in st if w != "s"]
					newValue = value.replace(word, " ".join(replaceValue))
					newValue = newValue.replace("-", " ")
					icb[key] = newValue

	return gics, icb

'''
Method that generates the similarity matrix based on a given embedding data structure 
for both resources, GICS and ICB
'''
def getEmbeddingAlignment(gics_lang, icb_lang, language, stopWordLang, embedDict):
	print("Generate similarity matrix based on cosine values of embeddings")
	Matrix = []
	gics_vec=[]
	icb_vec = []
	evalIDs = readEvalFile()
	
	for gicsID, gicsLabel in gics_lang.items():
		if str(gicsID)[0:4] != "4040" and gicsID != 25502010 and gicsID in evalIDs:	
			gics_vec.append(gicsID)
			gicsLabel = gicsLabel.replace("-", " ")
			gicsWords = getWords(gicsLabel, stopWordLang)

			vector = np.zeros(len(icb_lang))
			for icbID, icbLabel in icb_lang.items():
				if icbID in evalIDs: 
					lList = list()
					icb_vec.append(icbID)
					icbLabel = icbLabel.replace("-", " ")
					icbWords = getWords(icbLabel, stopWordLang)
					lList += [w for w in gicsWords]
					lList += [w for w in icbWords if w not in lList]

					gicsVec = []
					for elem in lList:
						gicsHelper = []
						if elem in embedDict.keys():
							for gicsWord in gicsWords:
								if gicsWord in embedDict.keys():
									gicsHelper.append(cosine_similarity(embedDict[elem], embedDict[gicsWord]))
								else:
									gicsHelper.append(-1)
						else:
							gicsHelper.append(-1)
						gicsVec.append(max(gicsHelper))

					if sum(gicsVec)/len(gicsVec) != -1:
						#Plus one to equal out the -1 of the OOV and -1 on the length to divide by the number 
						#of other elements of the combined vector of unique words across both strings
						gicsVec = [(sum(gicsVec)+1)/(len(gicsVec)-1) if x==-1 else x for x in gicsVec]
					
					icbVec = []
					for elem in lList:
						icbHelper = []
						if elem in embedDict.keys():
							for icbWord in icbWords:
								if icbWord in embedDict.keys():
									icbHelper.append(cosine_similarity(embedDict[elem], embedDict[icbWord]))
								else:
									icbHelper.append(-1)
						else:
							icbHelper.append(-1)
						icbVec.append(max(icbHelper))
					
					if sum(icbVec)/len(icbVec) != -1:
						#Plus 1 on the sum to equal out the -1 of the OOV and minus 1 on the length 
						#to ignore the OOV element when averaging
						icbVec = [(sum(icbVec)+1)/(len(icbVec)-1) if x==-1 else x for x in icbVec]

					vector[icb_vec.index(icbID)] = 	cosine_similarity(gicsVec, icbVec)
					
					#Used to print the example from the LREC2018 paper
					#if "Nondurable" in lList:
					#	print(gicsWords, icbWords)
					#	print(gicsVec, icbVec)
					#	print(cosine_similarity(gicsVec, icbVec))

			Matrix.append(vector)
	return Matrix, gics_vec, icb_vec

'''
Method to obtain the Word2Vec embeddings for all ICB and GICs labels 
'''
def getWord2VecAlignment(gics_lang, icb_lang, language, stopWordLang):
	print("Generate similarity matrix based on cosine values of embeddings")
	Matrix = []
	gics_vec=[]
	icb_vec = []
	evalIDs = readEvalFile()

	model = getWord2VecModel()
	
	for gicsID, gicsLabel in gics_lang.items():
		if str(gicsID)[0:4] != "4040" and gicsID != 25502010 and gicsID in evalIDs:	
			gics_vec.append(gicsID)
			gicsLabel = gicsLabel.replace("-", " ")
			gicsWords = getWords(gicsLabel, stopWordLang)

			vector = np.zeros(len(icb_lang))
			for icbID, icbLabel in icb_lang.items():
				if icbID in evalIDs: 
					lList = list()
					icb_vec.append(icbID)
					icbLabel = icbLabel.replace("-", " ")
					icbWords = getWords(icbLabel, stopWordLang)
					lList += [w for w in gicsWords]
					lList += [w for w in icbWords if w not in lList]

					gicsVec = []
					for elem in lList:
						gicsHelper = []
						elemVector = wordVector(elem, model)
						if elemVector is not None:
							for gicsWord in gicsWords:
								gicsVector = wordVector(gicsWord, model)
								if gicsVector is not None:
									gicsHelper.append(cosine_similarity(elemVector, gicsVector))
								else:
									gicsHelper.append(-1)
						else:
							gicsHelper.append(-1)
						gicsVec.append(max(gicsHelper))

					if sum(gicsVec)/len(gicsVec) != -1:
						#Plus one to equal out the -1 of the OOV and -1 on the length to divide by the number 
						#of other elements of the combined vector of unique words across both strings
						gicsVec = [(sum(gicsVec)+1)/(len(gicsVec)-1) if x==-1 else x for x in gicsVec]
					
					icbVec = []
					for elem in lList:
						icbHelper = []
						elemVect = wordVector(elem, model)
						if elemVect is not None:
							for icbWord in icbWords:
								icbVect = wordVector(icbWord, model)
								if icbVect is not None:
									icbHelper.append(cosine_similarity(elemVect, icbVect))
								else:
									icbHelper.append(-1)
						else:
							icbHelper.append(-1)
						icbVec.append(max(icbHelper))
					
					if sum(icbVec)/len(icbVec) != -1:
						#Plus 1 on the sum to equal out the -1 of the OOV and minus 1 on the length 
						#to ignore the OOV element when averaging
						icbVec = [(sum(icbVec)+1)/(len(icbVec)-1) if x==-1 else x for x in icbVec]

					vector[icb_vec.index(icbID)] = 	cosine_similarity(gicsVec, icbVec)
					
					#Used to print the example from the LREC2018 paper
					#if "Nondurable" in lList:
					#	print(gicsWords, icbWords)
					#	print(gicsVec, icbVec)
					#	print(cosine_similarity(gicsVec, icbVec))

			Matrix.append(vector)
	return Matrix, gics_vec, icb_vec

'''
Method to apply the Munkres algorithm (implementation sklearn) to the calculated similarity matrix
'''
def getAlignment(Matrix, gvec, ivec, gics, icb, lang):
	print("Input similarity matrix to Munkres")
	align = open(OUTPUTPATH+EMBEDDINGNAME+"/alignment_"+lang+".txt", "w")
	alignDict = {}
	simDict = {}
	m = np.matrix(Matrix)
	m = -1 * m
	row_ind, col_ind = linear_sum_assignment(m)
	for i, j in zip(row_ind, col_ind):
		alignDict[gvec[i]] = ivec[j]
		simDict[gvec[i]] = m[i,j]*-1
		align.write("Overall similarity: "+str(m[i, j]*-1)+"\n")
		align.write("GICS: "+str(gvec[i])+", ICB: "+str(ivec[j])+"\n")
		align.write("en: GICS "+gics[gvec[i]]+", ICB "+icb[ivec[j]]+"\n")
		align.write("\n")
	getResults(alignDict, lang)
	align.close()
	return alignDict, simDict

'''
Method to compare the obtained alignments in all four languages and retrieve the one ICB element for each GICS element
that obtained the highest similarity score across all for languages.
'''
def alignAllLanguages(align_en, sim_en, align_it, sim_it, align_de, sim_de, align_es, sim_es, gics_en, icb_en):	
	results = {}
	mostCommon = 0
	highSim = 0
	align = open(OUTPUTPATH+EMBEDDINGNAME+"/alignment_allLanguages.txt", "w")
	for key, value in gics_en.items():
		aligner = list()
		sims = list()
		if key in align_en.keys():
			aligner.append(align_en[key])
			sims.append(sim_en[key])
		if key in align_it.keys():
			aligner.append(align_it[key])
			sims.append(sim_it[key])
		if key in align_de.keys():
			aligner.append(align_de[key])
			sims.append(sim_de[key])
		if key in align_es.keys():
			aligner.append(align_es[key])
			sims.append(sim_es[key])
		if aligner:
			print(key, value)
			print(aligner, sims)
			c = Counter(aligner)
			most_common = c.most_common(1)
			if most_common[0][1] > 1:
				results[key] = most_common[0][0]
				mostCommon += 1
			else:
				results[key] = aligner[sims.index(max(sims))]
				highSim += 1
			results[key] = aligner[sims.index(max(sims))]
			align.write("Overall similarity: "+str(max(sims))+"\n")
			align.write("GICS: "+str(key)+", ICB: "+str(aligner[sims.index(max(sims))])+"\n")
			align.write("en: GICS "+gics_en[key]+", ICB "+icb_en[aligner[sims.index(max(sims))]]+"\n")
			align.write("\n")
	align.close()
	print("Number of most common target ID matching: ", mostCommon, "Number of matching based on highest sim value: "+highSim)
	getResults(results, "all")

'''
This method loads the previously stored input data - the program to store the input data dynamically 
and read the excel input is "loadData.py"
'''
def loadData(taxonomy, language):
	taxonomy = pickle.load(open(INPUTPATH+taxonomy+"_"+language+".pkl", "rb"))
	return taxonomy

''''
Main method to show how individual embedding libraries (which need to be downloaded first - see top of this file)
can be retrieved and compared based on the two industry classifications
'''
def main():
	'''Load data and stored embeddings for each language - to store the embeddings again, the "storeEmbeddings" lines
	need to be uncommented below'''
	gics_en = loadData("gics", "en")
	icb_en = loadData("icb", "en")

	gics_it = loadData("gics", "it")
	icb_it = loadData("icb", "it")

	gics_de = loadData("gics", "de")
	icb_de = loadData("icb", "de")
	
	'''This method only loads the already decomposed German strings and replaces them in the 
	data structure - the program for decomposing German compounds is a separate tool'''
	gics_de, icb_de = replaceGermanCompounds(gics_de, icb_de)

	gics_es = loadData("gics", "es")
	icb_es = loadData("icb", "es")

	'''This program has been designed to run the individual embeddings one after another changing 
	the global variables at the beginning of this file. If you wish to run all at once, you need 
	to rename the variables below'''

	# '''Jaccard baseline: These lines get the Jaccard distance matrix for the two input data sets for each language'''
	# jaccMatrix_en, gicsIndex_en, icbIndex_en = getEditDistance(gics_en, icb_en)
	# jaccMatrix_it, gicsIndex_it, icbIndex_it = getEditDistance(gics_it, icb_it)
	# jaccMatrix_de, gicsIndex_de, icbIndex_de = getEditDistance(gics_de, icb_de)
	# jaccMatrix_es, gicsIndex_es, icbIndex_es = getEditDistance(gics_es, icb_es)

	# align_en, sim_en = getAlignment(jaccMatrix_en, gicsIndex_en, icbIndex_en, gics_en, icb_en, "en")	
	# align_it, sim_it = getAlignment(jaccMatrix_it, gicsIndex_it, icbIndex_it, gics_it, icb_it, "it")
	# align_de, sim_de = getAlignment(jaccMatrix_de, gicsIndex_de, icbIndex_de, gics_de, icb_de, "de")
	# align_es, sim_es = getAlignment(jaccMatrix_es, gicsIndex_es, icbIndex_es, gics_es, icb_es, "es")	

	# alignAllLanguages(align_en, sim_en, align_it, sim_it, align_de, sim_de, align_es, sim_es, gics_en, icb_en)


	'''Word2Vec: Outcomment to run the alignment with word2Vec embeddings'''
	#Word2Vec Google News
	#M_en, gvec_en, ivec_en = getWord2VecAlignment(gics_en, icb_en, "en", "english")
	#align_en, sim_en = getAlignment(M_en, gvec_en, ivec_en, gics_en, icb_en, "en")

	#Wikipedia
	embedDict_en = getW2VEmbedDict(gics_en, icb_en, "en", "english")
	M_en, gvec_en, ivec_en = getEmbeddingAlignment(gics_en, icb_en, "en", "english", embedDict_en)
	align_en, sim_en = getAlignment(M_en, gvec_en, ivec_en, gics_en, icb_en, "en")

	embedDict_it = getW2VEmbedDict(gics_it, icb_it, "it", "italian")
	M_it, gvec_it, ivec_it = getEmbeddingAlignment(gics_it, icb_it, "it", "italian", embedDict_it)
	align_it, sim_it = getAlignment(M_it, gvec_it, ivec_it, gics_it, icb_it, "it")
	
	embedDict_de = getW2VEmbedDict(gics_de, icb_de, "de", "german")
	M_de, gvec_de, ivec_de = getEmbeddingAlignment(gics_de, icb_de, "de", "german", embedDict_de)
	align_de, sim_de =  getAlignment(M_de, gvec_de, ivec_de, gics_de, icb_de, "de")

	embedDict_es = getW2VEmbedDict(gics_es, icb_es, "es", "spanish")
	M_es, gvec_es, ivec_es= getEmbeddingAlignment(gics_es, icb_es, "es", "spanish", embedDict_es)
	align_es, sim_es = getAlignment(M_es, gvec_es, ivec_es, gics_es, icb_es, "es")

	alignAllLanguages(align_en, sim_en, align_it, sim_it, align_de, sim_de, align_es, sim_es, gics_en, icb_en)


	# '''FastText: Outcomment to run the alignment with FastText embeddings'''
	# embedDict_en = getFastTextEmbeddingDict(gics_en, icb_en, "en", "english")
	# M_en, gvec_en, ivec_en = getEmbeddingAlignment(gics_en, icb_en, "en", "english", embedDict_en)
	# align_en, sim_en = getAlignment(M_en, gvec_en, ivec_en, gics_en, icb_en, "en")
	
	# embedDict_it = getFastTextEmbeddingDict(gics_it, icb_it, "it", "italian")
	# M_it, gvec_it, ivec_it = getEmbeddingAlignment(gics_it, icb_it, "it", "italian", embedDict_it)
	# align_it, sim_it = getAlignment(M_it, gvec_it, ivec_it, gics_it, icb_it, "it")
	
	# embedDict_de = getFastTextEmbeddingDict(gics_de, icb_de, "de", "german")
	# M_de, gvec_de, ivec_de = getEmbeddingAlignment(gics_de, icb_de, "de", "german", embedDict_de)
	# align_de, sim_de =  getAlignment(M_de, gvec_de, ivec_de, gics_de, icb_de, "de")

	# embedDict_es = getFastTextEmbeddingDict(gics_es, icb_es, "es", "spanish")
	# M_es, gvec_es, ivec_es = getEmbeddingAlignment(gics_es, icb_es, "es", "spanish", embedDict_es)
	# align_es, sim_es = getAlignment(M_es, gvec_es, ivec_es, gics_es, icb_es, "es")

	# alignAllLanguages(align_en, sim_en, align_it, sim_it, align_de, sim_de, align_es, sim_es, gics_en, icb_en)


	'''Polyglot: Outcomment to run the alignment with Polyglot embeddings'''
	# embedDict = getPolyglotEmbeddingDict(gics_en, icb_en, "en", "english")
	# M_en, gvec_en, ivec_en = getEmbeddingAlignment(gics_en, icb_en, "en", "english", embedDict)
	# align_en, sim_en = getAlignment(M_en, gvec_en, ivec_en, gics_en, icb_en, "en")
	
	# embedDict = getPolyglotEmbeddingDict(gics_it, icb_it, "it", "italian")
	# M_it, gvec_it, ivec_it = getEmbeddingAlignment(gics_it, icb_it, "it", "italian", embedDict)
	# align_it, sim_it = getAlignment(M_it, gvec_it, ivec_it, gics_it, icb_it, "it")
	
	# embedDict = getPolyglotEmbeddingDict(gics_de, icb_de, "de", "german")
	# M_de, gvec_de, ivec_de = getEmbeddingAlignment(gics_de, icb_de, "de", "german", embedDict)
	# align_de, sim_de =  getAlignment(M_de, gvec_de, ivec_de, gics_de, icb_de, "de")

	# embedDict = getPolyglotEmbeddingDict(gics_es, icb_es, "es", "spanish")
	# M_es, gvec_es, ivec_es = getEmbeddingAlignment(gics_es, icb_es, "es", "spanish", embedDict)
	# align_es, sim_es = getAlignment(M_es, gvec_es, ivec_es, gics_es, icb_es, "es")

	# alignAllLanguages(align_en, sim_en, align_it, sim_it, align_de, sim_de, align_es, sim_es, gics_en, icb_en)


if __name__ == '__main__':
	main()