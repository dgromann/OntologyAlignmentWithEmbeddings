# -*- coding: utf-8 -*-
#
#
# Author: Dagmar Gromann
# Description: Different similarity measures tested and used for the multilingual ontology alignment approach

import numpy as np
from numpy.linalg import norm

import operator
import math

from math import*
from decimal import Decimal

from scipy.stats import entropy
from sklearn.preprocessing import normalize
from sklearn.metrics import jaccard_similarity_score

EVALUATIONFILE = "pathToEvauluationFIle"

def getJensenShanon(gics, icb):
    JSD = []
    gics_vec = []
    #obtain edit distance 
    for key, value in gics.items():
        print(value)
        if key > 10101010 and str(key)[0:4] != "4040" and key != 25502010:
            vectorJSD = np.zeros(len(icb))
            gics_vec.append(key)
            icbs_vec = []
            for key1, value1 in icb.items():
                print(value1)
                if str(key1)[len(key1)-1:] != "0":
                    icbs_vec.append(key1)
                    sim1 = JSD(value, value1)
                    '''alternative exp(-JSD(value, value1))'''
                    vectorJSD[icbs_vec.index(key1)] = sim1
            JSD.append(vectorJSD)
    M = np.array(JSD)
    return M, JSD, gics_vec, icbs_vec

def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def square_rooted(x):
    return round(sqrt(sum([a*a for a in x])),3)
 
def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)
 
def l2norm(P,Q):
    if P.all() == 0 or Q.all() == 0:
        return 0
    diff = set(P) - set(Q)
    diff = sum(abs(x) for x in diff)
    a = math.sqrt(sum(pow(abs(x),2) for x in P))
    b = math.sqrt(sum(pow(abs(x),2) for x in Q))
    means = math.sqrt((a+b)*0.5)
    result = diff/means
    #print(result/len(P))
    return result

def l2norm_old(P,Q):
    a = math.sqrt(sum(pow(abs(x),2) for x in P))
    b = math.sqrt(sum(pow(abs(x),2) for x in Q))
    anb = abs(sum(x*y for x,y in zip(P,Q)))
    aub=abs(a+b-anb)
    result = anb / aub
    return result/len(P)

def nth_root(value, n_root):
    root_value = 1/float(n_root)
    return round (Decimal(value) ** Decimal(root_value),3)

def minkowski_distance(x,y,p_value):
    return nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),p_value)

'''
This method calculates the Jaccard similarity between two input strings 
'''
def jaccard_similarity(x,y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

'''
This method reads the input data and returns a matrix with the Jaccard similarity between 
the individual labels for each langauge 
'''
def getEditDistance(gics, icb):
    jaccMatrix = []
    gics_vec = []
    evalIDs = readEvalFile()
    #obtain edit distance 
    for key, value in gics.items():
        if str(key)[0:4] != "4040" and key != 25502010 and key in evalIDs:
            vectorJac = np.zeros(len(icb))
            gics_vec.append(key)
            icbs_vec = []
            for key1, value1 in icb.items():
                if key1 in evalIDs:
                    icbs_vec.append(key1)
                    sim1 = jaccard_similarity(value, value1)
                    vectorJac[icbs_vec.index(key1)] = sim1
            jaccMatrix.append(vectorJac)

    return jaccMatrix, gics_vec, icbs_vec

'''
Method to load manual alignment IDs to ensure that only those IDs are used and not the 
entire set of IDs and labels of both resources
'''
def readEvalFile():
    evalFile = open(EVALUATIONFILE, "r")
    evalIDs = list()
    for row in evalFile:
        ids = row.split(" ")
        evalIDs.append(int(ids[0].strip()))
        evalIDs.append(ids[1].strip())
    evalFile.close()
    return evalIDs