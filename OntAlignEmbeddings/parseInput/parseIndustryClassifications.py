# -*- coding: utf8 -*-
# 
# Author: Dagmar Gromann
# Description: This program loads the two classifications GICS and ICB from their excel files and stores them as pickles
#

import pickle
import numpy as np
import pandas as pd
import re

PATH_TO_ICB_EXCEL = "yourPath"
PATH_TO_GICS_EXCEL = "yourPath"
OUTPUT = "yourPath"

def loadICB(filepath, sheetName, language): 
	amp = ''
	taxonomy = dict()
	definitionsICB = dict()
	xlsx = pd.ExcelFile(filepath)
	df = xlsx.parse(sheetName, skiprows=8)

	if language == "de":
		amp = "und"
	if language == "en":
		amp = "and"
	if language == "it":
		amp = "e"
	if language == "es":
		amp = "y"

	cols = ["Industry","Supersector","Sector","Subsector","Definition"]
	df.columns = cols

	for index, row in df.iterrows():
		if not pd.isnull(row["Industry"]):
			identifier = re.findall(r'\d+', row["Industry"])
			label = str(row["Industry"]).replace(str(identifier[0]), "").replace("&", amp)
			taxonomy[identifier[0]] = label.strip().replace("\n", " ")
		if not pd.isnull(row["Supersector"]):
			identifier = re.findall(r'\d+', row["Supersector"])
			label = str(row["Supersector"]).replace(str(identifier[0]), "").replace("&", amp)
			taxonomy[identifier[0]] = label.strip().replace("\n", " ")
		if not pd.isnull(row["Sector"]):
			identifier = re.findall(r'\d+', row["Sector"])
			label = str(row["Sector"]).replace(str(identifier[0]), "").replace("&", amp)
			taxonomy[identifier[0]] = label.strip().replace("\n", " ")
		if not pd.isnull(row["Subsector"]):
			identifier = re.findall(r'\d+', row["Subsector"])
			label = str(row["Subsector"]).replace(str(identifier[0]), "").replace("&", amp)
			taxonomy[identifier[0]] = label.strip().replace("\n", " ")
		if not pd.isnull(row["Definition"]):
			identifier = re.findall(r'\d+', row["Subsector"])
			definitionsICB[identifier[0]] = row["Definition"]
	
	return 	taxonomy, definitionsICB

def loadGICS(filepath, sheetName, skiprows, language):
	amp = ''
	taxonomy = dict()
	definitionsGICS = dict()
	xlsx = pd.ExcelFile(filepath)
	df = xlsx.parse(sheetName, skiprows=skiprows)
	#data = df.ix[3:4]

	cols = ["Sector","Sector_Label","Group","Group_Label","Industry","Industry_Label","SubIndustry","SubIndustry_Label"]
	df.columns = cols

	if language == "de":
		amp = "und"
	if language == "en":
		amp = "and"
	if language == "it":
		amp = "e"
	if language == "es":
		amp = "y"
	if language == "ru":
		amp = "и"


	for index, row in df.iterrows():
		if not pd.isnull(row["Sector"]):
			taxonomy[int(row["Sector"])] = row["Sector_Label"].replace("\n", " ").replace("&", amp)
		if not pd.isnull(row["Group"]):
			taxonomy[int(row["Group"])] = row["Group_Label"].replace("\n", " ").replace("&", amp)
		if not pd.isnull(row["Industry"]):
			taxonomy[int(row["Industry"])] = row["Industry_Label"].replace("\n", " ").replace("&", amp)
		if not pd.isnull(row["SubIndustry"]):
			taxonomy[int(row["SubIndustry"])] = row["SubIndustry_Label"].replace("\n", " ").replace("&", amp)
			definitionsGICS[int(row["SubIndustry"])] = df.ix[index+1]["SubIndustry_Label"]

	return taxonomy, definitionsGICS

def store(gics, gics_def, icb, icb_def, language):
	if gics: 
		pickle.dump( gics, open(OUTPUT+"gics_"+language+".pkl", "wb") )
		pickle.dump( gics_def, open(OUTPUT+"gics_defs_"+language+".pkl", "wb") )
	if icb: 
		pickle.dump( icb, open(OUTPUT+"icb_"+language+".pkl", "wb") )
		pickle.dump( icb_def, open(OUTPUT+"icb_defs_"+language+".pkl", "wb") )

def main(): 
	#English
	gics_en, gics_defs_en = loadGICS(PATH_TO_GICS_EXCEL+"GICS Structure effective Sep 1, 2016.xls", "Effective close of Aug 31,2016", 3, "en")
	icb_en, icb_defs_en = loadICB(PATH_TO_ICB_EXCEL+"Structure_Defs_English.xlsx", "ICB - FULL LISTING", "en")
	store(gics_en, gics_defs_en, icb_en, icb_defs_en, "en")

	#Spanish 
	gics_es, gics_defs_es = loadGICS(PATH_TO_GICS_EXCEL+"gics-spanish-083116.xlsx", "Cierre de mercado 31-08-2016", 4, "es")
	icb_es, icb_defs_es = loadICB(PATH_TO_ICB_EXCEL+"Structure_Defs_Spanish.xlsx", "ICB-Todas las Definiciones", "es")
	store(gics_es, gics_defs_es, icb_es, icb_defs_es, "es")

	#Italian 
	gics_it, gics_defs_it = loadGICS(PATH_TO_GICS_EXCEL+"gics-italian-083116.xls", "Effective close of Aug 31,2016", 6, "it")
	icb_it, icb_defs_it = loadICB(PATH_TO_ICB_EXCEL+"Structure_Defs_Italian.xlsx", "ICB_Tutte le Definizioni", "it")
	store(gics_it, gics_defs_it, icb_it, icb_defs_it, "it")

	#German 
	gics_de, gics_defs_de = loadGICS(PATH_TO_GICS_EXCEL+"gics-german-083116.xls", "Effektiv nach Geschaftsschluss3", 5, "de")
	icb_de, icb_defs_de = loadICB(PATH_TO_ICB_EXCEL+"Structure_Defs_German.xlsx", "ICB - Alle Definitionen", "de")
	store(gics_de, gics_defs_de, icb_de, icb_defs_de, "de")

	#Japanese 
	gics_ja, gics_defs_ja = loadGICS(PATH_TO_GICS_EXCEL+"gics-japanese-083116.xls", "gics_map", 4, "ja")
	icb_ja, icb_defs_ja = loadICB(PATH_TO_ICB_EXCEL+"Structure_Defs_Japanese.xlsx", "ICB_JPA", "ja")
	store(gics_ja, gics_defs_ja, icb_ja, icb_defs_ja, "ja")

	#Russian 
	gics_ru, gics_defs_ru = loadGICS(PATH_TO_GICS_EXCEL+"gics-russian-083116.xls", "Effective close of Aug 31,2016", 4, "ru")
	store(gics_ru, gics_defs_ru, None, None, "ru")

if __name__ == '__main__':
	main()