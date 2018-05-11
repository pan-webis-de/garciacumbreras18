import sys
sys.path.append('/home/garciacumbreras18/dist/freeling/APIs/python')
from ComplexityLanguage import ComplexityLanguage
import freeling
import os
import re
from functools import reduce
import numpy as np
import scipy.stats
import math

class ComplexityFrench(ComplexityLanguage):
    
     def __init__(self):
        
        lang = 'fr'
        ComplexityLanguage.__init__(self, lang)
       
        ## Modify this line to be your FreeLing installation directory
        FREELINGDIR = "/home/garciacumbreras18/dist/freeling"
        DATA = FREELINGDIR+"/data/"
        CLASSDIR = ""
        self.lang = lang
        freeling.util_init_locale("default")

        # create language analyzer
        self.la=freeling.lang_ident(DATA+"common/lang_ident/ident.dat")

        # create options set for maco analyzer. Default values are Ok, except for data files.
        op= freeling.maco_options(lang)
        op.set_data_files( "", 
           DATA + "common/punct.dat",
           DATA + lang + "/dicc.src",
           DATA + lang + "/afixos.dat",
           "",
           DATA + lang + "/locucions.dat", 
           DATA + lang + "/np.dat",
           DATA + lang + "/quantities.dat",
           DATA + lang + "/probabilitats.dat")

        # create analyzers
        self.tk=freeling.tokenizer(DATA+lang+"/tokenizer.dat")
        self.sp=freeling.splitter(DATA+lang+"/splitter.dat")        
        self.mf=freeling.maco(op)

        # activate mmorpho modules to be used in next call
        self.mf.set_active_options(False, True, True, True,  # select which among created 
                                   True, True, False, True,  # submodules are to be used. 
                                   True, True, True, True )  # default: all created submodules are used     
        
        # create tagger and sense anotator
        self.tg=freeling.hmm_tagger(DATA+lang+"/tagger.dat",True,2)
        self.sen=freeling.senses(DATA+lang+"/senses.dat")
       
     
        f = open(CLASSDIR + '/home/garciacumbreras18/DaleChall.txt')
        lines = f.readlines()
        f.close()

        listDaleChall = []
        for l in lines: 
            data = l.strip().split()
            listDaleChall += data
        self.listDaleChall=listDaleChall  
        """
        config es una lista de valores booleanos que activa o desactivan el cálculo de una medida
        config = [
            True|False,         # KANDEL MODELS
            True|False,         # DALE CHALL
            True|False,         # SOL
            ]
        """
        self.config += [True, True, True]
        self.metricsStr.extend(['KANDEL-MODELS','DALE CHALL', 'SOL'])
    
     def readability(self):
            
        #Number of low frequency words   
        count = 0
        for sentence in self.pos_content_sentences:
            for w in sentence:
                if w.get_form() not in self.listDaleChall:
                    count+=1
        N_difficultwords = count
        
        #Number of syllables and Number of words with 3 or more syllables:tagger
        N_syllables = 0
        N_syllables3 = 0
        for words in self.listwords:
            count=0
            for character in words:
                if re.match('a|e|i|o|u|y', character):
                    N_syllables +=1
                    count+=1
            if count>=3:
                N_syllables3 += 1
                                  
        self.N_syllables = N_syllables
        self.N_syllables3 = N_syllables3
        
           
        kandelmodelsreadability = 207 - 1.015 * (self.N_words  / self.N_sentences) - 73.6 *  (self.N_syllables / self.N_words) 
        #print("KANDEL-MODELS: ", kandelmodelsreadability, "\n")
        self.kandelmodelsreadability = kandelmodelsreadability
        
        dalechallreadability =15.79 * (N_difficultwords / self.N_words) + 0.04906 *  (self.N_words / self.N_sentences) 
        #print("DALE CHALL: ", dalechallreadability, "\n")
        self.dalechallreadability = dalechallreadability
        
        return self.kandelmodelsreadability, self.dalechallreadability
    
     def ageReadability(self):
                        
        solreadability= - 1.35 + 0.77 * (3.1291 + 1.0430 * math.sqrt(self.N_syllables3 * (30/self.N_sentences)))
        #print("READABILITY SOL: ", solreadability, "\n")
        self.solreadability = solreadability
        
        return self.solreadability
    
     def calcMetrics(self, text):
        """ 
        Calcula la métricas de complejidad activadas en la configuración 
        Si config == None se calculan todas las métricas de complejidad soportadas
        """
        self.textProcessing(text)
        metrics = super().calcMetrics(text)      
        metricsFr = self.metricsStr
        
        readability = None
        
        for i in range(len(metrics)-1, len(metricsFr)):
            
            if self.config == None or self.config[i] and metricsFr[i] == 'KANDEL MODELS':
                readability = self.readability()
                metrics['KANDEL-MODELS'] = readability[0]
            if self.config == None or self.config[i] and metricsFr[i] == 'DALE CHALL':
                if not readability: readability = self.readability()
                metrics['DALE CHALL'] = readability[1]
            if self.config == None or self.config[i] and metricsFr[i] == 'SOL':
                metrics['SOL'] = self.ageReadability()
                
        return metrics 
    
    
