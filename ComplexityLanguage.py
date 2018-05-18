# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/garciacumbreras18/dist/freeling/APIs/python')

import freeling
import os
import re
from functools import reduce
import numpy as np
import scipy.stats
import math

class ComplexityLanguage():

    def __init__(self, lang):

        ## Modify this line to be your FreeLing installation directory
        FREELINGDIR = "/home/garciacumbreras18/dist/freeling"
        DATA = FREELINGDIR+"/data/"
        self.DATA = DATA

        self.lang = lang
        freeling.util_init_locale("default")

        # create language analyzer
        self.la=freeling.lang_ident(DATA+"common/lang_ident/ident.dat")

        # create options set for maco analyzer. Default values are Ok, except for data files.
        op= freeling.maco_options(lang)
        op.set_data_files( "",
           self.DATA + "common/punct.dat",
           self.DATA + self.lang + "/dicc.src",
           self.DATA + self.lang + "/afixos.dat",
           "",
           self.DATA + self.lang + "/locucions.dat",
           self.DATA + self.lang + "/np.dat",
           self.DATA + self.lang + "/quantities.dat",
           self.DATA + self.lang + "/probabilitats.dat")

        # create analyzers
        self.tk=freeling.tokenizer(self.DATA+self.lang+"/tokenizer.dat")
        #self.sp=freeling.splitter("/home/sinai/Freeling/data/"+self.lang+"/splitter.dat")
        self.sp=freeling.splitter(self.DATA+self.lang+"/splitter.dat")
        self.mf=freeling.maco(op)

        # activate mmorpho modules to be used in next call
        self.mf.set_active_options(False, True, True, True,  # select which among created
                                   True, True, False, True,  # submodules are to be used.
                                   True, True, True, True )  # default: all created submodules are used

        # create tagger, sense anotator, and parsers
        self.tg=freeling.hmm_tagger(self.DATA+self.lang+"/tagger.dat",True,2)
        self.sen=freeling.senses(self.DATA+self.lang+"/senses.dat")
        #self.parser= freeling.chart_parser(DATA+lang+"/chunker/grammar-chunk.dat")
        #self.dep=freeling.dep_txala(DATA+lang+"/dep_txala/dependences.dat", self.parser.get_start_symbol())

        """
        config es una lista de valores booleanos que activa o desactivan el cálculo de una medida
        config = [
            True|False,         # PUNCTUATION MARKS
            True|False,         # SCI
            True|False,         # ARI
            True|False,         # MU
            ]
         Si config == None se calculan todas las métricas de complejidad soportadas
        """
        self.config = [True, True, True, True]
        self.metricsStr = ['AVERAGE PUNCTUATION MARKS', 'SCI', 'ARI', 'MU']
        self.configExtend = [True, True, True, True, True]
        self.metricsStrExtend = ['MEAN WORDS', 'STD WORDS','COMPLEX SENTENCES', 'MEAN SYLLABLES', 'STD SYLLABLES']
       

    pass


    def textProcessing(self, text):
        text = text.replace(u'\xa0', u' ').replace('"', '')
        # meter todas las funciones en una patron de los tokens válidos
        #ls = sen.analyze(ls)
        sid=self.sp.open_session()
        tokens = self.tk.tokenize(text)
        #print("Tokens:", [w.get_form() for w in tokens])
        #print("hay Tokens:", len(tokens))
        ls = self.sp.split(sid,tokens,True)
        #print("After split", len(ls))
        ls = self.mf.analyze(ls)
        #print("After morpho", len(ls))
        ls = self.tg.analyze(ls)
        #print("After tagger", len(ls))
        #ls = self.parser.analyze(ls)
        #print("After parser", len(ls))
        #ls = self.dep.analyze(ls)
        #print("After dependencies", len(ls))
        self.sentences = ls
        self.N_sentences = len(ls)
        self.sp.close_session(sid)

        #print('Las oraciones: ', self.sentences)
        '''
        Filtra aquellos tokens que no sean adjetivos, verbos o sustantivos
        '''
        pos_content_sentences = []
        for sentence in self.sentences:
            ws = sentence.get_words();
            pos_content_sentences.append([w for w in ws if re.match('N.*|V.*|A.*', w.get_tag())])
        self.pos_content_sentences = pos_content_sentences

        return self.pos_content_sentences, self.sentences, self.N_sentences


    def punctuationMarks(self):
        #Solo nos interesa contar los tokens que sean signo de puntuación.
        #Number of words.
        punctuation = []
        lsentences=[]
        
        for words in self.sentences:
			lwords = []
            for w in words:
                if re.match('F.*', w.get_tag()):
                    punctuation.append(w.get_form())
                else:
                    lwords.append(w.get_form())
            lsentences.append(len(lwords))


        self.N_words = sum(lsentences)
        #print('Number of words (N_w): ', self.N_words, '\n' )
		self.mean_words = np.mean(lsentences)
        self.std_words = np.std(lsentences)
        self.N_punctuation = len(punctuation)
        self.punctuation = punctuation

        if self.N_words == 0:
            punctuation_over_words = 0
        else:
            punctuation_over_words = self.N_punctuation / self.N_words

        self.punctuation_over_words = punctuation_over_words
        #print("PUNCTUATION MARKS = ", self.N_punctuation,'\n')

        return self.punctuation_over_words, self.mean_words, self.std_words, self.N_punctuation, self.punctuation, self.N_words

    def sentenceComplexity(self):

        #Number of complex sentences
        N_cs = 0
        for sentence in self.sentences:
            previous_is_verb = False
            count = 0
            for words in sentence:
                for w in words:
                    if re.match('V.*', w.get_tag()):
                        if (previous_is_verb):
                            count += 1
                            previous_is_verb = False
                        else:
                            previous_is_verb = True
                    else:
                        previous_is_verb = False
                if count>0:
                    N_cs += 1
        self.N_cs = N_cs
        #print("Number of complex sentences: ", self.N_cs, "\n")

        ASL = self.N_words / self.N_sentences
        self.ASL = ASL
        #print("Average Sentence Length (ASL) = ", self.ASL, '\n')

        CS = self.N_cs / self.N_sentences
        self.CS = CS
        #print("Complex Sentences (CS) = ", self.CS, '\n')

        SCI = (ASL + CS)/ 2
        self.SCI = SCI
        #print("SENTENCE COMPLEXITY INDEX:(SCI) = ", self.SCI, "\n")

        return self.SCI, self.CS, self.N_cs, self.ASL


    def autoReadability(self):
        #Number of characters
        count = 0
        listwords = []
        for words in self.sentences:
            for w in words:
                if re.match('\r\n.*', w.get_tag()):
                    count +=1
                else:
                    listwords.append(w.get_form())

        self.listwords = listwords
        N_charac = 0
        for characters in self.listwords:
            N_charac += len(characters)

        self.N_charac = N_charac
        #print("Number of characters: ", self.N_charac, "\n")

        ARI = 4.71 * self.N_charac / self.N_words + 0.5 * self.N_words / self.N_sentences - 21.43
        self.ARI = ARI
        #print("AUTOMATED READABILITY INDEX (ARI) = ", self.ARI, '\n')

        return self.ARI, self.N_charac,  self.listwords


    def mureadability(self):
        
        #Number of syllables and Number of words with 3 or more syllables:tagger
        N_syllables3 = 0
        punctuation = []
        lsyllablesentence=[]
        for words in self.sentences:
            lwords = []
            N_syllables = 0
            for w in words:
                if re.match('F.*', w.get_tag()):
                    punctuation.append(w.get_form())
                else:
                    lwords.append(w.get_form())
            #print('lwords', lwords)
            
            for words in lwords:
                count=0
                for character in words:
                    if re.match('a|e|i|o|u|y', character):
                        N_syllables+=1
                        count+=1
                if count>=3:
                    N_syllables3+= 1
                    
            lsyllablesentence.append(N_syllables)
            #print('lsyllablesentence', lsyllablesentence)
        
        self.N_syllables = sum(lsyllablesentence)
        self.N_syllables3 = N_syllables3
        self.mean_syllables = np.mean(lsyllablesentence)
        self.std_syllables = np.std(lsyllablesentence)
        #print('media', self.mean_syllables)
        #print('std', self.std_syllables)

        #Number of letters
        listwords = []
        for words in self.sentences:
            for w in words:
                if re.match('F.*', w.get_tag()):
                    punctuation.append(w.get_form())
                else:
                    listwords.append(w.get_form())
        
        N_letters= 0
        letters = []
        vecletters =[]
        for word in listwords:
                if re.match('[a-zA-Z]|á|ó|í|ú|é', word):
                    letters.append(word)
                    N_letters+=len(word)
                    vecletters.append(len(word))
                    
        self.letters = letters
        self.N_letters = N_letters
        self.vecletters = vecletters
        
        x=self.N_letters / self.N_words
        varianza=np.var(self.vecletters)
        
        mu = (self.N_words /(self.N_words-1))*(x/varianza)*100
        #print("READABILITY MU: ", mu, "\n")
        self.mu = mu
      
        return  self.mu,self.mean_syllables, self.std_syllables, self.N_syllables, self.N_syllables3, self.letters, self.N_letters, self.vecletters

    def calcMetrics(self, text):
        """ 
        Calcula la métricas de complejidad activadas en la configuración 
        """ 
        self.textProcessing(text)
        metrics = {}
        
        punctuationMarks = None
        autoreadability = None
        sentencecomplexity = None
        mureadability= None
       
        for i in range(0, len(self.metricsStr)):
            
            if self.config == None or self.config[i] and self.metricsStr[i] == 'AVERAGE PUNCTUATION MARKS':
                punctuationmarks = self.punctuationMarks()
                metrics['AVERAGE PUNCTUATION MARKS'] = punctuationmarks[0]
            if self.config == None or self.config[i] and self.metricsStr[i] == 'SCI':
                sentencecomplexity= self.sentenceComplexity()
                metrics['SCI'] = self.SCI
            if self.config == None or self.config[i] and self.metricsStr[i] == 'ARI':
                autoreadability = self.autoReadability()
                metrics['ARI'] = autoreadability[0]
            if self.config == None or self.config[i] and self.metricsStr[i] == 'MU':
                mureadability = self. mureadability()
                metrics['MU'] = mureadability[0]
                      
        return metrics 
    
    def calcMetricsExtend(self, text):
        """ 
        Calcula la métricas de complejidad activadas en la configuración 
        """ 
        self.textProcessing(text)
        metricsExtend = {}
        
        punctuationmarks = None
        sentencecomplexity = None
        mureadability= None
        
        for i in range(0, len(self.metricsStrExtend)):
            
            if self.configExtend == None or self.configExtend[i] and self.metricsStrExtend[i] == 'MEAN WORDS':
                punctuationmarks = self.punctuationMarks()
                metricsExtend['MEAN WORDS'] = punctuationmarks[1]
                
            if self.configExtend == None or self.configExtend[i] and self.metricsStrExtend[i] == 'STD WORDS':
                punctuationmarks = self.punctuationMarks()
                metricsExtend['STD WORDS'] = punctuationmarks[2]
            
            if self.configExtend == None or self.configExtend[i] and self.metricsStrExtend[i] == 'COMPLEX SENTENCES':
                sentencecomplexity= self.sentenceComplexity()
                metricsExtend['COMPLEX SENTENCES'] = sentencecomplexity[1]
            
            if self.configExtend == None or self.configExtend[i] and self.metricsStrExtend[i] == 'MEAN SYLLABLES':
                mureadability = self. mureadability()
                metricsExtend['MEAN SYLLABLES'] = mureadability[1]
                
            if self.configExtend == None or self.configExtend[i] and self.metricsStrExtend[i] == 'STD SYLLABLES':
                mureadability = self. mureadability()
                metricsExtend['STD SYLLABLES'] = mureadability[2]
                
        return metricsExtend

    def getPOS(self, text):
        self.textProcessing(text)
        pos_sentences = []
        for sentence in self.sentences:
            ws = sentence.get_words()
            pos_sentences.append([w.get_tag() for w in ws])
        #print('POS',pos_sentences)
        self.pos_sentences = pos_sentences

        return self.pos_sentences
