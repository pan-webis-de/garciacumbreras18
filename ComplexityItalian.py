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

class ComplexityItalian():
            
    def __init__(self, lang = 'it'):
               
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
                          "",
                          self.DATA + self.lang + "/probabilitats.dat")

        # create analyzers
        self.tk=freeling.tokenizer(self.DATA + self.lang + "/tokenizer.dat")
        self.sp=freeling.splitter(self.DATA + self.lang + "/splitter.dat")
        self.mf=freeling.maco(op)
        
        # activate mmorpho modules to be used in next call
        self.mf.set_active_options(False, True, True, True,  # select which among created 
                                   True, True, False, True,  # submodules are to be used. 
                                   True, True, True, True )  # default: all created submodules are used        
        # create tagger
        self.tg=freeling.hmm_tagger(self.DATA+self.lang+"/tagger.dat",True,2)
        self.sen=freeling.senses(DATA+lang+"/senses.dat")
        
        """ 
        config es una lista de valores booleanos que activa o desactivan el cálculo de una medida
        config = [
            True|False,         # PUNCTUATION MARKS
            True|False,         # SCI
            True|False,         # ARI 
            True|False,         # MU
            True|False,         # Flesch-Vaca
            True|False,         # Gulpease
            ]
         Si config == None se calculan todas las métricas de complejidad soportadas
        """
        self.config = [True, True, True, True, True, True]
        self.metricsIt = ['PUNCTUATION MARKS', 'SCI', 'ARI', 'MU', 'FLESCH-VACA', 'GULPEASE']
       
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
                 
        return self.sentences, self.N_sentences
    
    def punctuationMarks(self):
        #Solo nos interesa contar los tokens que sean signo de puntuación.
        #Number of words.
        punctuation = []
        N_words = []
        for words in self.sentences:
            for w in words:
                if re.match('F.*', w.get_tag()):
                    
                    punctuation.append(w.get_form())
                else:
                    N_words.append(w.get_form())

        #print('Las palabras del texto son : ', N_words)
        self.N_words = len(N_words) 
        #print('Number of words (N_w): ', self.N_words, '\n' )
        
        self.N_punctuation = len(punctuation)
        self.punctuation = punctuation
        #print("PUNCTUATION MARKS = ", self.N_punctuation,'\n')
        
        return self.N_punctuation, self.punctuation, self.N_words
    
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
       
        ARI = 4.71 * self.N_charac / self.N_words + 0.5 * self.N_words/ self.N_sentences - 21.43
        self.ARI = ARI
        #print("AUTOMATED READABILITY INDEX (ARI) = ", self.ARI, '\n')
        
        return self.ARI, self.N_charac,  self.listwords        
     
      
    def mureadability(self):
        
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
        
        #Number of letters
        N_letters= 0
        letters = []
        vecletters =[]
        for word in self.listwords:
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
      
        return  self.mu, self.N_syllables, self.N_syllables3, self.letters, self.N_letters, self.vecletters

    def readability(self):
        
        fleschvacareadability = 206  - 65 *  (self.N_syllables / self.N_words) - (self.N_words  / self.N_sentences)
        self.fleschvacareadability = fleschvacareadability
        
        gulpeasereadability = 89  - 10 *  (self.N_letters / self.N_words) + 300 * (self.N_sentences / self.N_words )
        self.gulpeasereadability = gulpeasereadability

        return self.fleschvacareadability, self.gulpeasereadability
    
    def calcMetrics(self, text):
        """ 
        Calcula la métricas de complejidad activadas en la configuración 
        """ 
        self.textProcessing(text)
        metrics = {}
        
        
        punctuationMarks = None
        autoreadability = None
        sentencecomplexity = None
        readability = None
        
        for i in range(0, len(self.metricsIt)):
            
            if self.config == None or self.config[i] and self.metricsIt[i] == 'PUNCTUATION MARKS':
                punctuationmarks = self.punctuationMarks()
                metrics['PUNCTUATION MARKS'] = punctuationmarks[0]
            if self.config == None or self.config[i] and self.metricsIt[i] == 'SCI':
                sentencecomplexity= self.sentenceComplexity()
                metrics['SCI'] = self.SCI
            if self.config == None or self.config[i] and self.metricsIt[i] == 'ARI':
                autoreadability = self.autoReadability()
                metrics['ARI'] = autoreadability[0]
            if self.config == None or self.config[i] and self.metricsIt[i] == 'MU':
                mureadability = self. mureadability()
                metrics['MU'] = mureadability[0]
            if self.config == None or self.config[i] and self.metricsIt[i] == 'FLESCH-VACA':
                if not readability: readability = self.readability()
                metrics['FLESCH-VACA'] = readability[0]
            if self.config == None or self.config[i] and self.metricsIt[i] == 'GULPEASE':
                if not readability: readability = self.readability()
                metrics['GULPEASE'] = readability[1]
                      
        return metrics
    
    def getPOS(self, text):
        self.textProcessing(text)
        
        pos_sentences = []
        for sentence in self.sentences:
            ws = sentence.get_words();
            pos_sentences.append([w.get_tag() for w in ws])
        self.pos_sentences = pos_sentences
           
        return self.pos_sentences

       
 
     

     
     
