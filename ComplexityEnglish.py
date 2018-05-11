import sys
sys.path.append('/home/garciacumbreras18/dist/freeling/APIs/python')
from ComplexityLanguage import ComplexityLanguage
import re
import math
from functools import reduce
import freeling
import numpy as np

class ComplexityEnglish(ComplexityLanguage):  
    
    def __init__(self):
        ComplexityLanguage.__init__(self,'en')
        
        # create parsers
        self.parser= freeling.chart_parser(self.DATA+self.lang+"/chunker/grammar-chunk.dat")
        self.dep=freeling.dep_txala(self.DATA+self.lang+"/dep_txala/dependences.dat", self.parser.get_start_symbol())
        
        """
        config es una lista de valores booleanos que activa o desactivan el cálculo de una medida
        config = [
            True|False,         # MAXIMUN EMBEDDING DEPTH OF SENTENCE (MaxDEPTH)
            True|False,         # MINIMUN EMBEDDING DEPTH OF SENTENCE (MinDEPTH)
            True|False,         # AVERAGE EMBEDDING DEPTH OF SENTENCE (MeanDEPTH)
            True|False,         # FOG
            True|False,         # FLESCH
            True|False,         # FLESCH-KINCAID
            True|False,         # SMOG
            ]
        """
        self.config += [True, True, True, True, True, True, True, True]
        self.metricsStr.extend(['MaxDEPTH','MinDEPTH', 'MeanDEPTH', 'StdDEPTH', 'FOG', 'FLESCH', 'FLESCH-KINCAID', 'SMOG'])
        
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
        ls = self.parser.analyze(ls)
        #print("After parser", len(ls))
        ls = self.dep.analyze(ls)
        #print("After dependencies", len(ls))
        self.sentences = ls
        #print("oraciones con split:", len(ls))
        self.N_sentences = len(ls)
       
        self.sp.close_session(sid)      
           
        return self.sentences, self.N_sentences
    
    def getDepth(self, ptree, depth=0):
               
        node = ptree.begin()
        info = node.get_info()
        nch = node.num_children()

        if (nch == 0) :
            return depth
        else :
            child_depth = []
            for i in range(nch) :
                child = node.nth_child_ref(i)
                child_depth.append(self.getDepth(child, depth+1))
            return max(child_depth)  
          
    def embeddingDepth(self):
        ##output results
        max_list = []
        for s in self.sentences:
            tr = s.get_parse_tree()
            max_list.append(self.getDepth(tr,0))

        #print('Longitud mi lista es:', len(max_list))
        #print('Mi lista es:', max_list)
        
        self.max_list = max_list
        mean_max_list = sum(max_list)/float(len(max_list))
        max_max_list = max(max_list)
        min_max_list = min(max_list)
        std_max_list= np.std(max_list)
        
        #print('MAXIMUN EMBEDDING DEPTH OF SENTENCE (MaxDEPTH): ', max_max_list, '\n')
        #print('MINIMUN EMBEDDING DEPTH OF SENTENCE (MinDEPTH): ', min_max_list, '\n')
        #print('AVERAGE EMBEDDING DEPTH OF SENTENCE (MeanDEPTH): ', mean_max_list, '\n')
        #print('STANDARD DEVIATION: ', std_max_list)
        
        
        #lin=sys.stdin.readline()
        self.max_max_list = max_max_list
        self.min_max_list = min_max_list
        self.mean_max_list = mean_max_list
        self.std_max_list = std_max_list
        
        return self.max_max_list, self.min_max_list, self.mean_max_list, self.std_max_list
    
    def readability(self):
        
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
        #print("sílabas: ", self.N_syllables)
        
        fogreadability = 0.4 * ( self.N_words / self.N_sentences  + 100 * self.N_syllables3 / self.N_words)
        #print("FOG: ", fogreadability, "\n")
        self.fogreadability = fogreadability
        
        fleschreadability = 206.835 - 84.6 * (self.N_syllables / self.N_words)  - 1.015  * (self.N_words  / self.N_sentences) 
        #print("FLESCH: ", fleschreadability, "\n")
        self.fleschreadability = fleschreadability
        
        fkincaidreadability = - 15.59 + 11.8 * (self.N_syllables / self.N_words) + 0.39 * (self.N_words  / self.N_sentences) 
        #print("FLESCH-KINCAID: ", fkincaidreadability, "\n")
        self.fkincaidreadability = fkincaidreadability
        
        return  self.fogreadability, self.fleschreadability, self.fkincaidreadability
        
    def ageReadability(self):
        
        smogreadability= 3.1291+1.0430*math.sqrt(self.N_syllables3*(30/self.N_sentences))
        #print("READABILITY SMOG: ", smogreadability, "\n")
        self.smogreadability = smogreadability
        
        return self.smogreadability
    
    def calcMetrics(self, text):
        """ 
        Calcula la métricas de complejidad activadas en la configuración 
        Si config == None se calculan todas las métricas de complejidad soportadas
        """
        self.textProcessing(text)
        metrics = super().calcMetrics(text)
        metricsEn = self.metricsStr
        embdep = None
        readability = None
                
        for i in range(len(metrics)-1, len(metricsEn)):  
            
            if self.config == None or self.config[i] and metricsEn[i] == 'MaxDEPTH':
                embdep = self.embeddingDepth()
                metrics['MaxDEPTH'] = embdep[0]
            if self.config == None or self.config[i] and metricsEn[i] == 'MinDEPTH':
                if not embdep: embdep = self.embeddingDepth()
                metrics['MinDEPTH'] = embdep[1]
            if self.config == None or self.config[i] and metricsEn[i] == 'MeanDEPTH':
                if not embdep: embdep = self.embeddingDepth()
                metrics['MeanDEPTH'] = embdep[2]
            if self.config == None or self.config[i] and metricsEn[i] == 'StdDEPTH':
                if not embdep: embdep = self.embeddingDepth()
                metrics['StdDEPTH'] = embdep[3]
            if self.config == None or self.config[i] and metricsEn[i] == 'FOG':
                readability = self.readability()
                metrics['FOG'] = readability[0]
            if self.config == None or self.config[i] and metricsEn[i] == 'FLESCH':
                if not readability: readability = self.readability()
                metrics['FLESCH'] = readability[1]
            if self.config == None or self.config[i] and metricsEn[i] == 'FLESCH-KINCAID':
                if not readability: readability = self.readability()
                metrics['FLESCH-KINCAID'] = readability[2]
            if self.config == None or self.config[i] and metricsEn[i] == 'SMOG':
                metrics['SMOG'] = self.ageReadability()
        
             
        return metrics 
 
