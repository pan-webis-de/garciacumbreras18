from ComplexityLanguage import ComplexityLanguage
import re
import math
from functools import reduce
import freeling
import numpy as np

class ComplexitySpanish(ComplexityLanguage):
    
    def __init__(self):
              
        ComplexityLanguage.__init__(self, 'es')    
        ## Modify this line to be your FreeLing installation directory
        
        # create parsers
        self.parser= freeling.chart_parser(self.DATA+self.lang+"/chunker/grammar-chunk.dat")
        self.dep=freeling.dep_txala(self.DATA+self.lang+"/dep_txala/dependences.dat", self.parser.get_start_symbol())
        
        # Para leer el texto que introducimos
        CLASSDIR = "/home/garciacumbreras18/"
                
        f = open(CLASSDIR + 'CREA_total.txt')
        lines = f.readlines()
        f.close()
        crea = {}
        for l in lines[1:1000]: # those words not in the 1000 most frequent words in CREA are low frequency words
            data = l.strip().split()
            crea[data[1]] = float(data[2].replace(',', ''))
        self.crea = crea
        
        """
        config es una lista de valores booleanos que activa o desactivan el cálculo de una medida
        config = [
            True|False,         # MAXIMUN EMBEDDING DEPTH OF SENTENCE (MaxDEPTH)
            True|False,         # MINIMUN EMBEDDING DEPTH OF SENTENCE (MinDEPTH)
            True|False,         # AVERAGE EMBEDDING DEPTH OF SENTENCE (MeanDEPTH)
            True|False,         # LC
            True|False,         # SSR
            True|False,         # HUERTA
            True|False,         # IFSZ
            True|False,         # POLINI
            True|False,         # MINIMUN AGE
            True|False,         # SOL
            True|False,         # CRAWFORD
            ]
        """
        self.config += [True, True, True, True, True, True, True, True, True, True, True, True]
        self.metricsStr.extend(['MaxDEPTH','MinDEPTH', 'MeanDEPTH', 'StdDEPTH', 'LC','SSR', 'HUERTA', 'IFSZ', 'POLINI', 'MINIMUN AGE', 'SOL', 'CRAWFORD'])
        
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
        
        mean_max_list = sum(max_list)/float(len(max_list))
        max_max_list = max(max_list)
        min_max_list = min(max_list)
        std_max_list= np.std(max_list)


        #print('MAXIMUN EMBEDDING DEPTH OF SENTENCE (MaxDEPTH): ', max_max_list, '\n')
        #print('MINIMUN EMBEDDING DEPTH OF SENTENCE (MinDEPTH): ', min_max_list, '\n')
        #print('AVERAGE EMBEDDING DEPTH OF SENTENCE (MeanDEPTH): ', mean_max_list, '\n')
        #print('Desviación típica : ', std_max_list)
        
        #lin=sys.stdin.readline()
        self.max_max_list = max_max_list
        self.min_max_list = min_max_list
        self.mean_max_list = mean_max_list
        self.std_max_list = std_max_list
        
        return self.max_max_list,  self.min_max_list, self.mean_max_list, self.std_max_list
    
    def lexicalComplexity(self):
        #Number of low frequency words   
        count = 0
        for sentence in self.pos_content_sentences:
            for w in sentence:
                if w.get_form() not in self.crea:
                    count+=1
        N_lfw = count
        self.N_lfw = N_lfw
        #print("Number of low frequency words (N_lfw): ", self.N_lfw, "\n")
        #Number of distinct content words 
        N_dcw = len(set([w.get_form().lower() for s in self.pos_content_sentences for w in s]))
        self.N_dcw =N_dcw
        #print('Number of distinct content words (N_dcw) = ', self.N_dcw, '\n')
        #Number of sentences
        N_sentences = len(self.pos_content_sentences)
        self.N_sentences = N_sentences
        #print("Number os sentences (N_s): ", self.N_sentences, "\n")
        #Number of total content words
        N_cw = reduce((lambda x, y: x + y), [len(s) for s in self.pos_content_sentences])
        self.N_cw = N_cw
        #print("Number of total content words (N_cw): ", self.N_cw, "\n")
        #Lexical Distribution Index
        LDI = N_dcw / float(self.N_sentences)
        self.LDI = LDI
        #print("Lexical Distribution Index (LDI) = ", self.LDI, '\n')
        #Index of Low Frequency Words
        ILFW = N_lfw / float(N_cw)
        self.ILFW =ILFW
        #print("Index Low Frequency Words (ILFW) = ", self.ILFW, '\n')
        #Lexical Complexity
        LC = (LDI + ILFW) / 2
        self.LC = LC
        #print ("LEXICAL COMPLEXITY INDEX (LC) =", LC, "\n")
        
        return self.LC, self.N_lfw, self.N_cw, self.N_dcw, self.N_sentences, self.LDI, self.ILFW 
    
    def ssReadability(self): 
        #Number of rare words
        byfreq = sorted(self.crea, key=self.crea.__getitem__, reverse=True)
        byfreq = byfreq[:1500]
        count = 0
        for sentence in self.pos_content_sentences:
            for w in sentence:
                if w.get_form().lower() not in byfreq:
                    count +=1
        
        N_rw = count
        self.N_rw = N_rw
        #print("Number of rare words (N_rw): ", self.N_rw, "\n")
        
        SSR = 1.609*(self.N_words / self.N_sentences) + 331.8* (self.N_rw /self.N_words) + 22.0 
        self.SSR= SSR
        #print ("SPAULDING SPANISH READABILITY (SSR) ", self.SSR, "\n")
        
        return self.SSR, self.N_rw 
    
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
        
        huertareadability = 206.835 - 60 * (self.N_syllables / self.N_words)  - 102 * (self.N_sentences / self.N_words)
        #print("THE READABILITY OF HUERTA: ", huertareadability, "\n")
        self.huertareadability = huertareadability
        
        ifszreadability = 206.835 - 62.3 * (self.N_syllables / self.N_words)  - (self.N_words / self.N_sentences) 
        #print("THE READABILITY IFSZ: ", ifszreadability, "\n")
        self.ifszreadability = ifszreadability
                
        polinicompressibility = 95.2 - 9.7 * (self.N_letters / self.N_words)  - 0.35 * (self.N_words / self.N_sentences) 
        #print("THE COMPRESSIBILITY OF GUTIÉRREZ POLINI: ", polinicompressibility, "\n")
        self.polinicompressibility = polinicompressibility  
            
        return  self.huertareadability, self.ifszreadability, self.polinicompressibility
    
    def ageReadability(self):
        
        minimumage = 0.2495 *(self.N_words / self.N_sentences) + 6.4763 * (self.N_syllables / self.N_words) - 7.1395
        #print("MINIMUM AGE TO UNDERSTAND A TEXT: ", minimumage, "\n")
        self.minimumage = minimumage
        
        solreadability= -2.51+0.74*(3.1291+1.0430*math.sqrt(self.N_syllables3*(30/self.N_sentences)))
        #print("THE READABILITY SOL: ", solreadability, "\n")
        self.solreadability = solreadability
        
        return self.minimumage, self.solreadability
    
    def yearsCrawford(self):
        
        years = -20.5 *(self.N_sentences / self.N_words) + 4.9 * (self.N_syllables / self.N_words ) - 3.407
        #print("YEARS NEEDED: ", years, "\n")
        self.years = years
        
        return self.years
    
    def calcMetrics(self, text):
        """ 
        Calcula la métricas de complejidad activadas en la configuración 
        Si config == None se calculan todas las métricas de complejidad soportadas
        """
        self.textProcessing(text)
        metrics = super().calcMetrics(text)      
        metricsEs = self.metricsStr
        embdep = None
        readability = None
        agereadability = None
        lexicalComplexity = None
        ssreadability = None
              
        for i in range(len(metrics)-1, len(metricsEs)):
            
            if self.config == None or self.config[i] and metricsEs[i] == 'MaxDEPTH':
                embdep = self.embeddingDepth()
                metrics['MaxDEPTH'] = embdep[0]
            if self.config == None or self.config[i] and metricsEs[i] == 'MinDEPTH':
                if not embdep: embdep = self.embeddingDepth()
                metrics['MinDEPTH'] = embdep[1]
            if self.config == None or self.config[i] and metricsEs[i] == 'MeanDEPTH':
                if not embdep: embdep = self.embeddingDepth()
                metrics['MeanDEPTH'] = embdep[2]
            if self.config == None or self.config[i] and metricsEs[i] == 'StdDEPTH':
                if not embdep: embdep = self.embeddingDepth()
                metrics['StdDEPTH'] = embdep[3]
            if self.config == None or self.config[i] and metricsEs[i] == 'LC':
                lexicalComplexity = self.lexicalComplexity()
                metrics['LC'] = lexicalComplexity[0]
            if self.config == None or self.config[i] and metricsEs[i] == 'SSR':
                ssreadability = self.ssReadability() 
                metrics['SSR'] = ssreadability[0]
            if self.config == None or self.config[i] and metricsEs[i] == 'HUERTA':
                readability = self.readability()
                metrics['HUERTA'] = readability[0]
            if self.config == None or self.config[i] and metricsEs[i] == 'IFSZ':
                if not readability: readability = self.readability()
                metrics['IFSZ'] =  readability[1]
            if self.config == None or self.config[i] and metricsEs[i] == 'POLINI':
                if not readability: readability = self.readability()
                metrics['POLINI'] = readability[2]
            if self.config == None or self.config[i] and metricsEs[i] == 'MINIMUM AGE':
                agereadability = self.ageReadability()
                metrics['MINIMUM AGE'] = agereadability[0]
            if self.config == None or self.config[i] and metricsEs[i] == 'SOL':
                if not agereadability: agereadability = self.ageReadability()
                metrics['SOL'] =  agereadability[1]
            if self.config == None or self.config[i] and metricsEs[i] == 'CRAWFORD':
                metrics['CRAWFORD'] = self.yearsCrawford()
              
        return metrics 
 
