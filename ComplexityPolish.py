import sys
sys.path.append('/home/garciacumbreras18/dist/treetagger')

import nltk
import numpy as np
import re
from treetagger import TreeTagger

class ComplexityPolish():

    def __init__(self, lang= 'pl'):
        """
        config es una lista de valores booleanos que activa o desactivan el cálculo de una medida
        config = [
            True|False,         # PUNCTUATION MARKS
            True|False,         # ARI
            True|False,         # FOG
            True|False,         # FLESCH
            True|False,         # FLESCH-KINCAID
            True|False,         # PISAREK
            ]
         Si config == None se calculan todas las métricas de complejidad soportadas
        """
        self.config = [True, True, True, True, True, True]
        self.metricsStr = ['AVERAGE PUNCTUATION MARKS', 'ARI', 'FOG', 'FLESCH', 'FLESCH-KINCAID', 'PISAREK']

        self.configExtend = [True, True, True, True]
        self.metricsStrExtend = ['MEAN WORDS', 'STD WORDS', 'MEAN SYLLABLES', 'STD SYLLABLES']
       

    def textProcessing(self, text):
        text = text.replace(u'\xa0', u' ')
        '''
        Cada token corresponde a un término (palabra, número...) o un signo de puntuación
        '''
        # patron de los tokens válidos
        pattern = r'''(?x)
              (?:[A-Z]\.)+              # permitimos abreviaturas como EE.UU., U.S.A., etc.
            | \w+(?:-\w+)*              # palabras con guiones intermedios
            | \$?\d+(?:\.\d+)?%?€?      # monedas y porcentajes, ejm: $12.40, 35%, 36.3€
            | \.\.\.                    # elipsis "..."
            | \s\s(?:\s)+               # más de dos espacios (' ', \r, \n) se considera un token, uno o dos se ignoran
            | [][.,;"'?():-_`']         # estos se consideran tokens aislados
        '''
        # extraemos los tokens desde el texto ya en minúsculas
        tokens = nltk.regexp_tokenize(text, pattern)
        self.text_tokens = tokens
        N_text_tokens = len(self.text_tokens)
        self.N_text_tokens = N_text_tokens
        #print('Tokens: ', self.N_text_tokens)
               
        # y ahora reorganizamos las oraciones a partir de los puntos aislados
        sentences = []
        ini = 0

        # Estos son los marcadores de fin de oración (el punto o nueva línea)
        sent_end = set(('.','!','?', '\n', '\r\n\r\n'))

        for i, x in enumerate(self.text_tokens):
            if x in sent_end:
                if i > ini: # para evitar oraciones con sólo el token de separación

                    # vamos añadiendo frases y eliminando el token de fin de oración
                    sentences.append(self.text_tokens[ini:i])
                ini = i+1
        self.sentences = sentences
        N_sentences = len(sentences)
        self.N_sentences = N_sentences
        #print('Sentences: ',self.sentences)
        
        N_charac=0
        for word in self.text_tokens:
            N_charac += len(word)
        self.N_charac = N_charac
        #print('The number the character is: ', self.N_charac)
        
        return self.text_tokens, self.N_text_tokens, self.sentences, self.N_sentences, self.N_charac 
    
    def punctuationMarks(self):

        N_punctuation = 0
        letters = []
        lsentences = []
        N_letters = 0
        N_syllables3 = 0
        lsyllablesentence=[]
        
        for words in self.sentences:
            lwords = []
            N_syllables = 0
        
            for w in words:
                if re.match('[a-zA-Z]|á|ó|í|ú|é', w):
                    lwords.append(w)
                    letters.append(w)
                    N_letters+=len(w)
                else:          
                    N_punctuation += 1
            lsentences.append(len(lwords))
            
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
        #print('list sentences: ',lsentences)
        self.N_words = sum(lsentences)
        #print('Number of words (N_w): ', self.N_words, '\n' )
        self.mean_words = np.mean(lsentences)
        self.std_words = np.std(lsentences)
        #print('media', np.mean(lsentences))
        #print('std', np.std(lsentences))
        
        self.words = letters         
        self.N_letters = N_letters
        self.N_punctuation = N_punctuation
     
        if self.N_words == 0:
            punctuation_over_words = 0
        else:
            punctuation_over_words = self.N_punctuation / self.N_words
            
        self.punctuation_over_words = punctuation_over_words
                
        #print('The number of letter is: ', N_letters)
        #print('The list of letter is: ', letters)
        #print('The PUNCTUATION MARKS is: ', self.N_punctuation, '\n')
        
        return self.punctuation_over_words, self.mean_words, self.std_words, self.mean_syllables, self.std_syllables, self.N_punctuation, self.words, self.N_words, self.N_letters, self.N_syllables, self.N_syllables3


    def readability(self):

        ARI = 4.71 * self.N_charac / self.N_words + 0.5 * self.N_words / self.N_sentences -21.43
        self.ARI = ARI
        #print("AUTOMATED READABILITY INDEX (ARI) = ", self.ARI, '\n')

        fogreadability = 0.4 * ( self.N_words / self.N_sentences  + 100 * self.N_syllables3 / self.N_words)
        self.fogreadability = fogreadability
        #print("FOG: ", self.fogreadability, "\n")

        fleschreadability = 206.835 - 84.6 * (self.N_syllables / self.N_words)  - 1.015  * (self.N_words  / self.N_sentences)
        self.fleschreadability = fleschreadability
        #print("Syllables:", self.N_syllables)
        #print("Sentences:", self.N_sentences)
        #print("FLESCH: ", self.fleschreadability, "\n")

        fkincaidreadability = - 15.59 + 11.8 * (self.N_syllables / self.N_words) + 0.39 * (self.N_words  / self.N_sentences)
        self.fkincaidreadability = fkincaidreadability
        #print("FLESCH-KINCAID: ", self.fkincaidreadability, "\n")
        self.fkincaidreadability = fkincaidreadability

        pisarekreadability = (self.N_words  / self.N_sentences)/3 + self.N_syllables3/3 +1
        self.pisarekreadability = pisarekreadability
        #print("PISAREK (2007): ", self.pisarekreadability, "\n")

        return self.ARI, self.fogreadability, self.fleschreadability, self.fkincaidreadability, self.pisarekreadability


    def calcMetrics(self, text):

        self.textProcessing(text)
        metrics = {}
        metricsPo = self.metricsStr

        readability = None

        for i in range(0, len(metricsPo)):

            if self.config == None or self.config[i] and metricsPo[i] == 'AVERAGE PUNCTUATION MARKS':
                punctuationmarks = self.punctuationMarks()
                metrics['AVERAGE PUNCTUATION MARKS'] = punctuationmarks[0]
            if self.config == None or self.config[i] and metricsPo[i] == 'ARI':
                readability = self.readability()
                metrics['ARI'] = readability[0]
            if self.config == None or self.config[i] and metricsPo[i] == 'FOG':
                if not readability: readability = self.readability()
                metrics['FOG'] = readability[1]
            if self.config == None or self.config[i] and metricsPo[i] == 'FLESCH':
                if not readability: readability = self.readability()
                metrics['FLESCH'] = readability[2]
            if self.config == None or self.config[i] and metricsPo[i] == 'FLESCH-KINCAID':
                if not readability: readability = self.readability()
                metrics['FLESCH-KINCAID'] = readability[3]
            if self.config == None or self.config[i] and metricsPo[i] == 'PISAREK':
                if not readability: readability = self.readability()
                metrics['PISAREK'] = readability[4]

        return metrics

    def getPOS(self, text):

        tt = TreeTagger(language='polish')
        sentences = tt.tag(text)

        pos_sentences = []
        sent = []
        for w in sentences:
            if len(w) < 3:
                continue
            if w[1].find(':') == -1:
                tag = w[1]
            else:
                tag = w[1].split(':')[0]
            if tag == 'SENT':
                pos_sentences.append(sent)
                sent = []
            else:
                sent += [tag]
        self.pos_sentences = pos_sentences

        return self.pos_sentences
        
    def calcMetricsExtend(self, text):
        """ 
        Calcula la métricas de complejidad activadas en la configuración 
        """ 
        self.textProcessing(text)
        metricsExtend = {}
        
        textprocessing =None
        punctuationmarks = None
        
        for i in range(0, len(self.metricsStrExtend)):
            
            if self.configExtend == None or self.configExtend[i] and self.metricsStrExtend[i] == 'MEAN WORDS':
                punctuationmarks = self.punctuationMarks()
                metricsExtend['MEAN WORDS'] = punctuationmarks[1]
                
            if self.configExtend == None or self.configExtend[i] and self.metricsStrExtend[i] == 'STD WORDS':
                punctuationmarks = self.punctuationMarks()
                metricsExtend['STD WORDS'] = punctuationmarks[2]
            
            if self.configExtend == None or self.configExtend[i] and self.metricsStrExtend[i] == 'MEAN SYLLABLES':
                punctuationmarks = self.punctuationMarks()
                metricsExtend['MEAN SYLLABLES'] = punctuationmarks[3]
                
            if self.configExtend == None or self.configExtend[i] and self.metricsStrExtend[i] == 'STD SYLLABLES':
                punctuationmarks = self.punctuationMarks()
                metricsExtend['STD SYLLABLES'] = punctuationmarks[4]
                
        return metricsExtend
