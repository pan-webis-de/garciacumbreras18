#!/home/garciacumbreras18/anaconda3/bin/python3
#/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# Authors:
# Rocío López-Anguita (rlanguit@ujaen.es)
# Arturo Montejo-Ráez (amontejo@ujaen.es)
# Centro de Estudios Avanzados en TIC (CEATIC)
#
# Universidad de Jaén - 2018
###############################################################################

import json
import os
from ComplexityLanguage import ComplexityLanguage
from ComplexitySpanish import ComplexitySpanish
from ComplexityEnglish import ComplexityEnglish
from ComplexityFrench import ComplexityFrench
from ComplexityPolish import ComplexityPolish
from ComplexityItalian import ComplexityItalian
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse

## ----------------------------------------------------------------------------
##
## Read command line arguments
##

parser = argparse.ArgumentParser(description='PAN2018 author identificator based on POS vectors')
parser.add_argument('-i', '--input', type=str, help='input directory')
parser.add_argument('-o', '--output', type=str, help='output directory')
parser.add_argument('-n', '--ngramsize', type=int, help='maximum n-gram size', choices=[1,2,3], default=2)
parser.add_argument('-f',  '--idf', action='store_true', help='apply inverse document frequency', default=False)
args = parser.parse_args()
INPUT_DIR, OUTPUT_DIR = args.input, args.output

## ----------------------------------------------------------------------------
##
## Load of analyzers
##

print('Loading complexity analyzers for different languages...\n', flush=True)
mlComplexityText = {
    'en': ComplexityEnglish(),
    'sp': ComplexitySpanish(),
    'fr': ComplexityFrench(),
    'pl': ComplexityPolish(),
    'it': ComplexityItalian()
}

## ----------------------------------------------------------------------------
##
## Corpus loading (both, train and test data sets)
##

postf = pd.DataFrame()
labels = {}
labels_cand = []

#
# Recorremos todos los problemas
#

print('Loading collection-info.json file from', INPUT_DIR, flush=True)
with open(INPUT_DIR+'/collection-info.json', 'r') as f:
    collectionInfo = json.load(f)
    f.close()

for problem in collectionInfo:
    print('\n\nProblem: ', problem['problem-name'], flush=True)
    print('Language: ', problem['language'], flush=True)

    #
    # Cargamos la clase para el cálculo de la complejidad del idioma correspondiente
    #
    complexityText = mlComplexityText[problem['language']]

    #
    # Recorremos todos los candidatos
    #
    print("Loading problem data...\n", flush=True)
    with open(INPUT_DIR + '/' + problem['problem-name'] + '/problem-info.json', 'r') as problem_info_fhd:
        problem_info= json.load(problem_info_fhd)
        problem_info_fhd.close()

    #
    # Leemos los textos de autoría conocida (TEXTOS DE ENTRENAMIENTO)
    #
    print("Loading training data")
    for candidate in problem_info['candidate-authors']:

        print('Candidate: ', candidate['author-name'], flush=True)

        files = os.listdir(os.path.join(INPUT_DIR, problem['problem-name'], candidate['author-name']))

        probcand = problem['problem-name'] + candidate['author-name']
        if not probcand in labels:
            labels[probcand] = len(labels)
            labels_cand += [probcand]

        #
        # Procesamos todo los textos de ese candidato
        #
        for i, nameFile in enumerate(files):
            print('Reading text file: ', nameFile, flush=True)

            with open(os.path.join(os.path.join(INPUT_DIR,problem['problem-name'], candidate['author-name']), nameFile),'r') as fhnd:
                postags = complexityText.getPOS(fhnd.read())
                fhnd.close()
                postags = " ".join([" ".join(p) for p in postags])
                dfi = pd.DataFrame({'Pos': postags}, index=[i])
                dfi['problem'] = problem['problem-name']
                dfi['language'] = problem['language']
                dfi['candidate'] = candidate['author-name']
                dfi['label'] = labels[probcand]
                dfi['filename'] = nameFile
                postf = postf.append([dfi])
    #
    # Si existe ground-truth, lo leemos para conocer los candidatos
    #
    unknown_candidates = False
    if os.path.isfile(INPUT_DIR +'/'+ problem['problem-name'] + '/ground-truth.json'):
        print("Reading ground truth...", flush=True)
        with open(INPUT_DIR +'/'+ problem['problem-name'] + '/ground-truth.json', 'r') as fhnd:
            ground_truth = json.load(fhnd)
            fhnd.close()
            unknown_candidates = {}
            for item in ground_truth['ground_truth']:
                unknown_candidates[item['unknown-text']] = item['true-author']

    #
    # Recorremos archivos sin etiquetar (TEXTOS DE TEST)
    #
    print("Loading test data", flush=True)

    for i, unknown_file in enumerate(os.listdir(os.path.join(INPUT_DIR, problem['problem-name'], problem_info['unknown-folder']))):
        print("Analyzing file", unknown_file, flush=True)
        with open(INPUT_DIR + '/' + problem['problem-name'] + '/' + problem_info['unknown-folder'] + '/' + unknown_file, 'r') as fhnd:
            postags = complexityText.getPOS(fhnd.read())
            fhnd.close()
            postags = " ".join([" ".join(p) for p in postags])
            dfi = pd.DataFrame({'Pos': postags}, index=[i])
            dfi['problem'] = problem['problem-name']
            dfi['language'] = problem['language']
            if unknown_candidates and unknown_candidates[unknown_file]:
                probcand = problem['problem-name'] + unknown_candidates[unknown_file]
                dfi['candidate'] = unknown_candidates[unknown_file]
                dfi['label'] = labels[probcand]

            else:
                dfi['candidate'] = None
                dfi['label'] = None
            dfi['filename'] = unknown_file
            postf = postf.append([dfi])

## ----------------------------------------------------------------------------
##
## Training and classification
##

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

from sklearn import svm
clf = svm.LinearSVC(C=1)
for problem in set(postf['problem']):

    answers = []

    print('------- Training and classifying ', problem, flush=True)

    #
    # Calculamos el modelo de espacio vectorial
    #
    tfidfVectorizer = TfidfVectorizer(ngram_range=(1, args.ngramsize), use_idf=args.idf, norm='l2')
    postf['POStfidf'] = list(tfidfVectorizer.fit_transform(postf['Pos']))


    #
    # Para el train cogemos los textos conocidos
    #
    train = postf[postf['filename'].str.contains(r"\bknown", regex=True)]
    train = train.loc[train['problem'] == problem]
    train = train.dropna(axis=1, how='any')
    train_target = train['label']
    train_data = np.array(list(train['POStfidf'].apply(lambda x: x.toarray()[0])))

    #
    # Para el test cogemos los textos desconocidos
    #
    test = postf[postf['filename'].str.contains(r"\bunknown", regex=True)]
    test = test.loc[test['problem'] == problem]
    test = test.dropna(axis=1, how='any')
    test_target = test['label']
    test_data = np.array(list(test['POStfidf'].apply(lambda x: x.toarray()[0])))

    # Entrenamos con los textos con candidatos conocidos y predecimos con los datos desconocidos
    y_pred = clf.fit(train_data, train_target).predict(test_data)

    for index, row in test.iterrows():
        probcand = labels_cand[y_pred[index]]
        answers.append({
            'unknown-text': row['filename'],
            'predicted-author': probcand[probcand.find("candidate"):],
        })

    with open(OUTPUT_DIR + '/answers-' + problem +'.json', 'w') as file:
        json.dump(answers, file, indent=4)


print("done!")
