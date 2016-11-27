#!/usr/bin/env python 
#coding:utf-8
import string
import json
import urllib2
import sys
from urllib2 import urlopen
import networkx as nx
import matplotlib.pyplot as plt 
import os
from mwtextextractor import get_body_text
import logging
import re
import random
import shutil
from time import time
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import SpaceTokenizer
import numpy as np
from sklearn import svm, tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier,  OneVsOneClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, hamming_loss, recall_score, confusion_matrix, classification_report, f1_score, precision_score
from sklearn.datasets import load_files
from sklearn import cross_validation
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import *
from scipy import *
import numpy as np


def construir_grafo(categorias):
  G = nx.Graph()
  for categoria in categorias:
    G.add_node(categoria)
    url = "http://en.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtype=subcat&cmtitle=Category:" + categoria + "&cmlimit=500&format=json&clshow=!hidden"
    data = urllib2.urlopen(url).read()
    data = json.loads(data)
    for raw in data['query']['categorymembers']:
      titulo = raw['title'].replace(' ', '_')
      for categ in categorias:
	if (titulo[9:] == categ):
	  G.add_edge(categoria, categ)
  return G
     
	
def escribir_archivo(nombre_archivo, ruta, contenido):
	ruta = ruta + nombre_archivo.replace("/","_")  
	f = open(ruta,'a')
	f.write((json.dumps(contenido)))
        f.close()


def descargar_paginas(categoria, ruta, min_car_example):
	url = "http://en.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtype=page&cmtitle=Category:" + categoria + "&cmlimit=500&format=json&clshow=!hidden"
	data = urllib2.urlopen(url).read()
	data = json.loads(data)
	for raw in data['query']['categorymembers']:
		titulo = raw['title'].replace(' ', '_')
		if not (os.path.exists(ruta + titulo.replace("/","_") )):
			url = "http://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&redirects=1&titles=" + titulo + "&format=json&clshow=!hidden"
			page = urllib2.urlopen(url).read()
			page = json.loads(page)
		 	texto = get_body_text(str(page['query']['pages']))
			texto = preprocesa_texto(texto)
			
			url_cat = "http://en.wikipedia.org/w/api.php?action=query&titles=" + titulo + "&prop=categories&format=json&clshow=!hidden"
			data =urllib2.urlopen(url_cat).read()
			data = json.loads(data)
			for key in page['query']['pages']:
				page_id = key
			try:
			  for key, value in data['query']['pages'][page_id].iteritems():
				 if key == 'categories':
					itemlist = value
			  
			  cat_example = []
			  categorias = '['
			  for c in itemlist:
				categ = str((c['title'].replace(" ", "_").replace("Category:", " ")))
				categoria =  '{"category":"'+categ.strip()+'"},'
				categorias += categoria
				cat_example.append(categ.strip())
			  categorias = categorias[:-1] +"]"
			  json_content =  {'title': titulo, 
					  'categories': categorias,
					  'text': texto}
			  if (numero_categorias(G, cat_example) >= min_car_example):
			    escribir_archivo (titulo, ruta, json_content)
			except:
			  print 'No existe el archivo'
			  
def numero_categorias(G, categories):
    number = 0
    for c in G.nodes():
      for cat in categories:
	if c == cat:
	  number += 1
    return number
			
def preprocesa_texto(texto):
    texto = re.sub("\[\[Category:.*?\]\]", "", texto)
    texto = ''.join([i for i in texto if not i.isdigit() and not i in set(string.punctuation)])
    return texto

    
def calcular_y(y, kf):
  y_c = []
  for element in kf:
    y_c.append(y[element])
  return y_c
 
def calcular_X(X, kf):
    X_c = []
    for element in kf:
      X_c.append(X[element])
    return X_c
    
    
def clasificar(G):
    target_names = G.nodes()
   
    wiki_raw = load_files(rutaBase, 'archivos', 'raw') 
    
    X = []

    for f in wiki_raw.filenames:
	    texto =json.loads((open(f).read()))['text']
	    X.append(str(texto))


    filas = len(wiki_raw.filenames) 
    columnas = len(target_names)
    
    y = []

    i = 0
    for f in wiki_raw.filenames:
	    data = (json.loads((open(f).read())))
	    data = json.loads(data['categories'])
	    aux = []
	    for j in range (0, len(target_names)):
		    encontrado = False
		    for item in data:
			if not(encontrado):
			    if (item['category'] == target_names[j]):
				    aux.append(j)
				    encontrado = True
	    y.append(aux)
	   
  
    classifier_nb = Pipeline([
	('tfidf', TfidfVectorizer(stop_words = "english")),
	('clf', OneVsRestClassifier(BernoulliNB()))])

    classifier_knn = Pipeline([
	('tfidf', TfidfVectorizer(stop_words = "english")),
	('clf', OneVsRestClassifier(KNeighborsClassifier()))])

    classifier_log = Pipeline([
	('tfidf', TfidfVectorizer(stop_words = "english")),
	('clf', OneVsRestClassifier(LogisticRegression()))])

    classifier_svc = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words = "english")),
        ('clf', OneVsRestClassifier(LinearSVC()))])

   
    kf = cross_validation.KFold(len(X), 10)
 
    hmnb = 0
    pnb = 0 
    rsnb = 0 
    dsnb = 0  
    f1micronb = 0
    f1macronb = 0
    acnb = 0
    

    hmknn = 0
    pknn = 0 
    rsknn = 0 
    dsknn = 0
    f1microknn = 0  
    f1macroknn = 0  
    acknn = 0

    hmlog = 0
    plog = 0
    rslog = 0
    dslog = 0
    f1microlog = 0
    f1macrolog = 0
    aclog = 0

    hmsvc = 0 
    psvc = 0 
    rssvc = 0 
    dssvc = 0
    f1microsvc = 0
    f1macrosvc = 0
    acsvc = 0

    hmdt = 0
    pdt = 0
    rsdt = 0
    dsdt = 0
    f1microdt = 0
    f1macrodt = 0
    acdt = 0

  
    for X_tra, X_te in kf:
      X_train = calcular_X(X, X_tra)
      X_test = calcular_X(X, X_te)
      y_train = calcular_y(y, X_tra)
      y_test = calcular_y(y, X_te)
      
      mlb = MultiLabelBinarizer().fit(y)
      yb_train = mlb.transform(y_train)
      yb_test = mlb.transform(y_test)

      classifier_nb.fit(X_train, yb_train)
      predicted_nb = classifier_nb.predict(X_test)
      hmnb += hamming_loss(yb_test, predicted_nb)
      pnb += precision_score(yb_test, predicted_nb)
      rsnb += recall_score(yb_test, predicted_nb)
      dsnb += distancia_simetrica(y_test, mlb.inverse_transform(predicted_nb), G)
      f1micronb += f1_score(yb_test, predicted_nb, average='micro')
      f1macronb += f1_score(yb_test, predicted_nb, average='macro')
      acnb += accuracy_score(yb_test, predicted_nb)

      classifier_knn.fit(X_train, yb_train)
      predicted_knn = classifier_knn.predict(X_test)
      hmknn += hamming_loss(yb_test, predicted_knn)
      pknn += precision_score(yb_test, predicted_knn)
      rsknn += recall_score(yb_test, predicted_knn)
      dsknn += distancia_simetrica(y_test, mlb.inverse_transform(predicted_knn), G)
      f1microknn += f1_score(yb_test, predicted_knn, average='micro')
      f1macroknn += f1_score(yb_test, predicted_knn, average='macro')
      acknn += accuracy_score(yb_test, predicted_knn)   

      classifier_log.fit(X_train, yb_train)
      predicted_log = classifier_log.predict(X_test)
      hmlog += hamming_loss(yb_test, predicted_log)
      plog += precision_score(yb_test, predicted_log)
      rslog += recall_score(yb_test, predicted_log)
      dslog += distancia_simetrica(y_test, mlb.inverse_transform(predicted_log), G)
      f1microlog += f1_score(yb_test, predicted_log, average='micro')
      f1macrolog += f1_score(yb_test, predicted_log, average='macro')
      aclog += accuracy_score(yb_test, predicted_log)      

      classifier_svc.fit(X_train, yb_train)
      predicted_svc = classifier_svc.predict(X_test)
      hmsvc += hamming_loss(yb_test, predicted_svc)
      psvc += precision_score(yb_test, predicted_svc)
      rssvc += recall_score(yb_test, predicted_svc)
      dssvc += distancia_simetrica(y_test, mlb.inverse_transform(predicted_svc), G)
      f1microsvc += f1_score(yb_test, predicted_svc, average='micro')
      f1macrosvc += f1_score(yb_test, predicted_svc, average='macro')
      acsvc += accuracy_score(yb_test, predicted_svc)
      
      tf = TfidfVectorizer(stop_words = "english").fit(X_train)
      tfXtrain =  TfidfVectorizer(stop_words = "english").fit_transform(X_train)

      classifier_dt = OneVsRestClassifier(DecisionTreeClassifier())
      classifier_dt.fit(tfXtrain.toarray(), yb_train)
      tfXtest = tf.transform(X_test)
      predicted_dt = classifier_dt.predict(tfXtest.toarray())
      hmdt += hamming_loss(yb_test, predicted_dt)
      pdt += precision_score(yb_test, predicted_dt)
      rsdt += recall_score(yb_test, predicted_dt)
      dsdt += distancia_simetrica(y_test, mlb.inverse_transform(predicted_dt), G)
      f1microdt += f1_score(yb_test, predicted_dt, average='micro')
      f1macrodt += f1_score(yb_test, predicted_dt, average='macro')
      acdt += accuracy_score(yb_test, predicted_dt)



    print  'hamming_loss_nb', hmnb/float(10)
    print 'precicion_nb', pnb/float(10)
    print 'recall_nb', rsnb/float(10)
    print 'ds_nb', dsnb/float(10)
    print 'f1_nb_micro', f1micronb/float(10)
    print 'f1_nb_macro', f1macronb/float(10)
    print 'acnb ', acnb/float(10) , '\n'

    print  'hamming_loss_knn', hmknn/float(10)
    print 'precision_knn', pknn/float(10)
    print 'recall_knn', rsknn/float(10)
    print 'ds_knn', dsknn/float(10)
    print 'f1_knn_micro', f1microknn/float(10)
    print 'f1_knn_macro', f1macroknn/float(10)
    print 'acknn', acknn/float(10), '\n'

    print  'hamming_loss_log', hmlog/float(10)
    print 'precision_log', plog/float(10)
    print 'recall_log', rslog/float(10)
    print 'ds_log', dslog/float(10)
    print 'f1_log_micro', f1microlog/float(10)
    print 'f1_log_macro', f1macrolog/float(10)
    print 'aclog ', aclog/float(10), '\n'

    print  'hamming_loss_svc', hmsvc/float(10)
    print 'precision_svc', psvc/float(10)
    print 'recall_svc', rssvc/float(10)
    print 'ds_svc', dssvc/float(10)
    print 'f1_micro_svc', f1microsvc/float(10)
    print 'f1_macro_svc', f1macrosvc/float(10)
    print 'acsvc' , acsvc/float(10), '\n'


    print  'hamming_loss_dt', hmdt/float(10)
    print 'precision_dt', pdt/float(10)
    print 'recall_dt', rsdt/float(10)
    print 'ds_dt', dsdt/float(10)
    print 'f1_micro_dt', f1microdt/float(10)
    print 'f1_macro_dt', f1macrodt/float(10)
    print 'acdt' , acdt/float(10), '\n'


    print 'cardinalidad etiquetas ', cardinalidad_etiquetas(y)
    print 'densidad etiquetas', densidad_etiquetas(y, G)
    print 'combinaciones etiquetas', combinaciones_etiquetas(y)
    print 'numero etiquetas', len(G)

    print y
    print predicted_knn

def cardinalidad_etiquetas(y):
  cardinality = 0
  for element in y:
     cardinality = cardinality + len(element)
  cardinality = float(cardinality) / float(len(y))
  return cardinality


def densidad_etiquetas(y, G):
  density = 0
  for element in y:
     density = density + len(element)
  density = (float(density) / float(len(y))) / len(G)
  return density

def combinaciones_etiquetas(y):
  distintos = []
  for i in range (0, len(y)):
    igual = False
    for j  in range (0, len(distintos)):
     if not igual:
       ejem1  = y[i]
       ejem2  = distintos[j]
       if(len(ejem1) == len(ejem2)):
         igual = True
	 for k in range (0, len(y[i])):
            if(ejem1[k] != ejem2[k]):
	       igual = False
    if not igual:
      distintos.append(y[i])
      igual = True
                 
  return len(distintos) 
 
def distancia_simetrica(y_test, predicted, G):
  target_names = G.nodes()
  md = []
  distancia_total = 0
  distancia_parcial = 0
  for i in range(0, len(predicted)):
          pred = predicted[i]
	  test = y_test[i]
	  encontrado = False
	  for p in pred:
	     for t in test: 
		if p == t:
		    encontrado = True
	     if not(encontrado):
		distancia_parcial = (float(1)/float((len(target_names)* len(pred)))) 
		for t in test:
		   dist, path = nx.bidirectional_dijkstra(G, target_names[t], target_names[p])
		   if dist < distancia_parcial:
			distancia_parcial = (float(dist)/float(len(target_names)))
                        distancia_parcial = (float(distancia_parcial) / float(len(test)))
		distancia_total +=  distancia_parcial
		#distancia_parcial = 1
		encontrado = False

  	  for t in test:
             for p in pred:
                if t == p:
                    encontrado = True
             if not(encontrado):
		distancia_parcial = (float(1)/float((len(target_names)*len(test))))
                for p in pred:
                   dist, path = nx.bidirectional_dijkstra(G, target_names[t], target_names[p])   
                   if dist < distancia_parcial:
                    distancia_parcial = float(dist)/float((len(target_names)))
                    distancia_parcial = (float(distancia_parcial)/float((len(pred))))	
		distancia_total +=  distancia_parcial
		#distancia_parcial = 1
		encontrado = False
  distancia_total = float(distancia_total) / float(len(y_test))
  return distancia_total

reload(sys)
sys.setdefaultencoding("utf-8")
sys.setrecursionlimit(1000000)
#os.mkdir(sys.argv[1])
rutaBase = sys.argv[1]    
ruta = rutaBase + '/raw/'
#os.mkdir(ruta) 


categEjem = [  'Individual_sports', 'Team_sports',  'Sports_by_type', 'Rackets_(sport)', 'Ball_games']

categSoftware = ['Mobile_software', 'Android_(operating_system)_software', 'BlackBerry_software', 'IOS_software', 'Windows_Phone_software',  'Android_games', 'Mobile_games', 'Android_games', 'Windows_Mobile_games', 'IOS_games', 'Software', 'Application_software',  'Free_application_software' ]


categDeportes = [ 'Ball_games', 'Individual_sports',  'Team_sports', 'Combat_sports',  'Water_sports',  'Water_polo', 'Bowling', 'Winter_sports', 'Ice_hockey', 'Gymnastics']

categBebidas = ['Beverages', 'Non-alcoholic_beverages', 'Alcoholic_beverages', 'Sports_drinks', 'Soft_drinks', 'Fermented_beverages', 'Mixed_drinks', 'Brand_name_beverage_products', 'Coca-Cola_brands', 'Cola_brands', 'Soft_drink_stubs'] 

categProfesiones = ['Occupations_by_type', 'Computer_occupations', 'Information_technology_qualifications', 'Cleaning_and_maintenance_occupations', 'Industrial_occupations', 'Metalworking_occupations', 'Machinists', 'Foundrymen', 'Consulting_occupations', 'Hospitality_occupations', 'Military_specialisms']

categComidas = ['Foods', 'Chocolate', 'Desserts', 'Meat', 'Sandwiches', 'Snack_foods', 'Soups', 'Salads', 'Vegetables', 'Condiments', 'Breads', 'Sweet_breads', 'Pancakes', 'Chocolate_desserts', 'Cookies', 'Biscuits_(British_style)', 'Ice_cream', 'Yogurts', 'Puddings', 'British_puddings', 'Rice_puddings', 'Doughnuts', 'Sweet_breads',  'Fast_food', 'Pizza', 'Brand_name_condiments', 'Poultry', 'Pork', 'Cold_soups']


G = construir_grafo(categEjem)


'''for category in G.nodes():
	descargar_paginas(category, ruta, 1)'''


#clasificar(G)

pos = nx.shell_layout(G)
nx.draw(G, pos, with_labels=True)
plt.show()

