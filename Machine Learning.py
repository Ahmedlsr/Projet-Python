# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:51:25 2022

@author: Ahmedlsr
"""
#Les librairies 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
#Stopwords
symbols = ['~', ':', "'", '+', '[', '\\', '@', '^', '{', '%', '(', '-', '"', '*', '|', ',', '&', '<', '`', '}', '.', '_', '=', ']', '!', '>', ';', '?', '#', '$', ')', '/']
inutile =['mr','mme','à','a,','quelqu','a','abord','absolument','afin','ah','ai','aie','aient','aies','ailleurs','ainsi','ait','allaient','allo','allons','allô','alors','anterieur','anterieure','anterieures','apres','après','as','assez','attendu','au','aucun','aucune','aucuns','aujourd','aujourd hui','aupres','auquel','aura','aurai','auraient','aurais','aurait','auras','aurez','auriez','aurions','aurons','auront','aussi','autant','autre','autrefois','autrement','autres','autrui','aux','auxquelles','auxquels','avaient','avais','avait','avant','avec','avez','aviez','avions','avoir','avons','ayant','ayez','ayons','b','bah','bas','basee','bat','beau','beaucoup','bien','bigre','bon','boum','bravo','brrr','c','car','ce','ceci','cela','celle','celle-ci','celle-là','celles','celles-ci','celles-là','celui','celui-ci','celui-là','celà','cent','cependant','certain','certaine','certaines','certains','certes','ces','cet','cette','ceux','ceux-ci','ceux-là','chacun','chacune','chaque','cher','chers','chez','chiche','chut','chère','chères','ci','cinq','cinquantaine','cinquante','cinquantième','cinquième','clac','clic','combien','comme','comment','comparable','comparables','compris','concernant','contre','couic','crac','d','da','dans','de','debout','dedans','dehors','deja','delà','depuis','dernier','derniere','derriere','derrière','des','desormais','desquelles','desquels','dessous','dessus','deux','deuxième','deuxièmement','devant','devers','devra','devrait','different','differentes','differents','différent','différente','différentes','différents','dire','directe','directement','dit','dite','dits','divers','diverse','diverses','dix','dix-huit','dix-neuf','dix-sept','dixième','doit','doivent','donc','dont','dos','douze','douzième','dring','droite','du','duquel','durant','dès','début','désormais','e','effet','egale','egalement','egales','eh','elle','elle-même','elles','elles-mêmes','en','encore','enfin','entre','envers','environ','es','essai','est','et','etant','etc','etre','eu','eue','eues','euh','eurent','eus','eusse','eussent','eusses','eussiez','eussions','eut','eux','eux-mêmes','exactement','excepté','extenso','exterieur','eûmes','eût','eûtes','f','fais','faisaient','faisant','fait','faites','façon','feront','fi','flac','floc','fois','font','force','furent','fus','fusse','fussent','fusses','fussiez','fussions','fut','fûmes','fût','fûtes','g','gens','h','ha','haut','hein','hem','hep','hi','ho','holà','hop','hormis','hors','hou','houp','hue','hui','huit','huitième','hum','hurrah','hé','hélas','i','ici','il','ils','importe','j','je','jusqu','jusque','juste','k','l','la','laisser','laquelle','las','le','lequel','les','lesquelles','lesquels','leur','leurs','longtemps','lors','lorsque','lui','lui-meme','lui-même','là','lès','m','ma','maint','maintenant','mais','malgre','malgré','maximale','me','meme','memes','merci','mes','mien','mienne','miennes','miens','mille','mince','mine','minimale','moi','moi-meme','moi-même','moindres','moins','mon','mot','moyennant','multiple','multiples','même','mêmes','n','na','naturel','naturelle','naturelles','ne','neanmoins','necessaire','necessairement','neuf','neuvième','ni','nombreuses','nombreux','nommés','non','nos','notamment','notre','nous','nous-mêmes','nouveau','nouveaux','nul','néanmoins','nôtre','nôtres','o','oh','ohé','ollé','olé','on','ont','onze','onzième','ore','ou','ouf','ouias','oust','ouste','outre','ouvert','ouverte','ouverts','o|','où','p','paf','pan','par','parce','parfois','parle','parlent','parler','parmi','parole','parseme','partant','particulier','particulière','particulièrement','pas','passé','pendant','pense','permet','personne','personnes','peu','peut','peuvent','peux','pff','pfft','pfut','pif','pire','pièce','plein','plouf','plupart','plus','plusieurs','plutôt','possessif','possessifs','possible','possibles','pouah','pour','pourquoi','pourrais','pourrait','pouvait','prealable','precisement','premier','première','premièrement','pres','probable','probante','procedant','proche','près','psitt','pu','puis','puisque','pur','pure','q','qu','quand','quant','quant-à-soi','quanta','quarante','quatorze','quatre','quatre-vingt','quatrième','quatrièmement','que','quel','quelconque','quelle','quelles',"quelqu un",'quelque','quelques','quels','qui','quiconque','quinze','quoi','quoique','r','rare','rarement','rares','relative','relativement','remarquable','rend','rendre','restant','reste','restent','restrictif','retour','revoici','revoilà','rien','s','sa','sacrebleu','sait','sans','sapristi','sauf','se','sein','seize','selon','semblable','semblaient','semble','semblent','sent','sept','septième','sera','serai','seraient','serais','serait','seras','serez','seriez','serions','serons','seront','ses','seul','seule','seulement','si','sien','sienne','siennes','siens','sinon','six','sixième','soi','soi-même','soient','sois','soit','soixante','sommes','son','sont','sous','souvent','soyez','soyons','specifique','specifiques','speculatif','stop','strictement','subtiles','suffisant','suffisante','suffit','suis','suit','suivant','suivante','suivantes','suivants','suivre','sujet','superpose','sur','surtout','t','ta','tac','tandis','tant','tardive','te','tel','telle','tellement','telles','tels','tenant','tend','tenir','tente','tes','tic','tien','tienne','tiennes','tiens','toc','toi','toi-même','ton','touchant','toujours','tous','tout','toute','toutefois','toutes','treize','trente','tres','trois','troisième','troisièmement','trop','très','tsoin','tsouin','tu','té','u','un','une','unes','uniformement','unique','uniques','uns','v','va','vais','valeur','vas','vers','via','vif','vifs','vingt','vivat','vive','vives','vlan','voici','voie','voient','voilà','voire','vont','vos','votre','vous','vous-mêmes','vu','vé','vôtre','vôtres','w','x','y','z','zut','à','â','ça','ès','étaient','étais','était','étant','état','étiez','étions','été','étée','étées','étés','êtes','être','ô']
inutile.append(symbols)
import re
def tokenize(txt):
    tokens = re.split('\W+', txt)
    return tokens
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
import time 
 
#Import fichier, lettre uniquement et en miniscule, stopwords, stemming
df=pd.read_excel('projet.xlsx')
df.loc[df["solution"]=='cassation','solution']=1
df.loc[df["solution"]=='rejet','solution']=0
df['moyens'] = df['moyens'].apply(lambda x: ' '.join([word for word in x.split() if word.isalpha()]))
df["moyens"] = df["moyens"].apply(lambda x: x.lower())
df['moyens'] = df['moyens'].apply(lambda x: ' '.join([word for word in x.split() if word not in inutile]))
df['moyens'] = df['moyens'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
df["moyens"]=df["moyens"].apply(lambda x: tokenize(x))
X = df['moyens']
y = df['solution']
X=[" ".join(t) for t in X]

#Vectorisation et array
vec = TfidfVectorizer()
X_vec = vec.fit_transform(X)
vec = TfidfVectorizer(max_features = 4000, ngram_range=(1,3))
X_vec2 = vec.fit_transform(X).toarray()
y = y.astype(np.uint8).values.ravel()
#Separation base test et train + True pour cassation dans le tableau 
X_train,X_test, y_train, y_test = train_test_split(X_vec2, y, test_size=0.10, random_state=54)
y_train_0 = (y_train == 1)
y_test_0 = (y_test == 1)

####################################################
#           IMPLEMENTER DES ALGORITHMES            #
####################################################

###########  Regression logistique #################
start = time.time()
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train_0)
print("Score regression logistique : ")
print(f'% Classification Correct - (Base Test) : {LogReg.score(X_test, y_test_0):.3f}')
print(f'% Classification Correct - (Base Train) : {LogReg.score(X_train, y_train_0):.3f}')
#Score de la regression 
y_pred_reg = LogReg.predict(X_test)
print(classification_report(y_test_0,y_pred_reg))
end = time.time()
print("Temps :",(end-start),'s')

################# K plus proche voisin #############
start = time.time()
neigh = KNeighborsClassifier()
neigh.fit(X_train, y_train_0)
# Score KNN
print("Score KNN")
print(f'% Classification Correct - (Base Test) : {neigh.score(X_test, y_test_0):.3f}')
print(f'% Classification Correct - (Base Train) : {neigh.score(X_train, y_train_0):.3f}')
y_pred_neigh = neigh.predict(X_test)
print(classification_report(y_test_0,y_pred_neigh))
end = time.time()
print("Temps :",(end-start),'s')

################# Random Forest #################### 
start = time.time()
RF_Model = RandomForestClassifier()#random_state=8,n_estimators = 150, max_depth=(23))
RF_Model.fit(X_train, y_train_0)
print("Score Random Forest")
print(f'% Classification Correct - (Base Test): {RF_Model.score(X_test, y_test_0):.3f}')
print(f'% Classification Correct - (Base Train) : {RF_Model.score(X_train, y_train_0):.3f}')
#Score RF
y_pred_forest = RF_Model.predict(X_test)
print(classification_report(y_test_0,y_pred_forest))
end = time.time()
print("Temps :",(end-start),'s')

########### Machine a vecteur de support  ##############
start = time.time()
svc = svm.SVC().fit(X_train, y_train_0)
print(f'% Classification Correct - Linéaire (Base Apprentissage) : {svc.score(X_train, y_train_0):.3f}')
print(f'% Classification Correct - Linéaire (Base Test) : {svc.score(X_test, y_test_0):.3f}')
print('Score vecteur de support')
y_pred_vec1 = svc.predict(X_test)
print(classification_report(y_test_0,y_pred_vec1)) 
end = time.time()
print("Temps :",(end-start),'s')

############## Reseau de neurones ##################
start = time.time()
rf = MLPClassifier(solver = 'lbfgs',activation = 'tanh', random_state=21,max_iter=500)
rf.fit(X_train, y_train_0)
print("Score reseaux de neurones ")
print(f'% Classification Correct - (Base Test) : {rf.score(X_test, y_test_0):.3f}')
print(f'% Classification Correct - (Base Train) : {rf.score(X_train, y_train_0):.3f}')
y_pred_neur = rf.predict(X_test)
print(classification_report(y_test_0,y_pred_neur))
end = time.time()
print("Temps :",(end-start),'s')

############### Ridge Classsifier ##################
start = time.time()
print("Score Ridge classifier")
ridg = RidgeClassifier().fit(X_train, y_train_0)
print(f'% Classification Correct - (Base Test) : {ridg.score(X_test, y_test):.3f}')
print(f'% Classification Correct - (Base Train) : {ridg.score(X_train, y_train_0):.3f}')
y_pred_ridg = ridg.predict(X_test)
print(classification_report(y_test_0, y_pred_ridg))
end = time.time()
print("Temps :",(end-start),'s')

####################################################
#      Choix des variables explicatives            #
####################################################
# On chosi dès maintenant le modèle le plus perf
# Random Forest
maxf = [4000, 40000]
ngram = [(1,1),(1,2), (1,3)]
m=0
n=()
pred=0
for a in maxf:
    for b in ngram:
        #Ensemble de variables explicatives
        vec = TfidfVectorizer()
        X_vec = vec.fit_transform(X)
        vec = TfidfVectorizer(max_features = a, ngram_range=b)
        X_vec2 = vec.fit_transform(X).toarray()
        #Separation base test et train + True pour cassation dans le tableau 
        X_train,X_test, y_train, y_test = train_test_split(X_vec2, y, test_size=0.10, random_state=54)
        y_train_0 = (y_train == 1)
        y_test_0 = (y_test == 1)
        # RF
        RF_Model = RandomForestClassifier(random_state= 13)
        RF_Model.fit(X_train, y_train)
        print("Max features",a,"Ngram",b)
        print(f'% Classification Correct - (Base Test) : {RF_Model.score(X_test, y_test_0):.3f}')
        print(f'% Classification Correct - (Base Train) : {RF_Model.score(X_train, y_train_0):.3f}')
        print()
        #Score RF
        if RF_Model.score(X_test, y_test_0)>pred:
            pred=RF_Model.score(X_test, y_test_0)
            m=a
            n=b
print(m,n)
####################################################
#            Choix des Hyperparametre              #
####################################################
vec = TfidfVectorizer()
X_vec = vec.fit_transform(X)
vec = TfidfVectorizer(max_features = m, ngram_range=n)
X_vec2 = vec.fit_transform(X).toarray()
#Separation base test et train + True pour cassation dans le tableau 
X_train,X_test, y_train, y_test = train_test_split(X_vec2, y, test_size=0.10, random_state=54)
y_train_0 = (y_train == 1)
y_test_0 = (y_test == 1)

# Random Forest
max_depths = [1,10,20,23,25,50]
n_arbre = [1,50,100,150,200]
val=0
for dep in max_depths:
    for arb in n_arbre: 
        rf =   RF_Model = RandomForestClassifier(n_estimators = arb, max_depth=dep, random_state=13)
        rf.fit(X_train, y_train)
        if rf.score(X_test, y_test_0)>val:
            val = rf.score(X_test, y_test_0)
            arb1=arb
            dep1=dep
print("Taux de classification",val,"Nombre d'arbres",arb1,"Profondeur",dep1)

# KNN
print()
error_rate = []
for i in range(1,40):
     neigh = KNeighborsClassifier(n_neighbors=i)
     neigh.fit(X_train,y_train_0)
     pred_i = neigh.predict(X_test)
     error_rate.append(np.mean(pred_i != y_test_0))
req_k_value = error_rate.index(min(error_rate))+1
print("Meilleur K voisin")
#graph 
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed',marker='o',markerfacecolor='red', markersize=10)
plt.title('Taux d erreur de classification vs. Nombre de voisins K')
plt.xlabel('K')
plt.ylabel('Taux d erreur de classification')
req_k_value = error_rate.index(min(error_rate))+1
fig1 = plt.gcf()
plt.show()
print("Minimum error:-",min(error_rate),"at K =",req_k_value)

# Score avec le meilleur hyperparametre
start = time.time()
neigh = KNeighborsClassifier(n_neighbors=req_k_value)
neigh.fit(X_train,y_train_0)
pred_i = neigh.predict(X_test)
print(f'% Classification Correct - (Base Test) : {neigh.score(X_test, y_test_0):.3f}')
print(f'% Classification Correct - (Base Train) : {neigh.score(X_train, y_train_0):.3f}')
y_pred_neigh = neigh.predict(X_test)
print(classification_report(y_test_0,y_pred_neigh))
end = time.time()
print("Temps :",(end-start),'s')

# Reseau de neurones :
n_couche = [1,10,20,50]
train_results = []
test_results = []
maxMLP = 0
n_couch = 0
for i in n_couche:
   rf = MLPClassifier(hidden_layer_sizes=(i),solver = 'lbfgs',activation = 'tanh', random_state=21,max_iter=500)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   test_r = rf.score(X_test,y_test)
   test_results.append(test_r)
   train_r = rf.score(X_train,y_train)
   train_results.append(train_r)
   if rf.score(X_test, y_test_0) >maxMLP:
       maxMLP=rf.score(X_test, y_test_0)
       n_couch = i
print("Taux de classification :",maxMLP,'Nombre de couche :',n_couch)       
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_couche, train_results, label='Erreur d apprentissage')
line2, = plt.plot(n_couche, test_results, label='Erreur de test')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Taux de Classification Correct')
plt.xlabel('Nombre de couches cachées')
fig1 = plt.gcf()
plt.show()

#SVM
list_C = [1,10,100]
for i in list_C:
    svc = svm.SVC(kernel='linear', C=i).fit(X_train, y_train)
    print(f'% Classification Correct - Linéaire (Base Apprentissage) : {svc.score(X_train, y_train_0):.3f}')
    print(f'% Classification Correct - Linéaire (Base Test) : {svc.score(X_test, y_test_0):.3f}')
#On prend c=10
svc = svm.SVC(kernel='linear', C=10).fit(X_train, y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=10).fit(X_train, y_train)
poly_svc = svm.SVC(kernel='poly', degree=3, C=10).fit(X_train, y_train)
print(f'% Classification Correct - Linéaire (Base Test) : {svc.score(X_test, y_test_0):.3f}')
print(f'% Classification Correct - RBF (Base Test) : {rbf_svc.score(X_test, y_test_0):.3f}')
print(f'% Classification Correct - Poly (Base Test) : {poly_svc.score(X_test, y_test_0):.3f}')

#fit de nouveau juste pour avoir le vrai temps 
start = time.time()
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=10).fit(X_train, y_train)
y_pred_vec1 = rbf_svc.predict(X_test)
print('Score svm')
print(classification_report(y_test_0,y_pred_vec1))
end= time.time()
print("Temps :",(end-start),'s') 

####################################################
#            Choix du meilleur modèle              #
####################################################
#Randome Forest
start = time.time()
RF_Model = RandomForestClassifier(random_state= 13 ,n_estimators = 150, max_depth=(23))
RF_Model.fit(X_train, y_train)
print("Score Random Forest")
print(f'% Classification Correct - (Base Test) : {RF_Model.score(X_test, y_test_0):.3f}')
print(f'% Classification Correct - (Base Train) : {RF_Model.score(X_train, y_train_0):.3f}')
#Score RF
y_pred_forest = RF_Model.predict(X_test)
print(classification_report(y_test_0,y_pred_forest))
end = time.time()
print("Temps :",(end-start),'s')

########### Dummy Classifier ##############
start = time.time()
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train_0)
print('Score dummy')
print(f'% Classification Correct - (Base test) : {dummy_clf.score(X_test, y_test_0):.3f}')
print(f'% Classification Correct - (Base Train) : {dummy_clf.score(X_train, y_train_0):.3f}')
y_pred_dummy = dummy_clf.predict(X_test)
print(classification_report(y_test_0,y_pred_dummy))
end = time.time()
print("Temps :",(end-start),'s')

######## Graph pour Random Forest #########
# EN FONCTION DE LA PROFONDEUR

max_depths = [1,10,20,23,25,50]
train_results = []
test_results = []
for i in max_depths:
   rf =   RF_Model = RandomForestClassifier(bootstrap=True, max_depth=i, random_state=13)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   test_r = rf.score(X_test,y_test)
   test_results.append(test_r)
   train_r = rf.score(X_train,y_train)
   train_results.append(train_r)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, label='Erreur d apprentissage')
line2, = plt.plot(max_depths, test_results, label='Erreur de test')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Taux de Classification Correct')
plt.xlabel('Profondeur d un arbre')
fig1 = plt.gcf()
#fig1.savefig('/Users/erreur_classification_profondeur_arbre_rf.png')
plt.show()

# EN FONCTION DU NOMBRE D ARBRES
n_arbre = [1,50,100,150,200,300]
train_results = []
test_results = []
for i in n_arbre:
   rf =   RF_Model = RandomForestClassifier(bootstrap=True,  n_estimators=i, random_state=13)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   test_r = rf.score(X_test,y_test)
   test_results.append(test_r)
   train_r = rf.score(X_train,y_train)
   train_results.append(train_r)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_arbre, train_results, label='Erreur d apprentissage')
line2, = plt.plot(n_arbre, test_results, label='Erreur de test')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Taux de Classification Correct')
plt.xlabel('Nombre d arbres')
fig1 = plt.gcf()
#fig1.savefig('/Users/erreur_classification_n_arbre_rf.png')
plt.show()