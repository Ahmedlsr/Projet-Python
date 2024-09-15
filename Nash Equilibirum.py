print("Ce projet à pour but de trouver l'equilibre de nash de la bimatrice des gains de deux joueurs : A et B pour un nombre illimité de stratégie")
A = int(input("Nombre de stratégies pour A :"))
B = int(input("Nombre de stratégies pour B :"))
Nom_StrategiesA = []
Nom_StrategiesB = []
print()
print("Pour A entrez une par une le nom de ses", A, "strategies")
for a in range(0, A): #On demande à l'utilisateur d'enter le nom de ses strat une par une
     i = str(input("Nom de la stratégie de A :"))
     Nom_StrategiesA.append(i)
print()
print("Pour B entrez une par une le nom de ses", B, "strategies") #idem
for a in range(0, B):
     i = str(input("Nom de la stratégie de B :"))
     Nom_StrategiesB.append(i)
StrategiesA = [0]*A #On a n strat donc n matrices
StrategiesB = [0]*B
for a in range(0, len(Nom_StrategiesA)): #On met en place la matrice A en la remplissant de 0 pour ensuite demander à l'utilisateur d'entrer ses valeurs
     StrategiesA[a] = [0] * B #Selon les valeurs de B pour remplir toutes les cases
for a in range(0, len(Nom_StrategiesB)): #idem
     StrategiesB[a] = [0] * A
print("Attention pour les nombres decimaux entrez des points et non pas des virgules ! ")
print()
for i in range(0, A): #ici l'utilisateur entre les valeurs des paiements pour une combinaison de stratégies
     for y in range(0, B):
         print("Si A joue sa stratégie", Nom_StrategiesA[i], "et B joue sa stratégie", Nom_StrategiesB[y])
         StrategiesA[i][y] = float(input("Entrez le paiement pour A :"))
         StrategiesB[y][i] = float(input("Entrez le paiement pour B :"))
         print()
print("En résumé")
print()
print("Stratégies de A avec ses paiements :")
print(Nom_StrategiesA)  # ligne
print(StrategiesA)  # ligne
print()
print("Stratégies de B avec ses paiements :")
print(Nom_StrategiesB)  # colonne
print(StrategiesB)  # colonne
print()
indiceA = -1 #valeur =-1 pour voir si ca prend une valeur de la matrice allant de 0 à A/B
indiceB = -1
compteA = 0
MR = 1
if A == 1:
    dominA = StrategiesA[0]
    indiceA = 0
else:
    for i in range(0, A):
        compteA = 0
        for t in range(0, A):
            for y in range(0, B):
                if StrategiesA[i][y] > StrategiesA[t][y]:
                    compteA = compteA + 1
                    if compteA == B*(A-1):  # Ex 2 strat de B il faut verifier que c'est vrai les 2 fois
                        dominA = StrategiesA[i]
                        indiceA = i
                        print("La stratégie dominante de A est :", Nom_StrategiesA[indiceA])
                        print("Les paiements de la stratégie strictement dominante de A sont", dominA)
if B == 1:
    dominB = StrategiesB[0]
    indiceB = 0
else:
    compteB = 0
    for r in range(0, B):
        compteB = 0
        for q in range(0, B):
            for s in range(0, A):
                if StrategiesB[r][s] > StrategiesB[q][s]:
                    compteB = compteB + 1
                    if compteB == (B-1)*A:  
                        dominB = StrategiesB[r]
                        indiceB = r
                        print("La stratégie dominante de B est :", Nom_StrategiesB[indiceB])
                        print("Les paiements de la stratégie strictement dominante de B sont", dominB)
                        MR=0
if indiceA != -1 and indiceB != -1:
    print()
    print("L'équilibre de nash est donc :", "(", Nom_StrategiesA[indiceA], ",", Nom_StrategiesB[indiceB], ")")
    print("Avec comme paiement", StrategiesA[indiceA][indiceB], "pour A et", StrategiesB[indiceB][indiceA],"Pour B")
    MR=0
elif indiceA != -1: # Indice A diff de 0 donc une strat dominante
    print("Pas de stratrégie dominante pour B")
    for r in range(0, B):
        compteDomiB = 0
        for q in range(0, B):
            if StrategiesB[r][indiceA] > StrategiesB[q][indiceA]:
                compteDomiB = compteDomiB + 1
                if compteDomiB == B - 1:  # Pour n strategie on a une strategie qui va dominer les n-1 autres
                    print("B sait que A va toujours jouer sa stratégie dominante, il va donc choisir le paiement le élévé sachant cela")
                    indiceB = r
                    print("La combinaison de stratégies", Nom_StrategiesA[indiceA], ",", Nom_StrategiesB[indiceB],"est donc un equilibre de Nash")
                    print("Avec comme paiement", StrategiesA[indiceA][indiceB], "pour A et", StrategiesB[indiceB][indiceA], "pour B:")
                    MR=0
elif indiceB != -1: # Indice B diff de 0 donc une strat dominante
    print("Pas de stratrégie dominante pour A") #On fixe l'indice de B et on va chercher le paiement le plus haut pour B
    for r in range(0, A):
        compteDomiA = 0
        for q in range(0, A):
            if StrategiesA[r][indiceB] > StrategiesA[q][indiceB]:
                compteDomiA = compteDomiA + 1
                if compteDomiA == A - 1:  # Pour n strategie on a une strategie qui va dominer les n-1 autres
                    print("Mais A sait que B va toujours jouer sa stratégie dominante, il va donc choisir le paiement le élévé sachant cela")
                    indiceA = r
                    print("La combinaison de stratégies", Nom_StrategiesA[indiceA], ",", Nom_StrategiesB[indiceB],"est donc un equilibre de Nash")
                    print("Avec comme paiement", StrategiesA[indiceA][indiceB], "pour A et", StrategiesB[indiceB][indiceA], "pour B:")
                    MR=0
if MR == 1:
  print("Pas d'equilibre avec les stratégies strictements dominantes, on passe a l'EN avec les meilleures réponses")
  MRGA = [0] * B
  for j in range(0,B):
    MRGA[j] = StrategiesA[0][j]
    for i in range(0, A):
      if StrategiesA[i][j]>MRGA[j] :
        MRGA[j] =  StrategiesA[i][j]
  MRGB = [0] * A
  for i in range(0,A):
    MRGB[i] = StrategiesB[0][i]
    for j in range(0, B):
      if StrategiesB[j][i]>MRGB[i] :
        MRGB[i] =  StrategiesB[j][i]
  EN=[]
  for i in range(0,A):
    for j in range(0,B):
      if StrategiesA[i][j] == MRGA[j] and StrategiesB[j][i] == MRGB[i]:
        EN.append([StrategiesA[i][j],StrategiesB[j][i]])
        print("La combinaison de la stratégies", Nom_StrategiesA[i], "de A et ", Nom_StrategiesB[j],"de B est un equilibre de Nash")
        print("Avec comme paiement", StrategiesA[i][j], "pour A et", StrategiesB[j][i], "pour B:")
        print()
  print("Il y'a donc",len(EN),"equilibres de nash")
  print(EN)