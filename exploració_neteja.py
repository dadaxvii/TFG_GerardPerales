import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('C:/Users/dadap/Desktop/TFG/df_original.csv')
print(df.shape) #mesures del Dataframe
#df.info() #Nom columnes i tipus de variables
print(df.isnull().sum())

#Amb el df.info() i el df.isnull().sum podem veure quines columnes tenen valors Nans, i en funció del tipus de variable,
#ens desfarem d'elles d'una forma o un altre.

#Gestio dels valors Nans
#Education
df['education'] = df['education'].fillna(df['education'].mode()[0]) #df.mode() retorna un dataframe amb
#els valors més comuns d'un atribut concret o d'un dataframe. En aquest cas li hem dit que agafi el valor més
#comú de la columna education i el posi als Nans.


#Glucose
print(df['glucose'].describe())
sns.boxplot(df['glucose'])
plt.show()
df['glucose']=df['glucose'].fillna(df['glucose'].median())
#Al haver-hi outliers, he decidit omplir els Nans amb la mediana ja que no és tan sensible com la mitjana, que es pot
#veure més afectada per valors extrems distants, ja que té en compte tots els valors.

#Sex
df['sex'] = df['sex'].replace({'M': 1, 'F': 0})
#CigsPerDay i is_smoking
'''
df['is_smoking'].unique() Amb aquesta línia veiem que els valora que hi ha a l'atribut
is_smoking són YES i NO. Aquesta columna no conté Nans, però sí que ens interessa passar
els strings a ints'''
df['is_smoking'] = df['is_smoking'].replace({'YES': 1, 'NO': 0})


#n = ((df['cigsPerDay'].isnull()) & (df['is_smoking'] == 1)).sum()
#print(n)
'''
Amb les dues línies superiors mirem dels valors que són nuls de cigsPerDay, quants són
fumadors i quants no, per omplir els Nans amb un valor o un altre. Veiem com tots els
Nans són de pacients fumadors.'''

df_fumadors = df.loc[df['is_smoking'] == 1]
print(df_fumadors['cigsPerDay'].describe())
'''Amb això agafem només els cigsPerDay dels fumadors, ja que al haver-hi gent que no fuma, la mitjana i la mediana es
 afectada pels zeros. Utilitzarem la mediana per omplir els nans'''
df['cigsPerDay'] = df['cigsPerDay'].fillna(df_fumadors['cigsPerDay'].median())



#heartRate
df['heartRate']=df['heartRate'].fillna(df['heartRate'].median())

#BMI
df['BMI']=df['BMI'].fillna(df['BMI'].mean())

#BPMeds
'''Al ser 0 o 1, omplirem els Nans amb la moda de l'atribut.'''
df['BPMeds']=df['BPMeds'].fillna(df['BPMeds'].mode()[0])

#totChol
df['totChol']=df['totChol'].fillna(df['totChol'].median())


'''El que hem anat fent és mirar per cada variable quins tipus de dades té. Si per exemple era categòrcia binària,
ompliem els Nans amb la moda. Si els valors eren enters, com la mitjana retornava un valor amb decimals, utilitzàvem la
mediana. En el cas que tots els valors fóssin decimals, utilitzàvem la mitjana.'''
df.to_csv('C:/Users/dadap/Desktop/TFG/df_net.csv', index=False)
