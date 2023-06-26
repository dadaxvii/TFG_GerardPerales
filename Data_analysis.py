import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from funcio import taula_TG
plt.style.use('ggplot')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm ## Permet ajustar models estadístics utilitzant fòrmules d'estil R
from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

grafics = False
table1 = False
table2 = False
table3 = False
table4 = False
table5 = False

df = pd.read_csv('C:/Users/dadap/Desktop/TFG/df_net.csv')
x = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

corr = df.corr()
print(corr)
correlacions = corr.loc['TenYearCHD']
corr_ordenades = correlacions.sort_values(ascending=False)

'''MATRIU DE CORRELACIÓ:
'''
if grafics:
    sns.set()
    sns.heatmap(corr, annot=True, cmap="YlGnBu")
    plt.show()


#print(corr_ordenades)
'''TenYearCHD         1.000000
age                0.224927
sysBP              0.212703
prevalentHyp       0.166544
diaBP              0.135979
glucose            0.133472
diabetes           0.103681
totChol            0.093605
BPMeds             0.087349
sex                0.084647
prevalentStroke    0.068627
BMI                0.066543
cigsPerDay         0.064745
is_smoking         0.034143
heartRate          0.020167
id                 0.009866
education         -0.051388
Name: TenYearCHD, dtype: float64
'''

'''BMI i CigsPerDay correlació baixa amb la variable dependent.'''
if grafics:
    sns.boxplot(x='TenYearCHD', y='BMI', data=df)
    plt.show()
    sns.boxplot(x='TenYearCHD', y='cigsPerDay', data=df)
    plt.show()

    '''L'edat és la variable amb més correlació.'''
    sns.violinplot(x='TenYearCHD', y='age', data=df)
    plt.show()








y = df['TenYearCHD']
# REGRESSIÓ MÚLTIPLE PER VEURE QUINES SÓN LES VARIABLES ESTADÍSTICAMENT SIGNIFICATIVES
x = df.drop('TenYearCHD', axis=1).copy()
x = sm.add_constant(x)
# Model de regressió múltiple
modelo = sm.Logit(y, x).fit()
pvalues = modelo.pvalues[1:]
pvalues_formatted = [format(p, ".4f") for p in pvalues]
print(pvalues_formatted)
'''
VARAIBLES ESTADÍSTICAMENT SIGNIFICATIVES:
        - age
        - sysBP
        - glucose
        - sex
        - prevalentStroke
        - cigsPerDay
'''
x1 = ['age', 'sysBP', 'glucose', 'sex', 'prevalentStroke', 'cigsPerDay']
seleccio_1 = ['age', 'sysBP', 'glucose', 'sex', 'prevalentStroke', 'cigsPerDay', 'TenYearCHD']
corr_1_1 = df[seleccio_1].corr()
if grafics:
    sns.set()
    sns.heatmap(abs(corr_1_1), annot=True, cmap="YlGnBu")
    plt.show()
#La correlació de prevalentStroke és molt baixa, no només amb la variable dependent, sinó amb totes les variables, de manera
#que he decidit descartar-la.
x2 = ['age', 'sysBP', 'glucose', 'sex', 'cigsPerDay']

# PCA
pca = PCA()
pca.fit(df[x2])

#Variança explicada acumulada
if grafics:
    explained_variance_ratio_cumulative = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(explained_variance_ratio_cumulative) + 1), explained_variance_ratio_cumulative, marker='o')
    plt.xlabel('Nombre de components principals')
    plt.ylabel('Variança explicada acumulada')
    plt.title('Gráfico de Scree')
    plt.show()

varianzas_explicadas = pca.explained_variance_ratio_
nombres_variables = df[x2].columns

importancia_variables = sorted(zip(nombres_variables, varianzas_explicadas), key=lambda x: x[1], reverse=True)
for variable, importancia in importancia_variables:
    print(f'{variable}: {importancia:.4f}')
'''
VARAIBLES AGAFADES PEL PCA:
- age: 0.4815
- sysBP: 0.3596
- glucose: 0.1128

sex: 0.0459
cigsPerDay: 0.0002
'''



### SELECCIÓ 1: variables estadísticament significatives

'''REGRESIONS MÚLTIPLES AMB LES 3 SELECCIONS:'''
modelo1 = LinearRegression()
# 1-
x_train1, x_test1, y_train1, y_test1 = train_test_split(df[x1], y, test_size=0.2, random_state=42)
modelo1.fit(x_train1, y_train1)
y_pred1 = modelo1.predict(x_test1)

mse_1 = mean_squared_error(y_test1, y_pred1)
r2_1 = r2_score(y_test1, y_pred1)

# 2-
x_train2, x_test2, y_train2, y_test2 = train_test_split(df[x2], y, test_size=0.2, random_state=42)
modelo1.fit(x_train2, y_train2)
y_pred2 = modelo1.predict(x_test2)

mse_2 = mean_squared_error(y_test2, y_pred2)
r2_2 = r2_score(y_test2, y_pred2)

# 3-
x_train2, x_test2, y_train2, y_test2 = train_test_split(df[x2], y, test_size=0.2, random_state=42)
components = 3
pca_final = PCA(n_components=components)
x_train_pca = pca_final.fit_transform(x_train2)
x_test_pca = pca_final.transform(x_test2)

modelo1.fit(x_train_pca, y_train2)
y_pred3 = modelo1.predict(x_test_pca)

mse_3 = mean_squared_error(y_test2, y_pred3)
r2_3 = r2_score(y_test2, y_pred3)


'''REGRESIONS LOGÍSTIQUES AMB LES 3 SELECCIONS:'''
modelo2 = LogisticRegression()
# 1-
x_train1, x_test1, y_train1, y_test1 = train_test_split(df[x1], y, test_size=0.2, random_state=42)
modelo2.fit(x_train1, y_train1)
y_pred4 = modelo2.predict(x_test1)

exact1 = accuracy_score(y_test1, y_pred4)
precision1 = precision_score(y_test1, y_pred4, average='weighted')
recall1 = recall_score(y_test1, y_pred4, average='weighted')
f11 = f1_score(y_test1, y_pred4, average='weighted')

# 2-
x_train2, x_test2, y_train2, y_test2 = train_test_split(df[x2], y, test_size=0.2, random_state=42)
modelo2.fit(x_train2, y_train2)
y_pred5 = modelo2.predict(x_test2)



exact2 = accuracy_score(y_test1, y_pred5)
precision2 = precision_score(y_test1, y_pred5, average='weighted')
recall2 = recall_score(y_test1, y_pred5, average='weighted')
f12 = f1_score(y_test1, y_pred5, average='weighted')

# 3-
x_train2, x_test2, y_train2, y_test2 = train_test_split(df[x2], y, test_size=0.2, random_state=42)
modelo2.fit(x_train2, y_train2)
pca_final = PCA(n_components=3)
x_train_pca = pca_final.fit_transform(x_train2)
x_test_pca = pca_final.transform(x_test2)

modelo2.fit(x_train_pca, y_train2)
y_pred6 = modelo2.predict(x_test_pca)

exact3 = accuracy_score(y_test2, y_pred6)
precision3 = precision_score(y_test2, y_pred6, average='weighted')
recall3 = recall_score(y_test2, y_pred6, average='weighted')
f13 = f1_score(y_test2, y_pred6, average='weighted')

if grafics:
    print(y_test2.value_counts())
    cm1 = (confusion_matrix(y_test2, y_pred6))
    labels = ['Sense risc', 'Amb risc']


    sns.heatmap(cm1, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Prediccions')
    plt.ylabel('Valors Reals')
    plt.title('Matriu de Confusió model logístic(2)')
    plt.show()






### SELECCIÓ 2: variables continues amb un mínim de correlació i variables no contínues que tenen correlació amb alguna de les contínues.
x7 = df.loc[:,['age', 'is_smoking', 'cigsPerDay', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'glucose']]
seleccio_2 = ['age', 'is_smoking', 'cigsPerDay', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'glucose', 'TenYearCHD']
corr2 = df[seleccio_2].corr().abs()
'''
sns.set()
sns.heatmap(corr2, annot=True, cmap="YlGnBu")
plt.show()
'''
#La variable cigsPerDay només té correlació amb la variable de 'is_smoking', per tant he decidit esborrar-la
x8 =['age', 'is_smoking', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'glucose']

# PCA
pca = PCA()
pca.fit(df[x8])
'''
explained_variance_ratio_cumulative = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(explained_variance_ratio_cumulative) + 1), explained_variance_ratio_cumulative, marker='o')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Explicada Acumulada')
plt.title('Gráfico de Scree')
plt.show()
'''

varianzas_explicadas = pca.explained_variance_ratio_
nombres_variables = df[x8].columns

importancia_variables = sorted(zip(nombres_variables, varianzas_explicadas), key=lambda x: x[1], reverse=True)
for variable, importancia in importancia_variables:
    print(f'{variable}: {importancia:.4f}')

'''
VARIABLES AGAFADES AL PCA:
- age: 0.6307
- is_smoking: 0.1889
- prevalentHyp: 0.1481

diabetes: 0.0200
totChol: 0.0122
sysBP: 0.0001
diaBP: 0.0000
glucose: 0.0000'''


modelo1 = LinearRegression()
'''REGRESIONS MÚLTIPLES AMB LES 3 SELECCIONS:'''
# 1-
x_train6, x_test6, y_train6, y_test6 = train_test_split(x7, y, test_size=0.2, random_state=42)
modelo1.fit(x_train6, y_train6)
y_pred6 = modelo1.predict(x_test6)

mse_6 = mean_squared_error(y_test6, y_pred6)
r2_6 = r2_score(y_test6, y_pred6)

# 2-
x_train7, x_test7, y_train7, y_test7 = train_test_split(df[x8], y, test_size=0.2, random_state=42)
modelo1.fit(x_train7, y_train7)
y_pred7 = modelo1.predict(x_test7)

mse_7 = mean_squared_error(y_test7, y_pred7)
r2_7 = r2_score(y_test7, y_pred7)

# 3-
x_train8, x_test8, y_train8, y_test8 = train_test_split(df[x8], y, test_size=0.2, random_state=42)
components = 3
pca_final = PCA(n_components=components)
x_train_pca = pca_final.fit_transform(x_train8)
x_test_pca = pca_final.transform(x_test8)

modelo1.fit(x_train_pca, y_train8)
y_pred8 = modelo1.predict(x_test_pca)

mse_8 = mean_squared_error(y_test8, y_pred8)
r2_8 = r2_score(y_test8, y_pred8)


'''REGRESIONS LOGÍSTIQUES AMB LES 3 SELECCIONS:'''
modelo2 = LogisticRegression()
# 1-
x_train9, x_test9, y_train9, y_test9 = train_test_split(x7, y, test_size=0.2, random_state=42)
modelo2.fit(x_train9, y_train9)
y_pred9 = modelo2.predict(x_test9)

exact9 = accuracy_score(y_test9, y_pred9)
precision9 = precision_score(y_test9, y_pred9, average='weighted')
recall9 = recall_score(y_test9, y_pred9, average='weighted')
f19 = f1_score(y_test9, y_pred9, average='weighted')

# 2-
x_train10, x_test10, y_train10, y_test10 = train_test_split(df[x8], y, test_size=0.2, random_state=42)
modelo2.fit(x_train10, y_train10)
y_pred10 = modelo2.predict(x_test10)

exact10 = accuracy_score(y_test1, y_pred5)
precision10 = precision_score(y_test10, y_pred10, average='weighted')
recall10 = recall_score(y_test10, y_pred10, average='weighted')
f110 = f1_score(y_test10, y_pred10, average='weighted')

# 3-
x_train11, x_test11, y_train11, y_test11 = train_test_split(df[x8], y, test_size=0.2, random_state=42)
modelo2.fit(x_train11, y_train11)
pca_final = PCA(n_components=3)
x_train_pca = pca_final.fit_transform(x_train11)
x_test_pca = pca_final.transform(x_test11)

modelo2.fit(x_train_pca, y_train11)
y_pred11 = modelo2.predict(x_test_pca)



exact11 = accuracy_score(y_test11, y_pred11)
precision11 = precision_score(y_test11, y_pred11, average='weighted')
recall11 = recall_score(y_test11, y_pred11, average='weighted')
f111 = f1_score(y_test11, y_pred11, average='weighted')

data = {'Mètrica': ['MSE', 'R^2', 'Exactitut', 'Precisió', 'Recall', 'Valor F1'],
        'Regressió Múltiple 1 (6)': [mse_1, r2_1, '-', '-', '-', '-'],
        'Regressió Múltiple 2 (5)': [mse_2, r2_2, '-', '-', '-', '-'],
        'Regressió Múltiple 3 (2)': [mse_3, r2_3, '-', '-', '-', '-'],
        'Regressió Logística 1 (6)': ['-', '-', exact1, precision1, recall1, f11],
        'Regressió Logística 2 (5)': ['-', '-', exact2, precision2, recall2, f12],
        'Regressió Logística 3 (2)': ['-', '-', exact3, precision3, recall3, f13]}

taula_metriques = pd.DataFrame(data)
taulell = tabulate(taula_metriques, headers='keys', tablefmt='pipe', showindex=False)
print(taulell)
print('-')
data2 = {'Mètrica': ['MSE', 'R^2', 'Exactitut', 'Precisió', 'Recall', 'Valor F1'],
        'Regressió Múltiple 1 (9)': [mse_6, r2_6, '-', '-', '-', '-'],
        'Regressió Múltiple 2 (8)': [mse_7, r2_7, '-', '-', '-', '-'],
        'Regressió Múltiple 3 (3)': [mse_8, r2_8, '-', '-', '-', '-'],
        'Regressió Logística 1 (9)': ['-', '-', exact9, precision9, recall9, f19],
        'Regressió Logística 2 (8)': ['-', '-', exact10, precision10, recall10, f110],
        'Regressió Logística 3 (3)': ['-', '-', exact11, precision11, recall11, f111]}

taula_metriques2 = pd.DataFrame(data2)
taulell2 = tabulate(taula_metriques2, headers='keys', tablefmt='pipe', showindex=False)
print(taulell2)



'''
TEORIA DE JOCS
'''
# 1- x_test2, y_test2
#Rangs escollits a partir d'una distribució de freqüències a l'hora de fer l'exploració del dataset.
x_train2['age_category'] = pd.cut(x_train2['age'], bins=(30, 49, 70), labels=['30-49', '50-70'])
#print(x_train2['age_category'].value_counts())
x_train2['glucose_level'] = pd.cut(x_train2['glucose'], bins=[-1, 69, 99, 1000], labels=['1- baixa', '2- òptima', '3- alta'])
x_train2['sysBP_level'] = pd.cut(x_train2['sysBP'], bins=[-1, 119, 140, 1000], labels=['1- òptima', '2- elevada', '3- molt elevada'])
#print(x_test2['sysBP_level'].value_counts())
#x_test2['cigsPerDay'].describe()
x_train2['cigs_level'] = pd.cut(x_train2['cigsPerDay'], bins=[-1, 0, 4, 19, 1000], labels=['1- No fumador', '2- Fumador lleuger', '3- Fumador moderat', '4- molt fumador'])
print(x_train2['cigs_level'].value_counts())


taula1 = pd.concat([x_train2, y_train2], axis=1)

df_pagos, categorias1, categorias2 = taula_TG(taula1, 'sysBP_level', 'glucose_level')

if table1:
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_pagos, annot=True, cmap='Reds', cbar=True)
    plt.xlabel('Glucosa')
    plt.ylabel('Préssió sistòlica')
    plt.title("Probabilitats de patir una enfermetat cardiovascular en funció dels nivells de glucosa i de la "
              "pressió sistòlica.")
    plt.xticks(np.arange(len(categorias2)) + 0.5, categorias2)
    plt.yticks(np.arange(len(categorias1)) + 0.5, categorias1, rotation=0)
    plt.show()


''' PER VEURE PERQUÈ LA GLUCOSA AFECTA TANT POC'''
tabla_frecuencias = (pd.crosstab(taula1['glucose_level'], df['TenYearCHD']))
porcentajes = tabla_frecuencias.apply(lambda x: x / x.sum() * 100, axis=1)
print(porcentajes)



grups = taula1.groupby('age_category')

grupo_1 = grups.get_group('30-49')
grupo_2 = grups.get_group('50-70')

if table2:
    df_pagos, categorias1, categorias2 = taula_TG(grupo_1, 'sysBP_level', 'cigs_level')

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_pagos, annot=True, cmap='Reds', cbar=True)
    plt.xlabel('Nivell de cigarros al dia')
    plt.ylabel('Nivell de la Pressió Sistòlica')
    plt.title("Nombre de cigarros al dia - Pressió Sistòlica (Menors de 50 anys).")
    plt.xticks(np.arange(len(categorias2)) + 0.5, categorias2)
    plt.yticks(np.arange(len(categorias1)) + 0.5, categorias1, rotation=0)
    plt.show()


    df_pagos, categorias1, categorias2 = taula_TG(grupo_2, 'sysBP_level', 'cigs_level')

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_pagos, annot=True, cmap='Reds', cbar=True)
    plt.xlabel('Nivell de cigarros al dia')
    plt.ylabel('Nivell de la Pressió Sistòlica')
    plt.title("Nombre de cigarros al dia - Pressió Sistòlica (Majors de 50 anys).")
    plt.xticks(np.arange(len(categorias2)) + 0.5, categorias2)
    plt.yticks(np.arange(len(categorias1)) + 0.5, categorias1, rotation=0)
    plt.show()











# 2- x_test10, y_test10
#Rangs escollits a partir d'una distribució de freqüències a l'hora de fer l'exploració del dataset.
x_train10['age_category'] = pd.cut(x_train10['age'], bins=(30, 49, 70), labels=['30-49', '50-70'])
#print(x_test2['age_category'].value_counts())

x_train10['colesterol_level'] = pd.cut(x_train10['totChol'], bins=[-1, 199, 249, 1000], labels=['1- òptim', '2- elevat', '3- molt elevat'])
#print(x_test10['colesterol_level'].value_counts())

taula2 = pd.concat([x_train10, y_train10], axis=1)
grups = taula2.groupby('age_category')

grupo_1 = grups.get_group('30-49')
grupo_2 = grups.get_group('50-70')

if table4:
    df_pagos, categorias1, categorias2 = taula_TG(grupo_1, 'prevalentHyp', 'diabetes')
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_pagos, annot=True, cmap='Reds', cbar=True)
    plt.xlabel('Diabetis')
    plt.ylabel('Hipertens')
    plt.title("Hipertens-diabetis")
    plt.xticks(np.arange(len(categorias2)) + 0.5, categorias2)
    plt.yticks(np.arange(len(categorias1)) + 0.5, categorias1, rotation=0)
    plt.show()

    df_pagos, categorias1, categorias2 = taula_TG(grupo_2, 'prevalentHyp', 'diabetes')
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_pagos, annot=True, cmap='Reds', cbar=True)
    plt.xlabel('Diabetis')
    plt.ylabel('Hipertens')
    plt.title("Hipertens-diabetis")
    plt.xticks(np.arange(len(categorias2)) + 0.5, categorias2)
    plt.yticks(np.arange(len(categorias1)) + 0.5, categorias1, rotation=0)
    plt.show()


if table5:
    df_pagos, categorias1, categorias2 = taula_TG(grupo_1, 'is_smoking', 'colesterol_level')
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_pagos, annot=True, cmap='YlOrBr', cbar=True)
    plt.xlabel('Nivell de colesterol')
    plt.ylabel('Fumador')
    plt.title("Fumador-nivel colesterol")
    plt.xticks(np.arange(len(categorias2)) + 0.5, categorias2)
    plt.yticks(np.arange(len(categorias1)) + 0.5, categorias1, rotation=0)
    plt.show()

    df_pagos, categorias1, categorias2 = taula_TG(grupo_2, 'is_smoking', 'colesterol_level')
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_pagos, annot=True, cmap='YlOrBr', cbar=True)
    plt.xlabel('Nivell de colesterol')
    plt.ylabel('Fumador')
    plt.title("Fumador-nivel colesterol")
    plt.xticks(np.arange(len(categorias2)) + 0.5, categorias2)
    plt.yticks(np.arange(len(categorias1)) + 0.5, categorias1, rotation=0)
    plt.show()

x_test10['age_category'] = pd.cut(x_test10['age'], bins=(30, 49, 70), labels=['30-49', '50-70'])
#print(x_test2['age_category'].value_counts())

x_test10['colesterol_level'] = pd.cut(x_test10['totChol'], bins=[-1, 199, 249, 1000], labels=['1- òptim', '2- elevat', '3- molt elevat'])
#print(x_test10['colesterol_level'].value_counts())

taula2 = pd.concat([x_test10, y_test10], axis=1)
grups = taula2.groupby('age_category')

grupo_1 = grups.get_group('30-49')
grupo_2 = grups.get_group('50-70')

tabla_frecuencias = (pd.crosstab(grupo_1['is_smoking'], df['TenYearCHD']))
porcentajes = tabla_frecuencias.apply(lambda x: x / x.sum() * 100, axis=1)
print(porcentajes)
tabla_frecuencias = (pd.crosstab(grupo_2['is_smoking'], df['TenYearCHD']))
porcentajes = tabla_frecuencias.apply(lambda x: x / x.sum() * 100, axis=1)
print(porcentajes)

tabla_frecuencias = (pd.crosstab(grupo_1['colesterol_level'], df['TenYearCHD']))
porcentajes = tabla_frecuencias.apply(lambda x: x / x.sum() * 100, axis=1)
print(porcentajes)
tabla_frecuencias = (pd.crosstab(grupo_2['colesterol_level'], df['TenYearCHD']))
porcentajes = tabla_frecuencias.apply(lambda x: x / x.sum() * 100, axis=1)
print(porcentajes)



# 1- x_test11, y_test11
#Rangs escollits a partir d'una distribució de freqüències a l'hora de fer l'exploració del dataset.
x_train11['age_category'] = pd.cut(x_train11['age'], bins=(30, 49, 70), labels=['30-49', '50-70'])
#print(x_test2['age_category'].value_counts())


taula2 = pd.concat([x_train11, y_train11], axis=1)
grups = taula2.groupby('age_category')

grupo_1 = grups.get_group('30-49')
grupo_2 = grups.get_group('50-70')

print(grupo_1['prevalentHyp'].value_counts())
print(grupo_2['prevalentHyp'].value_counts())


if table3:
    df_pagos, categorias1, categorias2 = taula_TG(grupo_1, 'is_smoking', 'prevalentHyp')
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_pagos, annot=True, cmap='YlOrBr', cbar=True)
    plt.xlabel('Hipertens')
    plt.ylabel('Fumador')
    plt.title("Fumador-Hipertens (menors de 50 anys)")
    plt.xticks(np.arange(len(categorias2)) + 0.5, categorias2)
    plt.yticks(np.arange(len(categorias1)) + 0.5, categorias1, rotation=0)
    plt.show()

    df_pagos, categorias1, categorias2 = taula_TG(grupo_2, 'is_smoking', 'prevalentHyp')
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_pagos, annot=True, cmap='Reds', cbar=True)
    plt.xlabel('Hipertens')
    plt.ylabel('Fumador')
    plt.title("Fumador-Hipertens (majors de 50 anys)")
    plt.xticks(np.arange(len(categorias2)) + 0.5, categorias2)
    plt.yticks(np.arange(len(categorias1)) + 0.5, categorias1, rotation=0)
    plt.show()





