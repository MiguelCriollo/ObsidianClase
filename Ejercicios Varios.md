Estos ejercicios tratan sobre el machine learning y tambien sobre la solucion de algunos errores simples.


2. **Introducción**: Machine Learning, que es una rama de la inteligencia artificial que se centra en la construcción de sistemas que pueden aprender de los datos. El proyecto específico mencionado es predecir la supervivencia de los pasajeros del Titanic, que es un problema de clasificación binaria (sobrevivir o no sobrevivir).

3. **Preparación del entorno de programación**: Aquí, el autor habla sobre las bibliotecas de Python que se utilizarán en el proyecto. Estas bibliotecas proporcionan funciones y métodos que facilitan la manipulación de datos, el análisis estadístico y la implementación de algoritmos de Machine Learning.

In [108]:

​

# Se importan las librerías de Python que se van a utilizar en el programa

​

import numpy as np  # numpy es una librería para el manejo de arrays y matrices de gran tamaño y funciones matemáticas de alto nivel
```python
import pandas as pd  # pandas es una librería para la manipulación y análisis de datos

from sklearn.model_selection import train_test_split  # Esta función de sklearn se utiliza para dividir arrays o matrices en subconjuntos aleatorios de entrenamiento y prueba

from sklearn.linear_model import LogisticRegression  # LogisticRegression es un algoritmo de clasificación que se utiliza en este proyecto

from sklearn.svm import SVC  # SVC (Support Vector Classification) es otro algoritmo de clasificación que se utiliza en este proyecto

from sklearn.neighbors import KNeighborsClassifier  # KNeighborsClassifier es un algoritmo de clasificación basado en los k vecinos más cercanos
```



#Este código importa las bibliotecas y funciones necesarias para el proyecto. Estas bibliotecas proporcionan las herramientas necesarias para manipular los datos, implementar los algoritmos de Machine Learning y dividir los datos en conjuntos de entrenamiento y prueba.

3. **Importación y exploración de datos**: En esta sección, el autor describe cómo importar datos desde una fuente externa (en este caso, Kaggle) y cómo explorar estos datos. La exploración de datos es un paso crucial en cualquier proyecto de Machine Learning, ya que permite entender la estructura y las características de los datos.


# Importar los datos desde la página de Kaggle
```python
url_test = 'test.csv'

url_train = 'train.csv'

df_test = pd.read_csv(url_test)  # Leer el archivo CSV de los datos de prueba y almacenarlo en un DataFrame de pandas

df_train = pd.read_csv(url_train)  # Leer el archivo CSV de los datos de entrenamiento y almacenarlo en un DataFrame de pandas

```

​

# Guardar los datos en un archivo local para tenerlos siempre disponibles

```python

dir_test = 'titanic_test.csv'

dir_train = 'titanic_train.csv'

df_test.to_csv(dir_test)  # Guardar el DataFrame de los datos de prueba en un archivo CSV

df_train.to_csv(dir_train)  # Guardar el DataFrame de los datos de entrenamiento en un archivo CSV
​
```


# Importar los datos de los archivos .csv almacenados

```python
df_test = pd.read_csv(dir_test)  # Leer el archivo CSV de los datos de prueba y almacenarlo en un DataFrame de pandas

df_train = pd.read_csv(dir_train)  # Leer el archivo CSV de los datos de entrenamiento y almacenarlo en un DataFrame de pandas
```

​

# Verificar la cantidad de datos que hay en los conjuntos de datos

```python
print('Cantidad de datos:')

print(df_train.shape)  # Imprimir la cantidad de filas y columnas en el conjunto de datos de entrenamiento

print(df_test.shape)  # Imprimir la cantidad de filas y columnas en el conjunto de datos de prueba

​
```

# Verificar el tipo de datos contenidos en ambos conjuntos de datos

```python
print('Tipos de datos:')

print(df_train.info())  # Imprimir información sobre el conjunto de datos de entrenamiento, incluyendo el tipo de datos de cada columna

print(df_test.info())  # Imprimir información sobre el conjunto de datos de prueba, incluyendo el tipo de datos de cada columna
```

​

# Verificar los datos faltantes de los conjuntos de datos

```python
print('Datos faltantes:')

print(pd.isnull(df_train).sum())  # Imprimir la cantidad de datos faltantes en cada columna del conjunto de datos de entrenamiento

print(pd.isnull(df_test).sum())  # Imprimir la cantidad de datos faltantes en cada columna del conjunto de datos de prueba
```

​

# Verificar las estadísticas del conjunto de datos

```python
print('Estadísticas del conjunto de datos:')

print(df_train.describe())  # Imprimir estadísticas descriptivas del conjunto de datos de entrenamiento

print(df_test.describe())  # Imprimir estadísticas descriptivas del conjunto de datos de prueba

​
```

Cantidad de datos:
(891, 13)
(418, 12)
Tipos de datos:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 13 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   Unnamed: 0   891 non-null    int64  
 1   PassengerId  891 non-null    int64  
 2   Survived     891 non-null    int64  
 3   Pclass       891 non-null    int64  
 4   Name         891 non-null    object 
 5   Sex          891 non-null    object 
 6   Age          714 non-null    float64
 7   SibSp        891 non-null    int64  
 8   Parch        891 non-null    int64  
 9   Ticket       891 non-null    object 
 10  Fare         891 non-null    float64
 11  Cabin        204 non-null    object 
 12  Embarked     889 non-null    object 
dtypes: float64(2), int64(6), object(5)
memory usage: 90.6+ KB
None
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   Unnamed: 0   418 non-null    int64  
 1   PassengerId  418 non-null    int64  
 2   Pclass       418 non-null    int64  
 3   Name         418 non-null    object 
 4   Sex          418 non-null    object 
 5   Age          332 non-null    float64
 6   SibSp        418 non-null    int64  
 7   Parch        418 non-null    int64  
 8   Ticket       418 non-null    object 
 9   Fare         417 non-null    float64
 10  Cabin        91 non-null     object 
 11  Embarked     418 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 39.3+ KB
None
Datos faltantes:
Unnamed: 0       0
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
Unnamed: 0       0
PassengerId      0
Pclass           0
Name             0
Sex              0
Age             86
SibSp            0
Parch            0
Ticket           0
Fare             1
Cabin          327
Embarked         0
dtype: int64
Estadísticas del conjunto de datos:
       Unnamed: 0  PassengerId    Survived      Pclass         Age  \
count  891.000000   891.000000  891.000000  891.000000  714.000000   
mean   445.000000   446.000000    0.383838    2.308642   29.699118   
std    257.353842   257.353842    0.486592    0.836071   14.526497   
min      0.000000     1.000000    0.000000    1.000000    0.420000   
25%    222.500000   223.500000    0.000000    2.000000   20.125000   
50%    445.000000   446.000000    0.000000    3.000000   28.000000   
75%    667.500000   668.500000    1.000000    3.000000   38.000000   
max    890.000000   891.000000    1.000000    3.000000   80.000000   

            SibSp       Parch        Fare  
count  891.000000  891.000000  891.000000  
mean     0.523008    0.381594   32.204208  
std      1.102743    0.806057   49.693429  
min      0.000000    0.000000    0.000000  
25%      0.000000    0.000000    7.910400  
50%      0.000000    0.000000   14.454200  
75%      1.000000    0.000000   31.000000  
max      8.000000    6.000000  512.329200  
       Unnamed: 0  PassengerId      Pclass         Age       SibSp  \
count  418.000000   418.000000  418.000000  332.000000  418.000000   
mean   208.500000  1100.500000    2.265550   30.272590    0.447368   
std    120.810458   120.810458    0.841838   14.181209    0.896760   
min      0.000000   892.000000    1.000000    0.170000    0.000000   
25%    104.250000   996.250000    1.000000   21.000000    0.000000   
50%    208.500000  1100.500000    3.000000   27.000000    0.000000   
75%    312.750000  1204.750000    3.000000   39.000000    1.000000   
max    417.000000  1309.000000    3.000000   76.000000    8.000000   

            Parch        Fare  
count  418.000000  417.000000  
mean     0.392344   35.627188  
std      0.981429   55.907576  
min      0.000000    0.000000  
25%      0.000000    7.895800  
50%      0.000000   14.454200  
75%      0.000000   31.500000  
max      9.000000  512.329200  

4. **Preprocesamiento de datos**: Aquí, se explica cómo preparar los datos para el análisis. Esto incluye la conversión de datos categóricos a numéricos (por ejemplo, convertir el género de los pasajeros de 'masculino' y 'femenino' a 1 y 0), el manejo de datos faltantes (por ejemplo, reemplazando los valores faltantes de la edad por la media de las edades), la creación de grupos de edades y la eliminación de columnas innecesarias.

In [126]:

# Cambiar los datos de sexos a números

```python
df_train['Sex'].replace(['female','male'],[0,1],inplace=True)  # Reemplazar 'female' por 0 y 'male' por 1 en la columna 'Sex' del conjunto de entrenamiento

df_test['Sex'].replace(['female','male'],[0,1],inplace=True)  # Hacer lo mismo para el conjunto de prueba
```

​

# Cambiar los datos de embarque en números

```python
df_train['Embarked'].replace(['Q','S', 'C'],[0,1,2],inplace=True)  # Reemplazar 'Q' por 0, 'S' por 1 y 'C' por 2 en la columna 'Embarked' del conjunto de entrenamiento

df_test['Embarked'].replace(['Q','S', 'C'],[0,1,2],inplace=True)  # Hacer lo mismo para el conjunto de prueba
```

​

# Reemplazar los datos faltantes en la edad por la media de esta columna

​

```python
df_train["Age"]=df_train["Age"].astype('float').replace(0, np.NaN)

df_test["Age"]=df_test["Age"].astype('float').replace(0, np.NaN)

print(df_train["Age"])

print(df_test["Age"])

​

print(df_train["Age"].mean(skipna=True))  # Imprimir la media de la columna 'Age' en el conjunto de entrenamiento

print(df_test["Age"].mean(skipna=True))  # Hacer lo mismo para el conjunto de prueba

promedio = 30  # Definir el valor promedio

df_train['Age'] = df_train['Age'].replace(np.nan, promedio)  # Reemplazar los valores faltantes en la columna 'Age' del conjunto de entrenamiento por el valor promedio

df_test['Age'] = df_test['Age'].replace(np.nan, promedio)  # Hacer lo mismo para el conjunto de prueba
```
​

# Crear varios grupos de acuerdo a bandas de las edades

# Bandas: 0-8, 9-15, 16-18, 19-25, 26-40, 41-60, 61-100

```python
bins = [0, 8, 15, 18, 25, 40, 60, 100]  # Definir las bandas de edades

names = ['1', '2', '3', '4', '5', '6', '7']  # Definir los nombres de los grupos

df_train['Age'] = pd.cut(df_train['Age'], bins, labels = names)  # Crear los grupos de edades en el conjunto de entrenamiento

df_test['Age'] = pd.cut(df_test['Age'], bins, labels = names)  # Hacer lo mismo para el conjunto de prueba

​

print(df_train)
```
# Eliminar la columna de "Cabin" ya que tiene muchos datos perdidos

```python
df_train.drop(['Cabin'], axis = 1, inplace=True)  # Eliminar la columna 'Cabin' del conjunto de entrenamiento

df_test.drop(['Cabin'], axis = 1, inplace=True)  # Hacer lo mismo para el conjunto de prueba

​
```

# Eliminar las columnas que no se consideran necesarias para el análisis
```python
df_train = df_train.drop(['PassengerId','Name','Ticket'], axis=1)  # Eliminar las columnas 'PassengerId', 'Name' y 'Ticket' del conjunto de entrenamiento

df_test = df_test.drop(['Name','Ticket'], axis=1)  # Hacer lo mismo para el conjunto de prueba
```
​

# Eliminar las filas con los datos perdidos

```python
df_train.dropna(axis=0, how='any', inplace=True)  # Eliminar las filas con datos faltantes en el conjunto de entrenamiento

df_test.dropna(axis=0, how='any', inplace=True)  # Hacer lo mismo para el conjunto de prueba

​
```
# Verificar los datos

```python
print(pd.isnull(df_train).sum())  # Imprimir la cantidad de datos faltantes en cada columna del conjunto de entrenamiento

print(pd.isnull(df_test).sum())  # Hacer lo mismo para el conjunto de prueba

print(df_train.shape)  # Imprimir la cantidad de filas y columnas en el conjunto de entrenamiento

print(df_test.shape)  # Hacer lo mismo para el conjunto de prueba

print(df_test.head())  # Imprimir las primeras filas del conjunto de prueba

print(df_train.head())  # Hacer lo mismo para el conjunto de entrenamiento

​
```

0      22.0
1      38.0
2      26.0
3      35.0
4      35.0
       ... 
886    27.0
887    19.0
888     NaN
889    26.0
890    32.0
Name: Age, Length: 891, dtype: float64
0      34.5
1      47.0
2      62.0
3      27.0
4      22.0
       ... 
413     NaN
414    39.0
415    38.5
416     NaN
417     NaN
Name: Age, Length: 418, dtype: float64
29.69911764705882
30.272590361445783
     Unnamed: 0  PassengerId  Survived  Pclass  \
0             0            1         0       3   
1             1            2         1       1   
2             2            3         1       3   
3             3            4         1       1   
4             4            5         0       3   
..          ...          ...       ...     ...   
886         886          887         0       2   
887         887          888         1       1   
888         888          889         0       3   
889         889          890         1       1   
890         890          891         0       3   

                                                  Name  Sex Age  SibSp  Parch  \
0                              Braund, Mr. Owen Harris    1   4      1      0   
1    Cumings, Mrs. John Bradley (Florence Briggs Th...    0   5      1      0   
2                               Heikkinen, Miss. Laina    0   5      0      0   
3         Futrelle, Mrs. Jacques Heath (Lily May Peel)    0   5      1      0   
4                             Allen, Mr. William Henry    1   5      0      0   
..                                                 ...  ...  ..    ...    ...   
886                              Montvila, Rev. Juozas    1   5      0      0   
887                       Graham, Miss. Margaret Edith    0   4      0      0   
888           Johnston, Miss. Catherine Helen "Carrie"    0   5      1      2   
889                              Behr, Mr. Karl Howell    1   5      0      0   
890                                Dooley, Mr. Patrick    1   5      0      0   

               Ticket     Fare Cabin  Embarked  
0           A/5 21171   7.2500   NaN       1.0  
1            PC 17599  71.2833   C85       2.0  
2    STON/O2. 3101282   7.9250   NaN       1.0  
3              113803  53.1000  C123       1.0  
4              373450   8.0500   NaN       1.0  
..                ...      ...   ...       ...  
886            211536  13.0000   NaN       1.0  
887            112053  30.0000   B42       1.0  
888        W./C. 6607  23.4500   NaN       1.0  
889            111369  30.0000  C148       2.0  
890            370376   7.7500   NaN       0.0  

[891 rows x 13 columns]
Unnamed: 0    0
Survived      0
Pclass        0
Sex           0
Age           0
SibSp         0
Parch         0
Fare          0
Embarked      0
dtype: int64
Unnamed: 0     0
PassengerId    0
Pclass         0
Sex            0
Age            0
SibSp          0
Parch          0
Fare           0
Embarked       0
dtype: int64
(889, 9)
(417, 9)
   Unnamed: 0  PassengerId  Pclass  Sex Age  SibSp  Parch     Fare  Embarked
0           0          892       3    1   5      0      0   7.8292         0
1           1          893       3    0   6      1      0   7.0000         1
2           2          894       2    1   7      0      0   9.6875         0
3           3          895       3    1   5      0      0   8.6625         1
4           4          896       3    0   4      1      1  12.2875         1
   Unnamed: 0  Survived  Pclass  Sex Age  SibSp  Parch     Fare  Embarked
0           0         0       3    1   4      1      0   7.2500       1.0
1           1         1       1    0   5      1      0  71.2833       2.0
2           2         1       3    0   5      0      0   7.9250       1.0
3           3         1       1    0   5      1      0  53.1000       1.0
4           4         0       3    1   5      0      0   8.0500       1.0

5. **Aplicación de algoritmos de Machine Learning**: En esta sección, el autor describe cómo implementar y entrenar tres algoritmos de Machine Learning: Regresión Logística, Máquinas de Vectores de Soporte (SVM) y Vecinos más Cercanos (KNN). Estos algoritmos se utilizan para crear modelos que pueden predecir si un pasajero sobrevivirá o no basándose en sus características.

In [137]:

# Separar los datos de "train" en entrenamiento y prueba para probar los algoritmos

```python
X = np.array(df_train.loc[:,df_train.columns!="Survived"])  # Crear un array con los datos de las características (todos los datos del conjunto de entrenamiento excepto la columna 'Survived')

y = np.array(df_train['Survived'])  # Crear un array con los datos de la variable objetivo (la columna 'Survived' del conjunto de entrenamiento)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # Dividir los datos en un conjunto de entrenamiento y un conjunto de prueba

print(X)

print(y)
```

# Regresión logística

```python
logreg = LogisticRegression()  # Crear una instancia del algoritmo de Regresión Logística

logreg.fit(X_train, y_train)  # Entrenar el algoritmo con los datos de entrenamiento

Y_pred = logreg.predict(X_test)  # Hacer predicciones con los datos de prueba

print('Precisión Regresión Logística:')

print(logreg.score(X_train, y_train))  # Imprimir la precisión del algoritmo en los datos de entrenamiento
```

​

# Support Vector Machines
```python
svc = SVC()  # Crear una instancia del algoritmo de Máquinas de Vectores de Soporte

svc.fit(X_train, y_train)  # Entrenar el algoritmo con los datos de entrenamiento

Y_pred = svc.predict(X_test)  # Hacer predicciones con los datos de prueba

print('Precisión Soporte de Vectores:')

print(svc.score(X_train, y_train))  # Imprimir la precisión del algoritmo en los datos de entrenamiento
```
​

# K neighbors

```python
knn = KNeighborsClassifier(n_neighbors = 3)  # Crear una instancia del algoritmo de Vecinos más Cercanos con vecinos

knn.fit(X_train, y_train)  # Entrenar el algoritmo con los datos de entrenamiento

Y_pred = knn.predict(X_test)  # Hacer predicciones con los datos de prueba

print('Precisión Vecinos más Cercanos:')

print(knn.score(X_train, y_train))  # Imprimir la precisión del algoritmo en los datos de entrenamiento
```

​

[[0 3 1 ... 0 7.25 1.0]
 [1 1 0 ... 0 71.2833 2.0]
 [2 3 0 ... 0 7.925 1.0]
 ...
 [888 3 0 ... 2 23.45 1.0]
 [889 1 1 ... 0 30.0 2.0]
 [890 3 1 ... 0 7.75 0.0]]
[0 1 1 1 0 0 0 0 1 1 1 1 0 0 0 1 0 1 0 1 0 1 1 1 0 1 0 0 1 0 0 1 1 0 0 0 1
 0 0 1 0 0 0 1 1 0 0 1 0 0 0 0 1 1 0 1 1 0 1 0 0 0 0 0 1 1 0 1 0 0 0 0 0 1
 0 0 0 1 1 0 1 1 0 1 1 0 0 1 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 1 0 1 0 0
 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 1 0 0 0 0 1 0 0 1 0 0 0 0 1 1 0 0 0 1 0 0
 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 1 0
 1 1 0 0 1 0 1 1 1 1 0 0 1 0 0 0 0 0 1 0 0 1 1 1 0 1 0 0 0 1 1 0 1 0 1 0 0
 0 1 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 1 1 1
 0 1 0 0 0 0 0 1 1 1 0 1 1 0 1 1 0 0 0 1 0 0 0 1 0 0 1 0 1 1 1 1 0 0 0 0 0
 0 1 1 1 1 0 1 0 1 1 1 0 1 1 1 0 0 0 1 1 0 1 1 0 0 1 1 0 1 0 1 1 1 1 0 0 0
 1 0 0 1 1 0 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 1 1 1 1 1
 0 0 0 0 1 1 0 0 0 1 1 0 1 0 0 0 1 0 1 1 1 0 1 1 0 0 0 0 1 1 0 0 0 0 0 0 1
 0 0 0 0 1 0 1 0 1 1 0 0 0 0 0 0 0 0 1 1 0 1 1 1 1 0 0 1 0 1 0 0 1 0 0 1 1
 1 1 1 1 1 0 0 0 1 0 1 0 1 1 0 1 0 0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0 1 0 0
 0 1 1 0 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 1 1 0 1 1 0 1 1 0 0 1 0 1
 0 1 0 0 1 0 0 1 0 0 0 1 0 0 1 0 1 0 1 0 1 1 0 0 1 0 0 1 1 0 1 1 0 0 1 1 0
 1 0 1 1 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 1 1 0 1 1 1 0 0 0 1 0 1 0 0 0 1 0
 0 0 0 1 0 0 1 1 0 0 0 1 0 0 1 1 1 0 0 1 0 0 1 0 0 1 0 0 1 1 0 0 0 0 1 0 0
 1 0 1 0 0 1 0 0 0 0 0 1 0 1 1 1 0 1 0 1 0 1 0 1 0 0 0 0 0 0 1 0 0 0 1 0 0
 0 0 1 1 0 0 1 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 1 0 0 1 1 0 0
 0 0 1 1 1 1 1 0 1 0 0 0 1 1 0 0 1 0 0 0 1 0 1 1 0 0 1 0 0 0 0 0 0 1 0 0 1
 0 1 0 1 0 0 1 0 0 1 1 0 0 1 1 0 0 0 1 0 0 1 1 0 1 0 0 0 0 0 0 0 0 1 0 0 1
 0 1 1 1 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 1 0 0 0 1 1 1 1 0 0 0 0 1 0 0 0 0 0
 0 0 0 0 0 1 1 0 1 0 0 0 1 1 1 1 0 0 0 1 0 0 1 1 0 0 1 0 0 0 0 0 0 1 0 0 0
 1 0 1 1 1 1 0 0 0 1 0 0 1 1 0 0 1 0 1 0 0 1 1 0 0 0 1 1 0 0 0 0 0 0 1 0 1
 0]
Precisión Regresión Logística:
0.7890295358649789
Precisión Soporte de Vectores:

C:\ProgramData\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    [https://scikit-learn.org/stable/modules/preprocessing.html](https://scikit-learn.org/stable/modules/preprocessing.html)
Please also refer to the documentation for alternative solver options:
    [https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
  n_iter_i = _check_optimize_result(

0.6765119549929677
Precisión Vecinos más Cercanos:
0.7862165963431786

6. **Predicciones**: Aquí, se utiliza los modelos entrenados para hacer predicciones sobre un conjunto de datos de prueba. Esto permite evaluar la eficacia de los modelos en datos que no se utilizaron durante el entrenamiento.

In [138]:

# Predicciones con los modelos entrenados

​

# Predicciones con Regresión Logística
```python
ids = df_test['PassengerId']  # Guardar los IDs de los pasajeros del conjunto de prueba

predictions = logreg.predict(df_test.drop('PassengerId', axis=1).astype('float'))  # Hacer predicciones con el modelo de Regresión Logística y los datos de prueba (excepto los IDs de los pasajeros)

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })  # Crear un DataFrame con los IDs de los pasajeros y las predicciones

print('Predicciones Regresión Logística:')

print(output.head())  # Imprimir las primeras filas del DataFrame de salida
```
​

# Predicciones con Máquinas de Vectores de Soporte (SVM)

```python
predictions = svc.predict(df_test.drop('PassengerId', axis=1))  # Hacer predicciones con el modelo SVM y los datos de prueba (excepto los IDs de los pasajeros)

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })  # Crear un DataFrame con los IDs de los pasajeros y las predicciones

print('Predicciones Soporte de Vectores:')

print(output.head())  # Imprimir las primeras filas del DataFrame de salida
```
​

# Predicciones con Vecinos más Cercanos (KNN)

```python
predictions = knn.predict(df_test.drop('PassengerId', axis=1))  # Hacer predicciones con el modelo KNN y los datos de prueba (excepto los IDs de los pasajeros)

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })  # Crear un DataFrame con los IDs de los pasajeros y las predicciones

print('Predicciones Vecinos más Cercanos:')

print(output.head())  # Imprimir las primeras filas del DataFrame de salida

​
```

Predicciones Regresión Logística:
   PassengerId  Survived
0          892         0
1          893         0
2          894         0
3          895         0
4          896         0
Predicciones Soporte de Vectores:
   PassengerId  Survived
0          892         0
1          893         0
2          894         0
3          895         0
4          896         0
Predicciones Vecinos más Cercanos:
   PassengerId  Survived
0          892         0
1          893         0
2          894         0
3          895         0
4          896         1

C:\ProgramData\anaconda3\Lib\site-packages\sklearn\base.py:432: UserWarning: X has feature names, but LogisticRegression was fitted without feature names
  warnings.warn(
C:\ProgramData\anaconda3\Lib\site-packages\sklearn\base.py:432: UserWarning: X has feature names, but SVC was fitted without feature names
  warnings.warn(
C:\ProgramData\anaconda3\Lib\site-packages\sklearn\base.py:432: UserWarning: X has feature names, but KNeighborsClassifier was fitted without feature names
  warnings.warn(