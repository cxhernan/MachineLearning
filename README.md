#RECONOCIMIENTO FACIAL:
La idea general es que de una base de datos de fotos de dos personas (para este proyecto proporcionamos 27 de un hombre y 27 de una mujer), se escoja una cantidad de fotos 
(de forma balanceada) para entrenar un modelo de reconocimiento facial basado en EIGENFACES y SVM, y que el resto de fotos se usen de prueba para que el modelo prediga 
si cada foto de prueba que se le muestre es la mujer (de nombre CATA) o del hombre (de nombre CRIS).

Este proyecto contiene en la carpeta "datos", 3 achivos:
1. X.csv: es una matriz de 54x2500 donde cada fila corresponde a una imagen en escala de grises de 50x50 (vectorizada como si fuese unidimensional)
2. Y.csv: contiene una matriz columna de 54x1, donde la fila i corresponde a la clase a la que pertenece la imagen de la fila i de la matrix X, 0 si es CATA y 1 si es CRIS.  
3. P.csv: contiene la matriz de 2x2:[[0,'CATA'],[1,'CRIS']]

Este proyecto tiene 3 etapas principales:
1. Extracción de características: proyectamos cada imagen sobre 2 vectores extraidos de la matriz de autovectores de la matriz de covarianza de nuestros datos de entrenamiento,
y producto de esa proyección, para cada imagen tendremos asociado una coordenada en R^2 con información de las caracteristicas de dicha imagen. 
En el proyecto se han graficado esos puntos para las 54 imagenes para percibir que el modelo entiende las caracteristicas de los rostros, ya que los puntos que corresponden
a las fotos de CATA se agrupan entre ellos (por tener caracteristicas similares) y los de CRIS de igual manera, de modo que se forman dos clusters.

2. Clasificador: una vez que identificamos que se están formando clusteres, podemos encontrar una frontera de decisión que los separe, y para ello se usó SVM con kernel lineal,
el cual también es graficado.

3. Prueba del modelo: Se evalúan en el modelo las fotos de prueba, y se las muestra en pantalla con sus respectivas predicciones.

Comentario: el algoritmo de clasificación fué SVM puesto que teníamos datos etiquetados, pero también se hizo el experimento suponiendo que no conocemos la etiqueta de los datos,
solo que en ese caso se usó el algoritmo K-MEANS para clustering no supervisado.

Espero les sirva.
Les dejo mi correo por si tienen alguna duda: cxhernan@gmail.com.

