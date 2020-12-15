# Introducción a Redes Neuronales


## Términos básicos
- **Modelo**: se entiende como la relacióne entre los atributos y la etiqueta.
- **Atributos**: son las variables de entrada a la red o modelo.
- **Clases**: es un set de posibles etiquetas a escoger en un problema de clasificación. De esta forma, si se están clasificando imagenes de perro y gatos, *perro* y *gato* son sus dos clases.
- **Label, etiqueta o target**: es el valor a predecir. Este puede ser una categoría o clasificación a predecir, como el tipo de animal en una imagen. Es una instancia específica de una clase, es decir, si una imagen tiene una clase específica *perro*, entonces *perro* es la etiqueta de esa imagen-ejemplo.
- **Ejemplo etiquetado**: es una instancia que incluye los atributos y su correspondiente etiqueta.
- **Ejemplo sin etiqueta**: es una instancia que incluye solo los atributos, sin la etiqueta correspondiente.
- **Mini-batch o Batch**: los minilotes o lotes de datos son pequeños set de muestras o ejemplos que son procesados simultáneamente por el modelo. El número de muestras o ejemplos es usualmente una potencia de 2, típicamente entre 8 a 128,  para facilitar la asignación de recursos en memoria. Durante el entrenamiento, un minilote (mini-batch) es usado para calcular el gradiente descendiente y actualizar los pesos del modelo.
Tamaños de minilotes más grandes ayudan a un entrenamiento y aprendizaje más rápido, pero requieren mayor espacio en memoria. Un buen tamaño por default es 32.
- **Entrenamiento**: es el proceso de aprendizaje gradual mediante el cual se relacionan los atributos y sus respectivas etiquetas.
- **Inferencia o predicción**: es la aplicación de un modelo entrenado a ejemplos sin etiqueta.
- **Clasificación**: es una tarea típica dentro del aprendizaje supervisado en donde se predicen *clases* dados ciertos atributos.
- **Regresión**: tarea típica dentro del aprendizaje supervisado en donde se predicen valores continuos, como el precio de una casa o un auto, dados ciertos atributos usados como *predictores*.



### Problemas lineales vs no-lineales
Los problemas dependiendo el tipo de datos, pueden ser clasificados en lineales y no lineales. Estos se describen a continuación.
- **Si un problema es lineal**, viene a significar que los datos pueden ser separados por una simple línea recta, y por lo tanto, el problema podría ser abordado con un simple Perceptron con función de activación lineal.
![Lineal](https://i.imgur.com/yqqzXUx.png)
- **Si el problema es no lineal**, los datos no pueden ser separados por una simple línea recta, por lo que se necesita más de una línea recta para separar los datos. De esta forma, este tipo de problemas debe ser abordados al menos mediante Perceptrones Multicapa, con posibles funciones de activación no lineales para mejores rendimientos.
![No Lineal](https://i.imgur.com/At2yhUV.png)


## Funciones de Activación
Las funciones de activación son a veces llamadas *funciones de transferencia* o *no linearidades*, porque transforman la combinación lineal de las entradas y sus pesos asociados a una forma no lineal.

Estas funciones de activación se ubican al final de cada Perceptron o neurona artificial, y es la forma de activar o no cada neurona.
- **¿Por qué se utilizan?**
  Se utilizan para introducir no linearidades, y de esta forma, dotar a la red de la capacidad de afrontar problemas no lineales. Sin una función de activación no-lineal, un Perceptron Multicapa sin importar su cantidad de capas ocultas, se comportará de forma similar a un simple Perceptron. Esto es debido a que la combinación de *funciones de activacion lineales*, es simplemente otra función lineal.
  Al mismo tiempo, se utilizan para restringir los valores de salida de cada neurona dentro de un rango finito.

Existe una gran cantidad de funciones de activación, sin embargo, las que se utilizan más comunmente son pocas. Entre las más usadas están las siguientes:

### Función Lineal
La función lineal, muchas veces utilizada como una *función identidad* pasando la señal inalterada. Es decir, la *salida* de la función de activación es igual a su entrada, y de esta forma, es como si no hubiera una función de activación. En el mejor de los casos, dependiendo el tipo de función lineal, sólo se escala la salida de la *función sumatoria*, sin la capacidad de transformar esta entrada en una función no lineal.

![Imgur](https://i.imgur.com/MYDv3zG.png)

### Función Step, Heaviside, o escalón
La función escalón unitario produce una salida binaria. Es una función simple, en la que básicamente:
- Si x>0, la salida es 1
- Si x<0, la salida es 0

Esta función es utilizada debido a su tipo de salida, en problemas de clasificación binaria del tipo verdadero o falso.
![Imgur](https://i.imgur.com/F0j1evV.png)

### Función Sigmoide o Logística
La función Sigmoide es una de las más comunmente utilizadas. Su uso es común en problemas de clasificación binaria en donde lo que se quiere es predecir la probabilidad de una clase *en problemas en donde existen 2 clases*.
La función toma todos los valores de entrada, y los reduce a un rango [0,1], convirtiendo valores continuos entre -infinito y +infinito en una simple probabilidad entre 0 y 1.

![Imgur](https://i.imgur.com/WOyjAH6.png)


#### Función Softmax
La función Softmax es una generalización de la **función sigmoide**, y por esto es utilizada en problemas de clasificación para obtener las probabilidades de una clase cuando *el problema tiene más de 2 clases*. Esta función de activación, fuerza la salida de la red en el rango 0 a 1. Para esto, implementa la siguiente ecuación:

La función Softmax es la función aconsejada para problemas de clasificación de más de dos clases, o donde existen sólo dos clases, en donde se comporta como una Sigmoide.



### Función Tangente Hiperbólica
La función tangente hiperbólica es una versión *desplazada* de la función sigmoide. De esta forma, en vez de reducir los valores de entrada a un rango [0,1], esta función lleva estos valores al rango [-1,1].
Esta función trabaja relativamente mejor que una función sigmoide en las *capas ocultas*, debido a que al llevar los valores al rango [-1,1] tiene el efecto de centrar los datos, haciendo que el promedio esté cercano a 0 (en la sigmoide el promedio está en 0.5), haciendo que el aprendizaje para la capa posterior (o siguiente) sea un poco más fácil.
La función Tangente Hiperbólica esta dada por:

Esta función, al igual que la función sigmoide, tiene como principal contra la saturación que ocurre para valores muy grandes (tanto positivos como negativos). Esto provoca que las derivadas locales, es decir, el *gradiente*, sea muy cercano a 0, y por lo tanto al momento de la *Propagación hacia atras* (**Backpropagation**) casi no exista gradiente a propagar.

![Imgur](https://i.imgur.com/flxLBtP.png)

### Función ReLU (Rectified Linear Unit)
La función de activación ReLU, activa un nodo o neurona sólo si la entrada es mayor a 0. Si la entrada es menor a 0, entonces la salida es siempre 0. Cuando la entrada es mayor a 0, el nodo o neurona se activa, y la salida es una relación lineal con la variable de entrada de la forma $f(x) = x$.

![Imgur](https://i.imgur.com/MXtRkkS.png)

Actualmente es muy utilizada debido a su buen funcionamiento para diferentes situaciones y problemas, teniendo incluso una tendencia a mejores entrenamientos para las capas ocultas que los realizados con la *función sigmoide* o *tanh*.

Como una de sus principales contras, sufre de un problema conocido como *"dying ReLU"*. Este problema consiste en la *muerte* de algunas neuronas durante el entrenamiento, dado que su salida, debido a esta función de activación, es siempre 0 y por lo tanto no llega a activarse. Esto puede ocurrir cuando los pesos de una neurona son actualizados de modo que la suma ponderada de entradas y pesos asociados es siempre menor a 0. Cuando esto pasa, la función de activación sólo genera como salida ceros, y el Gradiente Descendiente deja de tener efecto porque el gradiente de la función ReLU es cero cuando su entrada es negativa.

### Función Leaky ReLU
La función Leaky ReLU es una variante de la función de activación ReLU que viene a solucionar el problema conocido como *"dying ReLU"*. Para esto, en vez de tener una salida igual a 0 para entradas menos a 0, esta función introduce una pequeña pendiente típica de $0.01$. Esta pequeña pendiente es posible usarla como otro *hiperparámetro* a sintonizar.

Esta función está definida como:


![Imgur](https://i.imgur.com/wmH0Y3T.png)

El hiperparámetro *alpha* define la pendiente para la función cuando la entrada es x<0. Esta pequeña pendiente asegura que aún cuando una situación como *"dying ReLU"* suceda, la salida no sea totalmente 0, si no cercano a este, dejando siempre la posibilidad de una posible activación.





## Funciones de Pérdida
La función de pérdida, también conocida como *función de costo* o *función de error*, es una medida de *qué tan incorrecta* es una predicción de una red. Si esta pérdida es alta, el modelo no está haciendo un muy buen trabajo. Esto quiere decir que a *pérdida* más pequeña, mejor trabajo estará haciendo el modelo.
- **¿Por qué se necesitan?** porque el cálculo del error es un problema de optimización, y como tal, es necesario ajustar y optimizar sus parámetros para minimizar este el *error en la predicción*.

Entre las funciones de pérdida más usadas, están las siguientes:
### Mean square error (MSE)
Es el error cuadrático medio entre las predicciones y los valores reales. Es comunmente usada en problemas de regresión que requiere como salida un valor continuo.

### Mean Absolute Error (MAE)
El error medio absoluto calcula cuan lejos está la predicción de su valor real, tomando el valor absoluto y retornando el valor promedio. Es recomendada en problemas de regresión.

### Binary cross entropy
Es la función de pérdida recomendada para problemas de clasificación de dos clases.

### Categorical cross entropy
Función de pérdida  recomendada para problemas de clasificación de más de 2 clases.






## Optimizadores
Los optimizadores son algorítmos encargados de *encontrar* los pesos asociados que *minimicen la función de pérdida*. El algorítmo más popular es el Gradiente Descendiente que calcula el gradiente de la función de pérdida, para obtener la dirección de máximo crecimiento de la función, y así poder moverse en pequeños pasos en la dirección contraria. Cada *paso* o *step* dado en la dirección de mínimo crecimiento de la función de pérdida, es lo que se conoce como **Learning Rate** o **Tasa de Aprendizaje**.

### Stochastic Gradient Descent (SGD)

El *optimizador más simple*. Usa siempre el Learning Rate (Tasa de aprendizaje) como el multiplicador para gradientes.

### Momemtum
Acumula gradientes pasados de modo que la actulización de pesos-parámetros se hace más rápido si estos *gradiente pasados* se alinean con los *gradientes actualizados*. Si el gradiente comienza a cambiar mucho de dirección en cada actualización, la actualización de parámetros se ralentiza.

### RMSProp
Escala el factor multiplicativo de forma diferente para diferentes pesos-parámetros. Lo hace manteniendo un historial reciente del valor RMS (root mean square) de cada gradiente por peso.

### AdaDelta
Tiene un comportamiento parecido a RMSprop, debido a que escala la tasa de aprendizaje de forma individual para cada peso-parámetro.

### ADAM
El optimizador ADAM, es uno de los más utilizados, y puede ser entendido como una combinación entre una tasa de aprendizaje adaptativa como en AdaDelta y el método utilizado por el optimizador Momemtum.

### AdaMax
Es un optimizador similar a ADAM, que mantiene el rastro de las magnitudes de los gradientes.



















## Redes Neuronales
Las redes neuronales, tienen su inspiración en la neurona biológica. Es por esto, que el nombre correcto para referirse a ellas es "Redes Neuronales Artificiales".
Las redes neuronales artificiales están en el centro del Aprendizaje Profundo, y tienen como primer modelo el propuesto en 1943 por Pitts y McCulloch. Este primer modelo propuesto, conocido como "Threshold Logic Unit", era un modelo simple de una neurona biológica, de tipo binario, en donde cada neurona tenía un umbral prefijado. Tenía **una o más entradas binarias**, y **una salida binaria**. La neurona artificial activa su salida cuando al menos cierto número de entradas estan activas. Este modelo era capáz de aprender funciones de lógica binaria como AND y OR, y sirvió de base para modelos posteriores como el Perceptron y el Perceptron Multicapa.

### El Perceptron
Es un modelo propuesto en 1957 por Frank Rosenblatt, usado para clasificación binaria. Básicamente, es un tipo de neurona artificial en donde:
- Las entradas, a diferencia del TLU, son simple números en vez de valores binarios.
- La **función sumatoria** es la sumatoria lineal de las entradas o **producto puntro entre las entradas y sus pesos asociados**
$$MSE = \frac{1}{N} \sum (y - (prediccion(x)))^2$$
- La **función de activación** es la función escalón o **Heaviside** con un valor umbral típico de 0.5.
![Imgur](https://i.imgur.com/zUEbTR6.png)

![Estructura básica de una neurona](https://i.imgur.com/WS4gqql.jpg)

De esta forma, si la sumatoria lineal de las entradas, esto es, **el producto punto entre las entradas y sus pesos asociados** es:
- mayor al *valor umbral de la función escalón*, la salida del Perceptron será 1.
- menor al *valor umbral de la función escalón*, la salida del Perceptron será 0.

La lógica de aprendizaje del Perceptron es:
1. Se calcula la sumatoria entre entradas y sus pesos asociados. Esta *función sumatoria* es aplicada a la *función de activación* para generar una predicción **ŷ**.
Este proceso es llamado **FeedForward**.
2. Se compara la predicción hecha con la etiqueta correcta, para calcular el error:
error = y-ŷ
3. Se actualizan los pesos intentando minimizar el error. De esta forma se mejora la predicción intentando que el error sea lo más cercano a 0.
4. Se repite el proceso desde el paso 1.

**Una nota sobre el Perceptron**
- Como el perceptron implementa una sumatoria lineal de las entradas, **es en sí un modelo de función lineal**.
- Dado lo anterior, el Perceptron producirá una **línea recta** que "separa" o clasifica cierta parte de los datos.
- De esta forma, si el problema es lineal, o **linealmente separable**, es decir los datos pueden ser separados por una línea recta, el Perceptron funciona bien.
- Si los datos son no-lineales, el Perceptron fallará como modelo.


### El Perceptron Multicapa
El Multilayer Perceptron (MLP), o Perceptron Multicapa, son redes neuronales con una capa de entrada (Input Layers), una o más capas ocultas (Hidden Layers) y una capa de salida (Output Layer) compuestas de Perceptrones. Cada una de sus capas, a excepción de la capa de salida, tiene cada una de sus neuronas conectada a cada una de las neuronas de la capa siguiente.
![Imgur](https://i.imgur.com/To5a8yS.jpg)

Una red es **densa** o **Fully connected**, cuando cada uno de los nodos o neuronas de una capa está conectado a todos los nodos o neuronas de la siguiente capa. Esta arquitectura es conocida como **fully connected network** y es la arquitectura más básica de redes neuronales. Es posible referirse a ella usualmente como *Red Neuronal Artificial*, *Multilayer Perceptron* (MLP), *Fully connected network* o *Feedforward network*.
![Red Fully Connected](https://i.imgur.com/Fj7BwVo.png)

Cada neurona en el Perceptron Multicapa es similar al Perceptron, pero tiene la flexibilidad de elegir el tipo de función de activación a usar, y de esta forma añade la posibilidad de representar funciones de activación más complejas.

Al encadenar varios perceptrones, sólo se terminan obteniendo transformaciones lineales, incapaces de afrontar datos no-lineales. De esta forma, una red neuronal profunda con funciones de activación no-lineales puede teóricamente ser capáz de aproximar cualquier función continua, y con esto, afrontar y resolver problemas más complejos.

## El proceso de entrenamiento
El proceso de entrenamiento de una red neuronal, consiste básicamente en encontrar y aprender los valores de los pesos de todas las capas de la red que minimicen la función de pérdida, y por lo tanto, ajusten las predicciones de los datos de entrada con sus verdaderas etiquetas. Este proceso, es un proceso iterativo compuesto de una *propagación hacia adelante*, *forward propagation o feedforward*, y una *propagación hacia atrás*, *retropropagaión o backpropagation*.

En la primera fase, de *feedforward*, la información o datos de entrenamiento fluyen a través de la red desde la capa de entrada a la capa de salida en donde se calculan las predicciones. Estos datos pasan a través de la red, en donde cada neurona aplica su tranformación a la información que recibe de las neuronas de la capa anterior y la envía a las neuronas de la capa siguiente. Cuando se llega a la capa final, se genera una predicción de la etiqueta para los datos o ejemplos de entrada.

Con la predicción de la etiqueta, se utiliza una **Función de pérdida** como medida y estimación del error en la predicción. Esta **función de pérdida** tomas las predicciones que realiza la red y los valores reales, y calcula qué tan bueno o mala fue la predicción. De esta forma, se mide lo lejos que está cada predicción respecto a lo que se esperaba obtener.

Con la *estimación del error*, se inicia la propagación hacia atras o segunda fase llamada *backpropagation*, en donde se llama al **Optimizador** para la actualización y ajuste de los *pesos* de la red, en la dirección del gradiente que *reduce esta pérdida o error* y por lo tanto minimiza la **función de pérdida**.
Esta información se propaga hacia todas las neuronas que contribuyen directamente a la salida, basándose en la contribución relativa que haya aportado cada neurona.
El optimizador ajusta los pesos de cada neurona con la ayuda del cálculo del gradiente de la función de pérdida en pequeños pasos o *steps*, dado por el *Learning Rate* o *Tasa de Aprendizaje*. La actualización de los pesos se hace generalmente en lotes o batches, capa por capa, hasta que todas las neuronas de la red hayan sido actualizadas.

![Imgur](https://i.imgur.com/altrQa5.jpg)