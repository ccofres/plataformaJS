# La Inteligencia Artificial y el Aprendizaje Automático

La Inteligencia Artificial o IA, es un campo de investigación y estudio que intenta comprender *cómo los seres humanos pensamos* y construir a partir de esto entidades o máquinas que muestren capacidades cognitivas, que perciban, entiendan, infieran o deduzcan, es decir, que demuestren *inteligencia*.

El Aprendizaje Automático o Machine Learning, es un subcampo de la Inteligencia Artificial, que intenta dotar a una máquina o sistema con la capacidad de aprender de datos sin haber sido explícitamente programada. Para esto se desarrollan técnicas con la capacidad de generalizar comportamientos y aprender patrones, y de esta forma mejorar, describir y predecir ciertos resultados.
![Inteligencia Artificial, Aprendizaje Automático y Aprendizaje Profundo](https://lh3.googleusercontent.com/pw/ACtC-3eUjuOaLqIzECUMt44lYj8az-xB7r7IpqXZkbjojEgDM0kY5u_0bqtj0SRq9A9jE0H3RUygJgqXNXnIcbJh_B4yA4C8S3Kr4AHtPOvpFNXuZhdLiaXFctcVnw__I0S-R16PuAnEScwcFoKt1i3gBIY=w608-h377-no?authuser=2)

En general, las técnicas o enfoques del Aprendizaje Automático se pueden dividir en 3 categorías:
 - **Aprendizaje Supervisado:** al sistema o máquina se le presenta un conjunto de datos o ejemplos de entrenamiento, compuestos por los valores de entrada y los valores de salida deseados. A partir de ello se busca generalizar un patrón mediante algún algoritmo para hacer predicciones de datos no conocidos fuera del conjunto de datos de entrenamiento.
 - **Aprendizaje No Supervisado:** el sistema se dota de un conjunto de entrenamiento compuesto sólo por los valores de entrada, sin los valores de salida deseados. Es decir, el conjunto de entrenamiento no contiene los resultados debidamente etiquetados, clasificados o categorizados para cada uno de los valores de entrada, por lo que el algorítmo debe aprender a realizar la clasificación o categorización de los datos sólo a partir de los valores de entrada.
- **Aprendizaje por Refuerzo:** el sistema o máquina a través de la interacción aprende lo bueno o malo de una acción a través del resultado obtenido. Si la acción o comportamiento es el correcto, por ejemplo, la recompensa puede ser positiva, en caso contrario, la recompensa será negativa.

# El Aprendizaje Profundo
El Deep Learning o Aprendizaje Profundo, es un subcampo del Machine Learning que utiliza como arquitectura fundamental *Redes Neuronales*. Utiliza este tipo de redes como forma para extraer información de los datos con el menor esfuerzo humano posible, intentando realizar este proceso de forma automática.

Gran parte de los conceptos básicos del *Aprendizaje Profundo* surgieron en los años 60, 80 y 90, pero ha tenido su mayor auge en la última década debido principalmente a factores tales como: 
- Digitalización de la información y la consiguiente habilidad de acceder a datos fácilmente, haciendo que muchos problemas tengan ahora una forma digital. 
- Grandes avances de las telecomunicaciones y en especial el internet, que le permiten a las comunidades científicas la capacidad de trabajar y compartir remotamente. 
- Grandes avances en la computación y el diseño de nuevo hardware (CPU, GPU, TPU), permitiendo la ejecución efectiva a gran escala. 
- Desarrollo de herramientas como TensorFlow, PyTorch y Keras con grandes niveles de abstracción que ayudan a las personas a resolver problemas en cada vez menos tiempo y con cada vez menos conocimientos, dejando a la *idea* y los *datos* como el punto central.

Para esto, utiliza un conjunto de datos de ejemplo como **base o set de entrenamiento** que se utiliza para reconocer patrones. Una vez que se extraen estos patrones, el sistema puede ser capáz de utilizarlos para *etiquetar* nuevos datos de entrada.
Las Redes Neuronales son un modelo basado en el funcionamiento del cerebro, diseñado para el reconocimiento de patrones. Estos patrones son numéricos y están contenidos en vectores, como representación de los datos recibidos como entrada.

## Estructura de las Redes Neuronales
Las redes neuronales están compuestas por capas, y cada una de estas capas está compuesta por nodos o neuronas. Los nodos tienen la siguiente estructura:
- Una o más **entradas** que reciben los datos a procesar.
- **Pesos** dados a cada una de las entradas. Estos aumentan o disminuyen la importancia de dicha entrada.
- **Función sumatoria**, encargada de sumar todas las combinaciones de peso-entrada.
- **Función de activación**, que determina la activación o no de un nodo según el valor obtenido en la función sumatoria. Esta función puede ser una simple función escalón, una función lineal que devuelve el mismo valor calculado o una función lineal por tramos, como la función ReLU, que devuelve un valor si el valor de la función sumatoria está dentro de ciertos límites, entre otras.

![Estructura básica de una neurona](https://lh3.googleusercontent.com/pw/ACtC-3cG9-rW0wdGRG27bKnOO5c1rlechA9AMrKlpISfTTDxx0loI9hHwgf5-Etlky-M6jDws9hgwwXosk0iCoGNCup2XfvQWy-GQok9btzkKeN8gWqfoLv0UDB2OBBi123eE_I0ylmXiDFBrEa_Zfh8z68=w695-h502-no?authuser=2)

![Capas de una red neuronal](https://lh3.googleusercontent.com/pw/ACtC-3f-9j5M1fKeUbTXbgnE4w0iljfMIs_1VGl0Rn1LEBcDKEn9ZY0bjgWCze_lusfBfT4JZ2bsYqTtTI7HfTpwiUSX7S0bOMGZZuYw5cnZhTPSgk-2vZeXuGjRQ4_EzXZMLzjIRNEHm_2aux_igNx0fac=w769-h409-no?authuser=2)


## Entrenamiento de una red neuronal

Para entender los modelos de redes neuronales y su entrenamiento, se deben definir los siguientes conceptos básicos:
- **Etiquetas:** tambień llamadas *labels* por su nombre en inglés. Corresponde al valor, clasificación o categoría a predecir.
- **Atributos:** también llamados *features*. Corresponden a las variables de entrada a la red.
- **Set de datos:** corresponde a los ejemplos a utilizar para entrenar o hacer predicciones con el modelo. 
Estos ejemplos pueden corresponder a atributos debidamente etiquetados, que suelen utilizarse como ejemplos de entrenamiento para un modelo, o pueden corresponder a atributos sin etiquetar que se utilizan para probar el modelo ya entrenado o en instancias de aprendizaje no supervisado.

Con el fin de extraer información de los datos, el modelo de red neuronal debe definir la relación entre las entradas o atributos y su salida o etiquetas. Para esto el modelo debe pasar por el proceso de entrenamiento o aprendizaje para posteriormente poder hacer inferencias de acuerdo a los patrones aprendidos durante el entrenamiento.
En el **proceso de entrenamiento** ocurre un ajuste o modificación de los pesos asociados a cada entrada a un nodo, con el fin de minimizar una **función de pérdida**. Esta **función de pérdida** recibe la predicción **ŷ** y la etiqueta correcta **y**, asociadas a cierto atributo. Con esto, la *Función de Pérdida* calcula lo incorrecto o no de una predicción. 
Estas *Funciones de Pérdida o Costo* pueden eleguirse de acuerdo al tipo de modelo de red neuronal a implementar para evaluar su rendimiento o performance. Entre estas, una de las más utilizadas es la MSE, Mean Square Error, también conocida como *Costo cuadrático medio*, definida como:

$$MSE = \frac{1}{N} \sum (y - (prediccion(x)))^2$$
  
Los pesos se suelen inicializar con valores escogidos de forma aleatoria, y generalmente son números pequeños. El ajuste de estos pesos ocurre gracias a un algoritmo de optimización, que ayudan a reducir o minimizar la *Función de Pérdida*. Estos algoritmos de optimización suelen estar basados en el cálculo del **gradiente** de la *función de pérdida*, debido a que éste indica la dirección de máximo crecimiento de la función en cierto punto. Este tipo de algoritmos es por tanto llamado **Descenso de Gradiente**, y son técnicas conocidas como **Gradient Descent Optimization**. Entre ellas se encuentran una gran variedad de algoritmos que implementan el Descenso de Gradiente, tales como Adagrad, Adadelta, Adam, Adamax y Nadam.
Para actualizar el peso una vez que la *Función de Pérdida* a calculado el error y el algoritmo de optimización a recalculado los pesos para *minimizar la Función de Pérdida*, se recurre a un algoritmo de propagación hacia atrás, desde la capa de salida hacia las capas anteriores. Este algoritmo de propagación se conoce como **Backpropagation** o **Propagación hacia atras**.

