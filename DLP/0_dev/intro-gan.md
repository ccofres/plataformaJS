# Una introducción a las Redes Generativas Adversarias

  

## GANs: Generative Adversarial Networks

Las Redes Generativas Adversariales, o GANs, fueron descritas por primera vez el 2014 en el artículo de Ian Goodfellow, "Generative Adversarial Networks". Son una clase dentro de las técnicas del Machine Learning que permiten la generación de imagénes sintéticas, forzando las imágenes sintéticas generadas a ser estadísticamente indistinguibles de las imágenes originales. La gran capacidad de generación y la potencia de la idea de las redes adversariales generativas ha hecho que en los últimos años se haya puesto el foco en su investigación y en la generación de nuevas arquitecturas.

Las GANs consisten básicamente en dos redes que compiten mutuamente: una **red genera datos falsos**, y otra que **intenta distinguir los datos falsos de los reales**.
De esta forma, las GANs:
- Es un modelo _generativo_ porque tiene como proposito el generar nuevos datos.
- Son _redes_ porque fundamentalmente la arquitectura está compuesta de dos redes neuronales; **Discriminador** y **Generador**.
- Son _adversarias o antagónicas_ debido a que el Discriminador compite con el Generador.

  

## Funcionamiento de las GAN

Las GANs constan en su forma más básica de dos redes neuronales, _Generador_ y _Discriminador_. De acuerdo a lo anterior, los aspectos básicos y fundamentales de estas redes son:
### Generador
- Tiene como **entrada** un vector de números aleatorios, seleccionado de un espacio latente predefinido, como una función normal multivariada.
- La **salida** es un ejemplo sintetizado falso que intenta ser estadísticamente lo más parecido a un ejemplo real.
- El **objetivo** es generar datos falsos que sean indistinguibles de los datos reales.

### Discriminador
- Tiene como **entradas**
	- Los datos reales, que provienen de la base de datos de entrenamiento.	
	- Los datos falsos, sintetizados por el Discriminador.
- La **salida** es la probabilidad del ejemplo de entrada de ser real.
- El **objetivo** es distinguir los datos falsos provenientes del Generador y los datos reales provenientes de la base de datos.

![Elementos y funcionamiento de una GAN](https://lh3.googleusercontent.com/pw/ACtC-3fu9lCU-fdwbIp1vTI8Aj1EJAOamTyR-j5yy2lD_tXIt4qEMDtPQgC4Ddej_qvhRQCEb8j7suW6eqtS38kpxyzR77vpSNaUb6SSMXQHeQP14UpILcvbbtRgorO0zwqJ8njiZ2VjaVFlwg1fY2_ssWk=w679-h513-no?authuser=2)

De esta forma, se definen:

 1. **Conjunto de entrenamiento:** Base de datos de ejemplos reales. El Generador debe aprender a emular de forma perfecta estos datos. Estos datos sirven como entrada a la red Discriminador.
2.  **Vector de ruido aleatorio:** Vector **z** de entrada a la red Generador. Esta entrada es utilizada por el Generador como punto de partida para la síntesis de datos falsos.
3. **Red Generadora:** Toma como entrada un vector de números aleatorio **z**, y genera como salida un dato falso **x\***. El objetivo es que el dato falso sea indistinguible del dato real.
4.  **Red Discriminadora:** Toma como entrada un dato real **x** o un dato falso **x\***. El objetivo es determinar, para cada dato, la probabilidad si es real.
5. **Proceso iterativo de entrenamiento/sintonización:** Para cada una de las predicciones del Discriminador, se determina lo buena o no de esta, y se utiliza el resultado para volver a sintonizar la red Discriminadora y Generadora mediante _Propagación hacia atrás_ (Backpropagation).


  
##  Proceso básico de entrenamiento

El algorítmo de entrenamiento para una GAN para cada uno de los ciclos de iteración es como sigue:
1. Entrenamiento del _Discriminador_:
	- **a)** Se toma una muestra aleatoria **x** desde el conjunto de entrenamiento.
	- **b)** Se obtiene un nuevo vector aleatorio **z**, y usando la red del Generador se sintetiza un ejemplo falso **x\***.
	- **c)** Se usa la red del Discriminador para clasificar **x** y **x\***.
	- **d)** Se calculan los errores de clasificación y se propaga hacia atras (backpropagation) el error total para actualizar los parámetros de entrenamiento del Discriminador, intentando minimizar el error de clasificación.

2. Entrenamiento del _Generador_:
	- **a)** Se toma un nuevo vector aleatorio **z**, y se usa la red del Generador para sintetizar un ejemplo falso **x\***.
	- **b)** Se usa el Discriminador para clasificar **x**.
	- **c)** Se calculan los errores de clasificación y se propaga hacia atrás (backpropagation) el error para actualizar los parámetros de entrenamiento del Generador, intentando maximizar el error del Discriminador.  

![Proceso iterativo de entrenamiento de una GAN](https://lh3.googleusercontent.com/pw/ACtC-3d6Flsy6LjTv88bN3V7lOHl2_qTNkvUY24Xz8x2-NSPwBLdVYtIFagVt5N_v4rlsG8wxbYwQ048N8SVh30As5q3S7Jcxa9E5UIbCaAb0MczY6Mf6gHUuLsfj0PywA_SGUVSzpGURfQ_3OJkXWaz1DU=w361-h641-no?authuser=2)


Debido a que este proceso de entrenamiento es iterativo, cada vez que el _Discriminador_ es entrenado y mejora respecto al _Generador_, el _Generador_ es actualizado y mejora en el proceso.
Lo anterior, en palabras más simples, se debe a que el _Generador_ y el _Discriminador_ están inmersos en un _juego de suma cero_, debido a que cada red tiene como objetivo mejorar respecto a la otra, haciendo que la otra red empeore. Esto lleva a que la arquitectura deba tender a un punto de _equilibrio_, en el cual ninguna de las dos pueda seguir mejorando.

De acuerdo Ian Goodfellow, teóricamente para cada red _Generador_ existe una única red _Discriminador_ óptima. Se muestra también que el _Generador_ es óptimo cuando el _Discriminador_ alcanza predicciones de un valor de 0.5 para todas las entradas. Es decir, el _Generador_ es óptimo cuando el _Discriminador_ está completamente confundido y es incapaz de distinguir entre datos reales y datos falsos.

Sin embargo, alcanzar el _equilibrio_ para una GAN, significa en la práctica alcanzar el **Equilibrio de Nash** para un caso en el que no existen algorítmos, en donde las *funciones de costo* son no convexas y el espacio de parámetros es de altas dimensiones. Debido a lo anterior, la utilización de **Gradiente Descendiente** no garantiza su convergencia, y se han desarrollado diversas arquitecturas y "trucos" de entrenamiento heurístico para alcanzar la convergencia.

Para una explicación más detallada del funcionamiento de las GANs, se recomienda el artículo de Ian Goodfellow, "Generative Adversarial Network", el tutorial realizado en la conferencia NIPS de 2016 por él mismo y el Workshop realizado en la misma conferencia disponible en video.
