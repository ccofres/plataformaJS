Solución y marco teórico {#cap:solucion}
========================

Marco de desarrollo e implementación
------------------------------------

Para el desarrollo del proyecto, se decidió tomar como ejemplo un
proyecto Open Source creado por Reiichiro Nakano disponibe en GitHub de
nombre [*"Arbitrary style transfer in
TensorFlow.js"*](https://github.com/reiinakano/arbitrary-image-stylization-tfjs)
[@nakanoReiinakanoArbitraryimagestylizationtfjs2020]. Cuenta con una
[implementación
demostrativa](https://reiinakano.com/arbitrary-image-stylization-tfjs/)
de esta transferencia de estilos disponible en línea:
*https://reiinakano.com/arbitrary-image-stylization-tfjs/*\
Este proyecto, implementa la transferencia de estilos (figura
[1.1](#fig:stylize){reference-type="ref" reference="fig:stylize"}) y
además implementa una combinación de estilos para generar una imagen.

![Arbitrary style transfer in TensorFlow.js: cambio de
estilo](img/mesa3/stylize){#fig:stylize}

Para realizar esta transferencia de estilos, el creador de este proyecto
se basó principalmente en las siguientes dos publicaciones:

-   "Exploring the structure of a real-time, arbitrary neural artistic
    stylization network", Ghiasi et. al.
    [@ghiasiExploringStructureRealtime2017]

-   "Distilling the Knowledge in a Neural Network", Hinton et. al.
    [@hintonDistillingKnowledgeNeural2015]

En "Exploring the structure of a real-time, arbitrary neural artistic
stylization network", Ghiasi et. al.
[@ghiasiExploringStructureRealtime2017] se expone una arquitectura para
la transferencia de estilos para una *imagen que aporta el contenido y
otra que aporta el estilo*.

-   **Imagen de contenido**, es la imagen que se quiere generar con otro
    estilo.

-   **Imagen de estilo**, es la imagen que aporta el estilo, y por lo
    tanto, es el estilo que se quiere transferir a *imagen de
    contenido*.

Para ello se definen dos redes neuronales, una *red de transferencia de
estilo* (style transfer network) y una *red de predicción de estilo*
(style prediction network), tal como se muestra en la Figura
[1.2](#fig:arquitectura){reference-type="ref"
reference="fig:arquitectura"}.

-   **Style transfer network:** la red de transferencia de estilo, toma
    un set de parámetros representativos del estilo y la imagen que
    aporta el contenido, y genera la imagen que aporta contenido
    estilizada.

-   **Style prediction network:** la red de predicción de estilo genera
    para cada *imagen de estilo* un set de parámetros representativos,
    que después serán utilizados para proveer a la *style transfer
    network* con los parámetros necesarios para generar la imagen
    estilizada.

![Arquitectura para transferencia de estilos
implementada](img/mesa3/p2){#fig:arquitectura}

Con el objetivo de portar estos modelos a TensorFlow.js de una forma más
eficiente, se usó una técnica llamada destilación (Distillation)
propuesta en "Distilling the Knowledge in a Neural Network", Hinton et.
al. [@hintonDistillingKnowledgeNeural2015]. Esta técnica consiste
básicamente en comprimir el "conocimiento" interno de una red neuronal
de mayor tamaño y embebirlo en una red de menor tamaño.\
De esta forma, se toma una red neuronal más pequeña para replicar de
forma directa las salidas de la red de mayor tamaño. En este caso, se
comprimió un modelo Inception V3 en un modelo MobileNet V2. Este proceso
de acuerdo a la figura [1.3](#fig:destilacion){reference-type="ref"
reference="fig:destilacion"}, consiste en tomar la *imagen de estilo* y
pasarla a través del modelo Inception V3 y MobileNet V2, obteniendo de
esta forma 2 *set de parámetros representativos* para cada imagen.
Después de esto, se toma como función de pérdida (loss function) el
error cuadrático medio (MSE, Mean Square Error) entre estos 2 set de
parámetros representativos, y se usa para actualizar los pesos de
MobileNet V2.\
El código para realizar esta destilación entre estos modelos está
disponible en el [repositorio de Git Hub del proyecto Magenta para
transferencia de
estilos](https://github.com/magenta/magenta/tree/master/magenta/models/arbitrary_image_stylization).\
Se debe mencionar que el modelo MobileNet V2 tiene una implementación
Open Source disponible en python y ha sido portado a TensorFlow.js:

-   Implementación en python: [Repositorio TensorFlow
    Models](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)

-   Implementación portada a TensorFlow.js: [Repositorio de ejemplos de
    TensorFlow.js](https://github.com/tensorflow/tfjs-examples/tree/master/mobilenet)

![Proceso de entrenamiento: Destilación (Distillation
[@hintonDistillingKnowledgeNeural2015])
](img/mesa3/p1){#fig:destilacion}

### Implementación

Para el desarrollo de la plataforma, se optó por reutilizar parte del
código del proyecto [*"Arbitrary style transfer in
TensorFlow.js"*](https://github.com/reiinakano/arbitrary-image-stylization-tfjs)
[@nakanoReiinakanoArbitraryimagestylizationtfjs2020], y rescatar las
ideas básicas de su implementación para posteriormente realizar un
implementación completamente nueva. Ejemplo de esto es el código en
Listado [\[codMod\]](#codMod){reference-type="ref" reference="codMod"}
que implementa una función asíncrona genérica para cargar un modelo
preentrenado para su posterior ejecución.

``` {#codMod caption="Código en JavaScript para cargar modelos" label="codMod"}
async loadMobileNetStyleModel() {
    if (!this.mobileStyleNet) {
      this.mobileStyleNet = await tf.loadGraphModel(
        'saved_model_style_js/model.json');
    }

    return this.mobileStyleNet;
  }
```

#### Marco de desarrollo y trabajo

El marco de desarrollo y trabajo se define como la forma básica y
principal de utilización y trabajo en la plataforma. Este debe tener en
cuenta la forma de trabajo con los lenguajes de programación y librerías
implicadas.\
De esta forma, el proceso descrito a continuación puede tomarse como la
forma básica de trabajo para la implementación de modelos con
TensorFlow.js .

**Descripción del proceso:**

1.  Diseño y entrenamiento del modelo en lenguaje Python, utilizando la
    API TensorFlow.

2.  Guardado del modelo en un tipo compatible con [*convertidor de
    modelos*](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter)
    [@TensorflowTfjs].

3.  Conversion del modelo a formato web compatible, utilizando
    [*convertidor de
    modelos*](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter).

4.  Implementación del modelo en TensorFlow.js .

De esta forma de acuerdo a lo anterior, resumido en la Figura
[1.4](#fig:proceso){reference-type="ref" reference="fig:proceso"}:

-   Los puntos 1,2 y 3, pueden conseguirse con la inclusión en el código
    de entrenamiento y/o diseño en Python del template en Listado
    [\[codConv\]](#codConv){reference-type="ref" reference="codConv"}.

    -   Línea 1, importa TensorFlow.js

    -   Línea 5-7, guarda el modelo entrenado en formato .h5

    -   Línea 8, llama al convertidor de modelos, declara el formato de
        entrada como *keras* (en el caso de haber hecho uso de esta
        API), le entrega el directorio de guardado como
        *saved_model_path*, y declara el directorio de guardado del
        modelo convertido como *./* (directorio actual de ejecución).

![Proceso básico de desarrollo](img/fix0/diag_flujo){#fig:proceso}

``` {#codConv caption="Código base en Python para convertir modelos" label="codConv"}
!pip install tensorflowjs

    [CODIGO PYTHON]

    import time
    saved_model_path = "./{}.h5".format(int(time.time()))
    model.save(saved_model_path)
    !tensorflowjs_converter --input_format=keras {saved_model_path} ./
```

### Diseño base de plataforma

Como una primera etapa en la implementación, se propone un diseño basado
en el mismo proyecto base, pero haciendolo extensivo para su
reutilización en la implementación de otros modelos. De esta forma, el
diseño base se muestra en Figura
[1.5](#fig:diseno1){reference-type="ref" reference="fig:diseno1"}.

![Diseño interfaz modelo tipo](img/mesa3/d1){#fig:diseno1}

De acuerdo a la Figura [1.5](#fig:diseno1){reference-type="ref"
reference="fig:diseno1"}, el diseño básico demarcado consta de 4 partes.
Un título de la página, una selección para el *modelo*, una zona
relacionada propiamente con el modelo llamada "Marco de predicción" y
una última zona relacionada con la descripción de nombre "Marco de
descripción". Se definen por lo tanto, lo siguiente:

-   **Título de la página:** zona de demarcación para título propio de
    la pagina en cuestion. Por motivos de descripción y facilidad de
    navegación, debe de disponer mínimamente de un títulos principal
    correspondiente al título general de la página y un subtítulo propio
    del modelo seleccionado relacionado con algún modelo en la zona de
    demarcación *Selección de modelo*.\

-   **Selección de modelo:** zona de demarcación para la selección del
    modelo a utilizar. Esta zona está subdividida conforme a la cantidad
    de modelos implementados.\

-   **Marco de predicción:** zona que demarca el lugar en donde el
    modelo al ejecutarse muestra elresultado de su ejecución. Esta zona
    depende estrictamente del tipo de modelo a implementar, y por lo
    tanto, se definen como base dos tipos de modelos, *Modelo tipo 1* y
    *Modelo tipo 2*.\

-   **Marco de descripción:** es la zona en la cual el modelo
    seleccionado e implementado lleva una explicación propia de su
    funcionamiento. Esta explicación debe de contar con una explicación
    del funcionamiento del modelo a alto nivel mediante diagramas de
    bloques, y una explicación a más bajo nivel a nivel de código para
    comprender mejor su implementación. Citas y referencias deben
    acompañar esta descripción del modelo.

De esta forma, como una primera instancia y simplificación de la
plataforma, se propone la implementación de acuerdo a dos tipos de
modelos bases:

-   **Modelo tipo 1:** de acuerdo a Figura
    [1.6](#fig:diseno2){reference-type="ref" reference="fig:diseno2"},
    diseño útil para la implementación de modelos en el que el resultado
    de su ejecución es principalmente una imagen, como es el caso de la
    implementación de un modelo de transferencia de estilos. De acuerdo
    a esto, es un diseño apto para modelos tales como:

    -   [Estimación en tiempo real de posición del cuerpo con
        PoseNet](https://github.com/tensorflow/tfjs-models/tree/master/posenet).

    -   [Detección y reconomiento de rostros con Face
        API](https://github.com/justadudewhohacks/face-api.js).\

-   **Modelo tipo 2:** de acuerdo a Figura
    [1.7](#fig:diseno3){reference-type="ref" reference="fig:diseno3"},
    diseño útil para la implementación de modelos en el que el resultado
    de su ejecución es principalmente texto. De acuerdo a esto, es un
    diseño apto para modelos tales como:

    -   [Clasificación de imagenes con
        MobileNet](https://github.com/tensorflow/tfjs-models/tree/master/mobilenet).

    -   [Clasificación de toxicidad de
        texto](https://github.com/tensorflow/tfjs-models/tree/master/toxicity).\

![Diseño interfaz modelo: tipo 1](img/mesa3/d2){#fig:diseno2}

![Diseño interfaz modelo: tipo 2](img/mesa3/d3){#fig:diseno3}

### Herramientas para el desarrollo

De acuerdo a la implementación base, y teniendo en consideración la
facilidad de implementación, programación y posterior claridad de uso,
las herramientas de desarrollo utilizadas se pueden resumir en las
siguientes: HTML, JavaScript, TensorFlow.js

-   HTML, para generar la estructura de la página web.

-   CSS, como herramienta para inserción de estilos y orden en la
    estructura.

-   JavaScript, como lenguaje de programación para describir los
    procedimientos y acciones, e implementación de TensorFlow.js.

Las principales herramientas de desarrollo a utilizar para esta
implementación se listan a continuación:

-   **VS Code:** editor de texto desarrollado por Microsoft, open
    source, con gran cantidad de herramientas y extensiones, con
    implementación nativa de autocompletado de código en JavaScript.

-   **Git:** herramienta para control de versiones y seguimiento en el
    desarrollo.

-   **Git Hub:** servicio en la nube que implementa Git para el control
    de versiones, permitiendo de esta manera tener un respaldo del
    código en toda su fase de desarrollo e implementación, y con
    capacidad de servir sitios web de forma estática directamente desde
    un repositorio de código.

### Implementación base de plataforma

De acuerdo a lo anterior, se implementa una primera plataforma online
con base en las siguientes características:

-   Se genera un repositorio en GitHub, ordenando archivos para mayor
    claridad.

    -   **Repositorio:**
        <https://github.com/ccofres/ccofres.github.io/tree/gh-pages>

-   Se genera sitio web de forma automática por GitHub Pages desde el
    repositorio.

-   Bajo la carpeta **Modelos** pueden ser agregadas nuevas
    implementaciones, para posteriormente ser agregados al menú editando
    directamente el documento *model.html*.

![Estructura de archivos en GitHub](img/fix0/gh1 "fig:"){#fig:gh-pages}
![Estructura de archivos en GitHub](img/fix0/gh2 "fig:"){#fig:gh-pages}

De esta forma, es posible clonar o descargar de forma local el proyecto,
y utilizar estos modelos a través de la generación de un servidor web
local a través de las **Chrome Apps** en el navegador Google Chrome:

-   Las Chrome Apps están disponibles en <chrome://apps/>

    ![Chrome Apps de Google
    Chrome](img/fix0/chrome-apps){#fig:chrome-apps}

    ![Servidor local generado en base a la carpeta del proyecto
    ](img/fix0/chrome-apps2){#fig:chrome-apps2}

Parte de los modelos en JavaScript implementados quedan disponibles en
Apéndice [\[cap:unApendice\]](#cap:unApendice){reference-type="ref"
reference="cap:unApendice"}. Esta plataforma, aún en desarrollo, toma
por nombre *Plataforma JS* y está disponible en el dominio
<https://plataformajs.studio/>.
