# Configuración de TensorFlow.js para uso en el navegador


#### A continuación se presenta una breve exposición sobre la forma de uso de TensorFlow.js, las configuraciones mínimas para el uso en navegador , y con esto, la forma en que fue diseñado este proyecto en su entorno.


Para el uso de TensorFlow.js en el navegador existen 2 formas:
- Uso de NPM (gestor de paquetes para JavaScript) y herramientas de ayuda a la compilación de código como Parcel o Webpack.
- Inserción de un script tag en una simple página HTML que descarga el contenido desde servidores CDN ([Content Delivery Network](https://es.wikipedia.org/wiki/Red_de_distribuci%C3%B3n_de_contenidos)).

Debido a la facilidad de uso, y la claridad que presenta la inserción de un simple script tag, la segunda opción es la recomendada tanto para la **exposición de código** como el **prototipado**. Esto se detalla a continuación:
1. Creación de documento HTML5 index.html, e inserción de script tag. Se debe mencionar que es posible elegir la version de TensorFlow.js modificando levemente el enlace:
	- *https://cdn.jsdelivr.net/npm/\@tensorflow/**tfjs\@2.0.0**/dist/tf.min.js* = **TFjs 2.0.0**
	- *https://cdn.jsdelivr.net/npm/\@tensorflow/**tfjs\@2.6.0**/dist/tf.min.js*= **TFjs 2.6.0**
2. Con index.html y el script insertado, el archivo está preparado para TensorFlow.js.
3. El código puede ir en el *body tag* del HTML encerrado mediante *script tag*, o puede ir en un archivo aparte *script.js* y ser insertado en el HTML mediante una etiqueta del tipo *\<script src="script.js"\>\</script\>*

Lo anterior se resumen en la siguiente figura:
![Configuración y forma de uso de TensorFlow.js](https://lh3.googleusercontent.com/pw/ACtC-3dtbi8Y3dC_BPsUSx1stmsxt-ZBIV1_4SmzrrQobOZ7WNKM5zQVJsPlZqER9my2Q41AYQjyHE1IVL8oRw-bcv7VSdYlhcZ6zcUfkEiGu8DQEIqVII-NueB2VxAeusyVkGBvkBOEkPSrNNkFHYksSMcE=w711-h641-no?authuser=2)




## Desarrollo y creación de ejemplos/aplicaciones de forma local

T
De esta forma, es posible clonar o descargar de forma local el proyecto,
y utilizar estos modelos a través de la generación de un servidor web
local.

La forma más sencilla y que garantiza una buena compatibilidad con TensorFlow.js, es utilizar un servidor generado a través de **Chrome Apps** en el navegador Google Chrome:
- Las Chrome Apps están disponibles en el navegador Google Chrome en [Chrome Apps](chrome://apps/).

![Chrome Apps de Google Chrome](https://lh3.googleusercontent.com/pw/ACtC-3fH8pYbuhfrNL7YFuOeD6RejUF3RELgWtkHfOIRuHUDvH7Qn3XAiKcWlAQ__G3QqT8jQtI_dwwdxlzm-U0o693v5kOdCEB-tHL-kYLRc_m4Gg6tFLOkcClhvFPEaKopJ1BsWXW76JIeYPhpSmxCpzAx=w1015-h327-no?authuser=2)



![Servidor local generado en base a la carpeta del proyecto](https://lh3.googleusercontent.com/pw/ACtC-3fK_u6BS6UCo8d6VsmQJBB4T0L2pcmG1fZWxzney8Vjtofn4Z5ImxgbDZH93XFQzh2KE1B-Ztf35BVW_eyqCLxWSRZIOHD1suKjtMp9GbhfbrL1VlQPwwCC4REpkbBOKKkFyC9V3ocu3iIu9wPex88N=w346-h177-no?authuser=2)
