# Desarrollo de ejemplos y aplicaciones en TensorFlow.js

Para el desarrollo de modelos en TensorFlow.js, se debe de tener en cuenta que los conocimientos mínimos necesarios son:
- **Un conocimiento básico de HTML**, para estructurar la forma de visualizar el modelo. El documento HTML es en sí el archivo que contiene el modelo a entrenar y ejecutar.
- **Conocimientos básicos de CSS**. Como lenguaje que acompaña a HTML para estructurar el proyecto-modelo, es el encargado de los estilos (tipos de letras, tamaños de letras, colores de fondo, etc) y por lo tanto es importante para una buena visualización.
- **Conocimientos de JavaScript**. Es el lenguaje encargado de hacer de las páginas web contenidos dinámicos, y que al igual que Python, es un lenguaje interpretado y que es la base para el desarrollo de proyectos y modelos en TensorFlow.js.

Para estos conocimientos propios del desarrollo web, se sugieren los siguientes recursos:
- [Frontend Masters Bootcamp](https://frontendmasters.com/bootcamp/), un curso en inglés, **gratis**, que repasa lo necesario de HTML, CSS y JavaScript.
- [MDN - Mozilla Developer Network](https://developer.mozilla.org/es/), mantenido por la Fundación Mozilla, creadores del navegador Mozilla Firefox, contiene la información más actual de los estandares de desarrollo web, así como una gran cantidad de ejemplos.
- [W3Schools](https://www.w3schools.com/), contiene una gran cantidad de ejemplos, tutoriales, ejercicios y referencias sobre desarrollo web.



Con los conocimientos anteriores como referencias, el proceso se puede dividir en 2 subprocesos:
- Configuración de TensorFlow.js, descrito en [Configuración de TensorFlow.js para uso en navegador](), cuya descripción termina con la generación de un archivo `index.html` y un `script.js` que contiene la descripción del modelo en JavaScript.
- Generación de un servidor local, para la ejecución del proyecto-modelo.



## Generación de un servidor local
Se crea una carpeta con los archivos del proyecto-modelo y se ejecutan a través de la generación de un servidor web local.

Para la generación de este servidor local existen dos formas simples de hacerlo:

### Forma 1: Chrome Apps
La forma más sencilla y que garantiza una buena compatibilidad con TensorFlow.js, es utilizar un servidor generado a través de **Chrome Apps** en el navegador Google Chrome:
- Las Chrome Apps están disponibles en el navegador Google Chrome en [Chrome Apps](chrome://apps/).

![Chrome Apps de Google Chrome](https://lh3.googleusercontent.com/pw/ACtC-3fH8pYbuhfrNL7YFuOeD6RejUF3RELgWtkHfOIRuHUDvH7Qn3XAiKcWlAQ__G3QqT8jQtI_dwwdxlzm-U0o693v5kOdCEB-tHL-kYLRc_m4Gg6tFLOkcClhvFPEaKopJ1BsWXW76JIeYPhpSmxCpzAx=w1015-h327-no?authuser=2)



![Servidor local generado en base a la carpeta del proyecto](https://lh3.googleusercontent.com/pw/ACtC-3fK_u6BS6UCo8d6VsmQJBB4T0L2pcmG1fZWxzney8Vjtofn4Z5ImxgbDZH93XFQzh2KE1B-Ztf35BVW_eyqCLxWSRZIOHD1suKjtMp9GbhfbrL1VlQPwwCC4REpkbBOKKkFyC9V3ocu3iIu9wPex88N=w346-h177-no?authuser=2)

### Forma 2: Live Server en VScode

VSCode, o Visual Studio Code, es un editor de texto capaz de generar un servidor local mediante la instalación de una extensión.
- Instalación de VScode https://code.visualstudio.com/
- Instalación de extensión Live Server https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer
- Abrir carpeta del proyecto, y en `index.html`, iniciar Live Server mediante **Open with Live Server**