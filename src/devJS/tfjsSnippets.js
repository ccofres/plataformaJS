/**
 * Snippets para TensorFlow.js
 */

//Obtiene info del Backend actual o inicia el "mejor" Backend
tf.backend();

//Obtener el Backend en uso, ej:'cpu', 'webgl', 'wasm'
tf.getBackend();

//Setea el Backend a usar: 'cpu', 'webgl', 'wasm'
tf.setBackend("cpu");
tf.setBackend("webgl");
