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

//OPTIMIZADORES
/**
 * optimizer: [optimizador(LEARNING_RATE)]
 */
//Stochastic Gradient Descent (SGD)
optimizer: "sgd";
optimizer: tf.train.sgd();

//Momentum
optimizer: "momemtum";
optimizer: tf.train.momemtum();

//RMSProp
optimizer: "rmsprop";
optimizer: tf.train.rmsprop();

//AdaDelta
optimizer: "adadelta";
optimizer: tf.train.adadelta();

//ADAM
optimizer: "adam";
optimizer: tf.train.adam();

//AdaMax
optimizer: "adamax";
optimizer: tf.train.adamax();
