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

//FUNCIONES DE ACTIVACIÓN
const model = tf.sequential();
model.add(
  tf.layers.dense({
    units: 10, //Número de nodos o neuronas
    activation: "sigmoid", //Función de Activación
    inputShape: [data.shape[1]], //Dimensión de los datos de entrada
  })
);
//Lineal
activation: "linear";

//Step, Heaviside o escalón
activation: "step";

//Sigmoide, Logística
activation: "sigmoid";

//Softmax
activation: "softmax";

//tanh
activation: "tanh";

//ReLU
activation: "relu";

//Leaky ReLU
activation: tf.layers.leakyReLU(alpha);
//alpha >=0, default alpha=0.3

//FUNCIONES DE PÉRDIDA
model.compile({
  optimizer: tf.train.adam(0.001), //Optimizador ADAM con Learning Rate de 0.001
  loss: "meanSquaredError", //Función de Pérdida
});
//Mean square error (MSE)
loss: "meanSquaredError";

//Mean Absolute Error (MAE)
loss: "meanAbsoluteError";

//Binary cross entropy
loss: "binaryCrossentropy";

//Categorical cross entropy
loss: "categoricalCrossentropy";
