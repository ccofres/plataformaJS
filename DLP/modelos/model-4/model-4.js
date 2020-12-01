async function run(){
  //Datos de entrenamiento
  const trainingUrl = 'wdbc-train.csv';
  const trainingData = tf.data.csv(trainingUrl, {
    columnConfigs: {
      diagnosis: {
        isLabel: true
      }
    }
  });
  const convertedTrainingData =
    trainingData.map(({xs,ys}) => {
      return {xs: Object.values(xs), ys: Object.values(ys)};
    }).batch(10);
  //Datos de test
  const testingUrl = 'wdbc-test.csv';
  const testingData = tf.data.csv(testingUrl, {
    columnConfigs: {
      diagnosis: {
        isLabel: true
      }
    }
  });
  const convertedTestingData =
    testingData.map(({xs, ys}) => {
      return {xs: Object.values(xs), ys: Object.values(ys)};
    }).batch(10);
//Número de características de entrenamiento
  const numOfFeatures = (await trainingData.columnNames()).length - 1;
  //Descripción del modelo
  const model = tf.sequential();
  model.add(tf.layers.dense({inputShape: [numOfFeatures] , activation: "relu" , units: 30 }));
  model.add(tf.layers.dense({activation: "relu", units: 10}));
  model.add(tf.layers.dense({activation: "relu", units: 15}));
  model.add(tf.layers.dense({activation: "relu", units: 20}));
  model.add(tf.layers.dense({activation: "relu", units: 10}));
  model.add(tf.layers.dense({activation: "sigmoid", units: 1}));
  model.compile({loss: 'binaryCrossentropy',
                optimizer: tf.train.rmsprop(0.02),
                metrics: ['accuracy']});
  //Entrenamiento del modelo
  await model.fitDataset(convertedTrainingData,
                        {epochs:30,
                        validationData: convertedTestingData,
                        callbacks:{
                          onEpochEnd: async(epoch, logs) =>{
                            console.log(`Epoch: ${epoch}, Loss: ${logs.loss}, Accuracy: ${logs.acc}`);
                          }
                        }});
  //Guardado del modelo
  await model.save('downloads://my_model');
          /*
          //Maligno
  const testVal = tf.tensor([-0.2017560352,0.3290785951,-0.1308675428,-0.2714550596,1.029197687,0.8641183587,0.7336389793,0.8566968842,1.120327751,1.553584804,-0.04197565532,-0.5158820604,0.1315408672,-0.13875636,-0.5595397256,-0.137973541,0.09807079797,0.2875119649,-0.4244614077,0.1130514903,0.03150414385,0.6762888632,0.185286211,-0.0628080803,1.10353068,0.8744426707,1.219090897,1.389329095,1.082032838,1.540296642], [1, 30]);
          // Benigno
  //const testVal = tf.tensor([-0.2555577276,1.467633187,-0.317804369,-0.3240024372,-0.6168907233,-1.016540315,-0.769012291,-0.7264947466,-0.695676578,-1.002450691,-0.6833941839,0.2588258504,-0.7424401506,-0.4762289853,-0.4349154117,-0.9708820224,-0.526937769,-0.8819559204,-0.8617142529,-0.7220657697,-0.3901797168,1.426216202,-0.4652823041,-0.4238830643,-0.157481925,-0.9517515072,-0.6443316824,-0.8336936431,-0.7313157685,-0.877325222], [1, 30]);
  const prediction = model.predict(testVal);
  console.log(prediction);
          */
}
run();