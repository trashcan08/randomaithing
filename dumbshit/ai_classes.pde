
// Neuron (YAY ML YIPPEE)
class Neuron {
  int weightNum; // no. of weights

  float[] prevActivations; // activations of previous layers
  float prevSum; // sum of previous layers

  float biasLearn = 0.0; // bias learn value, mostly made for convenience, for other neurons to use

  float[] weights; // weights
  float val; // activation value
  float bias;

  FloatList[] learnVals; // values to tweak the weights by, bias is last element (average them out per learn cycle)

  float learningRate = modelLearningRate;

  Neuron(int weightAmt) {
    weightNum = weightAmt;
    weights = new float[weightAmt];
    for (int i = 0; i < weightAmt; i++) {
      weights[i] = random(-1/sqrt(weightNum) , 1/sqrt(weightNum) ); // initialise random weights
      // print(weights[i], " ");
    }
    bias = random(-sqrt(1/weightNum) , sqrt(1/weightNum));

    prevActivations = new float[weightNum];
    learnVals = new FloatList[weightNum + 1];
    for (int i = 0; i < weightNum + 1; i++) {
      learnVals[i] = new FloatList();
    }
  }

  float setActivation(float[] actArr, boolean relu) {
    if (actArr.length != weights.length) {
      println("incorrect array length");
    }  
    float sum = 0.0;
    for (int i = 0; i < weights.length; i++) {
      sum += actArr[i] * weights[i] ;
      prevActivations[i] = actArr[i];
    }
    sum += bias;
    prevSum = sum;
    
    // val = sigmoid(sum); // sigmoid
    if (relu) {
      val = relu(sum);
    } else {
      val = sum;
    }
     
    // print(val, " ");
    
    return val;
  }
  
  
  
  void addLearnVal(float inputVal, boolean relu) { 
    // inputVal is specifically dC/da
    // relu is whether the activation function (sigmoid or relu) should be applied
    
    // get learnVals for each weight
    for (int i = 0; i < weights.length; i++) {
      float learnVal = prevActivations[i] * inputVal; 
      if (relu) {
        learnVal *= reluDiff(prevSum); // relu
        // learnVal *= sigmoid(prevSum) * (1 - sigmoid(prevSum)); // sigmoid
      }
      
      learnVals[i].append(learnVal);
      // print(learnVal, " ");
    }

    // get bias learn val
    // float biasLearnVal = sigmoid(prevSum) * (1 - sigmoid(prevSum)) * inputVal;
    float biasLearnVal = inputVal; // relu
    if (relu) {
      biasLearnVal *= reluDiff(prevSum);
    }
    learnVals[weights.length].append(biasLearnVal);

    biasLearn = biasLearnVal;
  }

  void updateWeights() {
    for (int i = 0; i < weights.length; i++) {
      float average = average(learnVals[i]);
      learnVals[i].clear();
      weights[i] = weights[i] - (average * learningRate);
      
    }

    float biasAverage = average(learnVals[weights.length]);
    learnVals[weights.length].clear();
    bias -= biasAverage * learningRate;
  }
}

// ML Model
class Model {
  // 784 start neurons, 10 end neurons
  Neuron neurons[][];

  Model(int istart, int iend, int ilayers[]) {

    neurons = new Neuron[ilayers.length + 1][];

    for (int i = 0; i < ilayers.length; i++) {
      neurons[i] = new Neuron[ilayers[i]];
    }

    neurons[ilayers.length] = new Neuron[iend];

    for (int i = 0; i < ilayers.length; i++) {
      for (int j = 0; j < ilayers[i]; j++) {
        if (i == 0) {
          neurons[i][j] = new Neuron(istart);
        } else {
          neurons[i][j] = new Neuron(ilayers[i - 1]);
        }
      }
    }
    for (int i = 0; i < iend; i++) {
      neurons[ilayers.length][i] = new Neuron(ilayers[ilayers.length - 1]);
    }
  }

  float[] predictNum(float[] input) {
    for (int i = 0; i < neurons.length; i++) {
      float[] prevAct; // activations of neurons from the previous layer
      if (i == 0) {
        prevAct = new float[1];
      } else {
        prevAct = new float[neurons[i-1].length];
        for (int k = 0; k < neurons[i-1].length; k++) {
          prevAct[k] = neurons[i-1][k].val;
        }
      }
      for (int j = 0; j < neurons[i].length; j++) {
        if (i == 0) {
          neurons[i][j].setActivation(input, true);
        } else if (i == neurons.length - 1) {
          neurons[i][j].setActivation(prevAct, false);
        } else {
          neurons[i][j].setActivation(prevAct, true);
        }
      }
    }
    float[] lastLayer = new float[neurons[neurons.length - 1].length];
    float softmaxTotal = 0;
    for (int i = 0; i < neurons[neurons.length - 1].length; i++) {
      lastLayer[i] = neurons[neurons.length - 1][i].val;
      // println(neurons[neurons.length - 1][i].val);
      softmaxTotal += exp(lastLayer[i]);
    }
    // softmax
    float [] softmax = new float[neurons[neurons.length - 1].length];
    
    for (int i = 0; i < neurons[neurons.length - 1].length; i++) {
      softmax[i] = exp(lastLayer[i]) / softmaxTotal;
    }
    
    //print('\n');
    return softmax;
  }
  float getCost (float[] input, float[] ans) {
    float[] prediction = predictNum(input);
    float sum = 0.0;
    for (int i = 0; i < ans.length; i++) {
      sum += -ans[i] * log(prediction[i]);
    }
    return sum;
  }
  
  void printActivations() {
    for (int i = 0; i < neurons.length; i++) {
      for (int j = 0; j < neurons[i].length; j++) {
        print(neurons[i][j].val, ' ');
      }
    }
  }
  
  void printWeights() {
    for (int i = 0; i < neurons.length; i++) {
      for (int j = 0; j < neurons[i].length; j++) {
          print(neurons[i][j].weights[0], ' ');
      }
    }
  }
  
  void learn(float[] input, float[] ans) { // back propagation
    float[] prediction = predictNum(input);
    for (int i = neurons.length - 1; i >= 0; i--) {
      for (int j = 0; j < neurons[i].length; j++) {
        if (i == neurons.length - 1) {
          float learnVal = 0.0;
          int k = getResult(ans);
            float mult =  -1 / prediction[k];
            if (j == k) {
              learnVal = mult * prediction[k] * (1 - prediction[k]);
            } else {
              learnVal = mult * -prediction[k] * prediction[j];
            }
          neurons[i][j].addLearnVal(learnVal, false);
        } else {
          float sumLearnVals = 0.0;
          for (int k = 0; k < neurons[i+1].length; k++) {
            sumLearnVals += neurons[i+1][k].biasLearn * neurons[i+1][k].weights[j];
          }
          neurons[i][j].addLearnVal(sumLearnVals, true);
        }
      }
    }
    //for (int i = 0; i < prediction.length; i++) {
    //  print(prediction[i]);
    //  print(" ");
    //}
    //print("\n");
  }

  void updateAllWeights() {
    for (int i = neurons.length - 1; i >= 0; i--) {
      for (int j = 0; j < neurons[i].length; j++) {
        neurons[i][j].updateWeights();
      }
    }
  }
}
