import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.Objects;
import java.util.function.Consumer;
import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;

// MNist reader stuff
FileInputStream inImage = null;
FileInputStream inLabel = null;
String inputImagePath = "C:\\Users\\zenit\\Desktop\\Processing\\randomaithing\\dumbshit\\train-images-idx3-ubyte.idx3-ubyte";
String inputLabelPath = "C:\\Users\\zenit\\Desktop\\Processing\\randomaithing\\dumbshit\\train-labels.idx1-ubyte";
int magicNumberImages;
int numberOfImages;
int numberOfRows;
int numberOfColumns;

int magicNumberLabels;
int numberOfLabels;

int iterator = 0; // iterator for images
int epochIterator = 0; // iterator per epoch
int epoch = 0; // no. of epochs
int maxEpoch = 10; // max no. of epochs
float storedImgs[][][] = new float[10000][28][28];
int storedLabels[] = new int[10000];

int correct = 0, wrong = 0; // no of correct and wrong

FloatList sumCost = new FloatList();

int gridX = 28, gridY = 28; // DO NOT CHANGE, FOR DATASET


int start = 784, end = 10; // start and end no. of neurons (also DO NOT CHANGE)

int layers[] = {16, 16};

float modelLearningRate = 0.001; // learning rate of model
int penSize = 2; // radius of pen
boolean showGraph = false;

Grid grid;
Model model;
Bar bar;
Graph graph;

// util functions
float clamp(float num, float min, float max) {
  return min(max, max(min, num));
}

float average(FloatList arr) {
  if (arr.size() == 0) return 0.0;
  float sum = 0.0;
  int size = arr.size();
  // Iterate without removing elements
  for (int i = 0; i < arr.size(); i++) {
    sum += arr.get(i);
  }
  return sum / size;
}

float sigmoid(float x) {
  return 1 / (1 + exp(-x)); //sigmoid
}

float relu(float x) {
  return max(0, x); // relu
}

float reluDiff(float x) {
  if (x < 0) { // reludiff
    return 0;
  } else {
    return 1;
  }
}

float iLerp(float a, float b, float v) {
  return (v-a)/(b-a);
}

int getResult(float[] ansArr) {
  float max = 0.0;
  int maxIndex = 0;
  for (int i = 0; i < ansArr.length; i++) {
    if (ansArr[i] > max) {
      max = ansArr[i];
      maxIndex = i;
    }
  }
  return maxIndex;
}

float[] getAnsArray(int ans) {
  float[] ansArr = new float[10];
  for (int i = 0; i < 10; i++) {
    if (ans == i) {
      ansArr[i] = 1.0;
    } else {
      ansArr[i] = 0.0;
    }
  }
  return ansArr;
}

float[] softmax(float[] arr) {
  float sumExp = 0;
  float[] output = new float[arr.length];
  for (int i = 0; i < arr.length; i++) {
    output[i] = exp(arr[i]);
    sumExp += output[i];
  }
  for (int i = 0; i < arr.length; i++) {
    output[i] /= sumExp;
  }
  return output;
}

void drawArrow(PVector v0, PVector v1, float thickness, float arrowLength, float arrowWidth, color arrowColour) {

  stroke(arrowColour);
  strokeWeight(thickness);
  line(v0.x, v0.y, v1.x, v1.y);

  PVector v0tov1 = v1.copy().sub(v0);

  noStroke();
  fill(arrowColour);
  PVector firstPoint = v0.copy().add(v0tov1.copy().mult(1 + arrowLength)); // 0.05 arrowLength
  PVector otherPoint = v0.copy().add(v0tov1.copy().mult(1 - arrowLength));
  PVector secondPoint = otherPoint.copy().add(v0tov1.copy().rotate(HALF_PI).setMag(arrowWidth));
  PVector thirdPoint = otherPoint.copy().add(v0tov1.copy().rotate(-HALF_PI).setMag(arrowWidth));

  triangle(firstPoint.x, firstPoint.y, secondPoint.x, secondPoint.y, thirdPoint.x, thirdPoint.y);
}

void setup() {
  
  rectMode(CENTER);
  textAlign(CENTER, CENTER);
  size(800, 600);
  grid = new Grid(gridX, gridY, width/3, height/2, 400, 400);
  model = new Model(start, end, layers);
  bar = new Bar(width/2, 9*height/10, 4*width/5, height/20);
  graph = new Graph(width/2, height/2, 500, 500);

  //for (int i = 0; i < 10; i++) {
  //  graph.graphData.push(random(10));
  //}

  // MNIST reader stuff
  // TODO Auto-generated method stub

  try {
    inImage = new FileInputStream(inputImagePath);
    inLabel = new FileInputStream(inputLabelPath);

    magicNumberImages = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
    numberOfImages = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
    numberOfImages = 10000;
    numberOfRows  = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
    numberOfColumns = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());

    magicNumberLabels = (inLabel.read() << 24) | (inLabel.read() << 16) | (inLabel.read() << 8) | (inLabel.read());
    numberOfLabels = (inLabel.read() << 24) | (inLabel.read() << 16) | (inLabel.read() << 8) | (inLabel.read());
  }

  catch (IOException e) {
    e.printStackTrace();
  }

  for (int i = 0; i < numberOfImages; i++) {
    try {
      float[][] imgPixels = new float[numberOfRows][numberOfColumns];
      for (int j = 0; j < numberOfRows; j++) {
        for (int k = 0; k < numberOfColumns; k++) {
          float gray = inImage.read();
          gray /= 255;
          imgPixels[k][j] = gray;
        }
      }

      storedImgs[i] = imgPixels;

      int label = inLabel.read();

      storedLabels[i] = label;
    }
    catch(IOException e) {
      e.printStackTrace();
    }
  }
  // frameRate(1); // slooooww mooooddeeee
}
void draw() {
  background(0);
  //// grid.allowDraw();
  //if (showGraph) {
  //  graph.display();
  //} else {
  //  grid.display();
  //  bar.display();
  //}
  //float[] gridVals = new float[784];
  //int num = 0;
  //for (int i = 0; i < gridX; i++) {
  //  for (int j = 0; j < gridY; j++) {
  //    gridVals[num] = grid.vals[i][j];
  //    num++;
  //  }
  //}

  //float[] predictedNum = model.predictNum(gridVals);
  //for (int i = 0; i < predictedNum.length; i++) {
  //  print(predictedNum[i], ' ');
  //}
  //print('\n');

  if (epoch <= maxEpoch) {


    float[][] imgPixels = storedImgs[epochIterator];
    int label = storedLabels[epochIterator];


    float[] ansArr = getAnsArray(label);

    float[] gridVals = new float[784];
    int num = 0;
    for (int i = 0; i < gridX; i++) {
      for (int j = 0; j < gridY; j++) {
        gridVals[num] = imgPixels[i][j];
        num++;
      }
    }

    // predict val
    float[] ans = model.predictNum(gridVals);
    //for (int i = 0; i < ans.length; i++) {
    //  print(ans[i], ' ');
    //}
    //print('\n');
    
    int result = getResult(model.predictNum(gridVals));
    float cost = model.getCost(gridVals, ansArr);
    
    // print(cost, '\n');
    sumCost.append(cost);
    
    //for (int i = 0; i < gridVals.length; i++) {
    //  print(gridVals[i], ' ');
    //}
    
    // graph.graphData.append(cost);
    println("Before backprop:");
    println("Activations:");
    model.printActivations();
    //print('\n');
    //println("Weights:");
    //model.printWeights();
    model.learn(gridVals, ansArr);
    //print('\n');
    //println("After backprop:");
    //println("Activations:");
    //model.printActivations();
    //print('\n');
    //println("Weights:");
    //model.printWeights();

    if (label == result) {
        correct++;
      } else {
        wrong++;
      }

    if (showGraph) {
      
      graph.display();
    } else {
      // display image
      grid.vals = imgPixels;
      grid.display();

      // display label
      textSize(40);
      text(label, 4 * width/5, height/2);

      // display bar
      bar.progress = (float(epochIterator) / numberOfImages);
      //print(float(iterator) / numberOfImages);
      //print('\n');
      bar.display();

      // predict and display ans

      push();
      if (label == result) {
        fill(0, 255, 0);
      } else {
        fill(255, 0, 0);
      }
      textSize(30);
      text(result, 4 * width/5, height/2 - 50);
      pop();

      // show wrong and correct
      push();
      fill(255, 0, 0);
      textSize(20);
      text("Wrong:", 7 * width/10, height/5);
      textSize(40);
      text(wrong, 7 * width/10, 3 * height/10);
      pop();

      push();
      fill(0, 255, 0);
      textSize(20);
      text("Correct:", 9 * width/10, height/5);
      textSize(40);
      text(correct, 9 * width/10, 3 * height/10);
      pop();
    }
    epochIterator++;
    if (epochIterator != 0 && epochIterator % ((numberOfImages - 1) / 100) == 0) {
      model.updateAllWeights();
      print("updated weights", '\n');
    }
    if (epochIterator != 0 && epochIterator % ((numberOfImages - 1)) == 0) {
      epochIterator = 0;
      epoch++;
      float averageCost = average(sumCost);
      graph.graphData.append(averageCost);
      sumCost.clear();
    }
  } else {
    if (epoch == maxEpoch) {
      grid.clearGrid();
      epoch++;
    }
    if (showGraph) {
      graph.display();
    } else {

      grid.allowDraw();
      grid.display();

      // show wrong and correct
      push();
      fill(255, 0, 0);
      textSize(20);
      text("Wrong:", 7 * width/10, height/5);
      textSize(40);
      text(wrong, 7 * width/10, 3 * height/10);
      pop();

      push();
      fill(0, 255, 0);
      textSize(20);
      text("Correct:", 9 * width/10, height/5);
      textSize(40);
      text(correct, 9 * width/10, 3 * height/10);
      pop();

      float[] gridVals = new float[784];
      int num = 0;
      for (int i = 0; i < gridX; i++) {
        for (int j = 0; j < gridY; j++) {
          gridVals[num] = grid.vals[i][j];
          num++;
        }
      }

      int result = getResult(model.predictNum(gridVals));
      push();
      fill(255);
      textSize(40);
      text(result, 4 * width/5, height/2);
      pop();
    }
  }
}

void keyPressed() { // display graph
  if (key == ENTER) {
    graph.graphData.remove(0);
  } else if (graph.graphData.size() != 0) {
    showGraph = !showGraph;
  }
}

class Grid { // grid for drawing/displaying
  int rows;
  int columns;
  float Xpos; // centre position of grid
  float Ypos;
  float Width;
  float Height;
  float[][] vals;

  boolean showOutlines = false;
  boolean showVals = false;

  Grid(int iRows, int iColumn, float iXpos, float iYpos, float iWidth, float iHeight) {
    rows = iRows;
    columns = iColumn;
    Xpos = iXpos;
    Ypos = iYpos;
    Width = iWidth;
    Height = iHeight;

    vals = new float[rows][columns];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        vals[i][j] = 0.0;
      }
    }
  }



  void display() {
    push();
    float rectWidth = Width / rows;
    float rectHeight = Height / columns;
    float cornerX = Xpos - Width/2 + rectWidth / 2;
    float cornerY = Ypos - Height/2 + rectHeight / 2;

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        if (showOutlines) {
          strokeWeight(1);
          stroke(255);
        } else {
          strokeWeight(0);
          noStroke();
        }
        fill(vals[i][j] * 255);
        rect(cornerX + (i * rectWidth), cornerY + j * rectHeight, rectWidth + 1, rectHeight+1);
        if (showVals) {
          fill(255, 0, 0);
          textSize( rectHeight / 2);
          text(str(vals[i][j]).substring(0, 3), cornerX + (i * rectWidth), cornerY + j * rectHeight);
        }
      }
    }
    pop();
  }



  void allowDraw() { // lmao draw slowly im lazy to code properly
    if (mouseX < Width/2 + Xpos && mouseX > Xpos - Width/2 && mouseY < Height/2 + Ypos && mouseY > Ypos - Height/2) {
      float cornerX = Xpos - Width/2;
      float cornerY = Ypos - Height/2;
      int XMouseGrid = int((mouseX - cornerX) * rows / Width);
      int YMouseGrid = int((mouseY - cornerY) * columns / Height);

      if (mousePressed && mouseX != pmouseX && mouseY != pmouseY) {

        for (int i = XMouseGrid - penSize; i <= XMouseGrid + penSize; i++) {
          for (int j = YMouseGrid - penSize; j <= YMouseGrid + penSize; j++) {
            if (i >= 0 && i < rows && j >= 0 && j < columns) {
              float distance = sqrt(sq(abs(XMouseGrid - i)) + sq(abs(YMouseGrid - j)));
              if (distance < penSize) {
                if (mouseButton == LEFT) {
                  vals[i][j] += 0.1 * penSize / distance;
                  vals[i][j] = clamp(vals[i][j], 0.0, 1.0);
                } else {
                  vals[i][j] -= 0.1 * penSize / distance;
                  vals[i][j] = clamp(vals[i][j], 0.0, 1.0);
                }
              }
            }
          }
        }
      }
    }
  }
  void clearGrid() {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        vals[i][j] = 0.0;
      }
    }
  }
}

class Bar {
  float Xpos, Ypos;
  float Width, Height;
  float progress = 0.0; // number from 0 to 1
  Bar(float iXpos, float iYpos, float iWidth, float iHeight) {
    Xpos = iXpos;
    Ypos = iYpos;
    Width = iWidth;
    Height = iHeight;
  }

  void display() {
    push();
    stroke(255);
    strokeWeight(3);
    fill(0);
    rect(Xpos, Ypos, Width, Height); // border
    pop();

    push();
    noStroke();
    fill(255);
    rect(Xpos - Width/2 + Width*progress/2, Ypos, Width*progress, Height);
    pop();
  }
}

class Graph {
  FloatList graphData;
  float Xpos, Ypos, Width, Height;

  Graph(float iXpos, float iYpos, float iWidth, float iHeight) {
    Xpos = iXpos;
    Ypos = iYpos;
    Width = iWidth;
    Height = iHeight;
    graphData = new FloatList();
  }

  void display() {
    // arrows
    PVector v1, v2, v3, v4;
    v1 = new PVector(Xpos - Width/2 - 10, Ypos + Height/2);
    v2 = new PVector(Xpos + Width/2, Ypos + Height/2);
    v3 = new PVector(Xpos - Width/2, Ypos - Height/2);
    v4 = new PVector(Xpos - Width/2, Ypos + Height/2 + 10);
    drawArrow(v1, v2, 5, 0.02, 10, color(255));
    drawArrow(v4, v3, 5, 0.02, 10, color(255));

    // data
    float cornerX = Xpos - Width/2 + 20;
    float cornerY = Ypos - Height/2 + 20;
    float max = graphData.max();
    float min = graphData.min();
    FloatList graphDataCopy = graphData.copy();
    float prevX = 0, prevY = 0;
    for (int i = 0; i < graphDataCopy.size(); i++) {
      float Xval = cornerX + i * (Width - 40) / graphDataCopy.size();
      float Yval = cornerY + (Height - 40) * (1 - iLerp(min, max, graphDataCopy.get(i)));
      if (i > 0) {
        push();
        strokeWeight(3);
        stroke(255);
        line(prevX, prevY, Xval, Yval);
        pop();
      }
      prevX = Xval;
      prevY = Yval;
    }

    push();
    fill(255);
    textSize(30);
    String strMax = str(max);
    while (strMax.length() < 6) {
      strMax = strMax.concat("0");
    }

    String strMin = str(min);
    while (strMin.length() < 6) {
      strMin = strMin.concat("0");
    }

    text(strMax.substring(0, 5), cornerX - width/10, cornerY);
    text(strMin.substring(0, 5), cornerX - width/10, cornerY + Width - 40);
    pop();
  }
}

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
    bias = 0;

    prevActivations = new float[weightNum];
    learnVals = new FloatList[weightNum + 1];
    for (int i = 0; i < weightNum + 1; i++) {
      learnVals[i] = new FloatList();
    }
  }

  float setActivation(float[] actArr) {
    if (actArr.length != weights.length) {
      println("incorrect array length");
    }
    float sum = 0;
    for (int i = 0; i < weights.length; i++) {
      sum += actArr[i] * weights[i] ;
      prevActivations[i] = actArr[i];
    }
    sum += bias;
    prevSum = sum;

    // val = sigmoid(sum); // sigmoid
    val = relu(sum); // relu
     
    // print(val, " ");
    
    return val;
  }
  
  float setActivationNoActivation(float[] actArr) {
    if (actArr.length != weights.length) {
      println("incorrect array length");
    }
    float sum = 0;
    for (int i = 0; i < weights.length; i++) {
      sum += actArr[i] * weights[i] ;
      prevActivations[i] = actArr[i];
    }
    sum += bias;
    prevSum = sum;

    val = sum;

    return val;
  }
  
  
  
  void addLearnVal(float inputVal) { // inputVal is specifically dC/da
    
    // get learnVals for each weight
    for (int i = 0; i < weights.length; i++) {
      // float learnVal = prevActivations[i] * sigmoid(prevSum) * (1 - sigmoid(prevSum)) * inputVal; // for sigmoid
      float learnVal = prevActivations[i] * reluDiff(prevSum) * inputVal; // relu
      learnVals[i].append(learnVal);
      // print(learnVal, " ");
    }

    // get bias learn val
    // float biasLearnVal = sigmoid(prevSum) * (1 - sigmoid(prevSum)) * inputVal;
    float biasLearnVal = reluDiff(prevSum) * inputVal; // relu
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
          neurons[i][j].setActivation(input);
        } else if (i == neurons.length - 1) {
          neurons[i][j].setActivationNoActivation(prevAct);
        } else {
          neurons[i][j].setActivation(prevAct);
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
          
          float sum = 0;
          for (int k = 0; k < neurons[i].length; k++) {
            float difference = -ans[k]/prediction[k];
            if (j == k) {
              sum += difference * prediction[k] * (1 - prediction[k]);
            } else {
              sum += difference * -prediction[k] * prediction[j];
            }
          }
          neurons[i][j].addLearnVal(sum);
        } else {
          float sumLearnVals = 0.0;
          for (int k = 0; k < neurons[i+1].length; k++) {
            sumLearnVals += neurons[i+1][k].biasLearn * neurons[i+1][k].weights[j];
          }
          neurons[i][j].addLearnVal(sumLearnVals);
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
