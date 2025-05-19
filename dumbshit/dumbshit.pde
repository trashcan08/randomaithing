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
int numberOfImgs = 10000;
float storedImgs[][][] = new float[numberOfImgs][28][28];
int storedLabels[] = new int[numberOfImgs];

int correct = 0, wrong = 0; // no of correct and wrong

FloatList sumCost = new FloatList();

int gridX = 28, gridY = 28; // DO NOT CHANGE, FOR DATASET


int start = 784, end = 10; // start and end no. of neurons (also DO NOT CHANGE)

int layers[] = {32, 32};

float modelLearningRate = 0.01; // learning rate of model
int penSize = 2; // radius of pen
boolean showGraph = false;

Grid grid;
Model model;
Bar bar;
Graph graph;
ResultDisplay resultDisplay;

void setup() {

  rectMode(CENTER);
  textAlign(CENTER, CENTER);
  size(800, 600);
  grid = new Grid(gridX, gridY, width/3, height/2, 400, 400);
  model = new Model(start, end, layers);
  bar = new Bar(width/2, 9*height/10, 4*width/5, height/20);
  graph = new Graph(width/2, height/2, 500, 500);
  resultDisplay = new ResultDisplay(4 * width/5, 2.7 * height/4, width / 5, height / 5);

  //for (int i = 0; i < model.neurons.length; i++) {
  //  print(model.neurons[i].length , ' ');

  //}
  //print('\n');

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
    numberOfImages = numberOfImgs;
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
    model.learn(gridVals,ansArr);
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
    //println("Before backprop:");
    //println("Activations:");
    //model.printActivations();
    //print('\n');
    //println("Weights:");
    //model.printWeights();
    
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

      // display as bar chart
      resultDisplay.results = ans;
      resultDisplay.display();

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

      // display percent correct
      float percentRight = (float(correct) / float(correct + wrong));
      push();
      fill(255 * (1 - percentRight), 255 * percentRight, 0);
      textSize(50);
      if (percentRight * 100 < 10) {
        text(str(percentRight * 100).substring(0, 3) + '%', 8 * width/10, 1 * height/10);
      } else {
        text(str(percentRight * 100).substring(0, 4) + '%', 8 * width/10, 1 * height/10);
      }
      pop();
    }
    epochIterator++;
    if (epochIterator != 0 && epochIterator % ((numberOfImages - 1) / 1000) == 0) {
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

      // display percent correct
      float percentRight = (float(correct) / float(correct + wrong));
      push();
      fill(255 * (1 - percentRight), 255 * percentRight, 0);
      textSize(50);
      if (percentRight * 100 < 10) {
        text(str(percentRight * 100).substring(0, 3) + '%', 8 * width/10, 1 * height/10);
      } else {
        text(str(percentRight * 100).substring(0, 4) + '%', 8 * width/10, 1 * height/10);
      }
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

      // display as bar chart
      resultDisplay.results = model.predictNum(gridVals);
      resultDisplay.display();
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
