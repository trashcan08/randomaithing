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

class ResultDisplay { 
  float[] results; // has values 0 < x < 1
  float x, y, Width, Height; // x and y are centre coords
  ResultDisplay(float ix, float iy, float iWidth, float iHeight) {
    x = ix;
    y = iy;
    Width = iWidth;
    Height = iHeight;
  }
  
  void display() {
    // bottom corner
    float cornerX = x - Width/2;
    float cornerY = y + Height/2;
    float rectWidth = Width / results.length;
    push();
    stroke(255);
    strokeWeight(3);
    line(cornerX - 5, cornerY, cornerX - 5, cornerY - Height);
    pop();   
    for (int i = 0; i < results.length; i++) {
      push();
      noStroke();
      fill(255);
      rect(cornerX + rectWidth/2 + i * rectWidth, cornerY - (results[i] * Height)/2, rectWidth, results[i] * Height);
      pop();
    }
  }
}
