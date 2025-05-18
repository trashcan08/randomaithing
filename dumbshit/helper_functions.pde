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
