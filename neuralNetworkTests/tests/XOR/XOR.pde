neuralNetwork nn;
final int precision = 10;

void setup() {  
  size(400, 400);

  nn = new neuralNetwork(2, 8, 1);
  nn.setLearningRate(0.1);
  nn.randomiseWeights(0, 1, 2);
  /*
  nn.feedForwardDebug(new float[]{0, 0}, true);
   nn.feedForwardDebug(new float[]{1, 0}, true);
   nn.feedForwardDebug(new float[]{1, 1}, true);
   nn.feedForwardDebug(new float[]{0, 1}, true);
   */
}

final float rectSize = (width / precision);

void draw() {

  for (int i=0; i<1000; i++) {
    switch(int(random(0, 4))) {
    case 0: 
      nn.backpropagation(new float[]{1, 0}, new float[]{1}); 
      break;
    case 1: 
      nn.backpropagation(new float[]{0, 1}, new float[]{1}); 
      break;
    case 2: 
      nn.backpropagation(new float[]{0, 0}, new float[]{0}); 
      break;
    default: 
      nn.backpropagation(new float[]{1, 1}, new float[]{0});
    }
  }

  noStroke();
  background(255);
  for (int i = 0; i < width; i += rectSize) {
    for (int j = 0; j < height; j += rectSize) {
      float res = nn.feedForward(new float[]{i/(width-rectSize), j/(height-rectSize)})[0];
      fill(res * 255);
      rect(i, j, rectSize, rectSize);
    }
  }
}

void mousePressed() {
  setup();
}
