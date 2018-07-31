neuralNetwork nn;

void setup(){
  nn = new neuralNetwork(2,3,4);
  nn.randomizeWeights(0,1,2);
  for(int i=0;i<10000;i++){
    switch(int(random(0,4))){
      case 0: nn.backpropagation(new float[]{1,0},new float[]{1,1,1,1}); break;
      case 1: nn.backpropagation(new float[]{0,1},new float[]{1,1,1,1}); break;
      case 2: nn.backpropagation(new float[]{0,0},new float[]{0,0,0,0}); break;
      default: nn.backpropagation(new float[]{1,1},new float[]{0,0,0,0});
    }
  }
  /*Matrix a = new Matrix(3,1);
  a.define(new float[]{.1,.2,.5});
  Matrix b = new Matrix(1,2);
  b.define(new float[]{.3,.6});
  multiply(a,b).debug();
  nn.feedForwardDebug(new float[]{1,0});*/
}