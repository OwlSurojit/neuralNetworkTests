package mainClass;
class neuralNetwork {
  Matrix I, H, O, Wih, Who, Bih, Bho;
  float learningRate = 0.05;

  neuralNetwork(int inputNodes, int hiddenNodes, int outputNodes) {
    this.I = new Matrix(inputNodes, 1);
    this.H = new Matrix(hiddenNodes, 1);
    this.O = new Matrix(outputNodes, 1);
    this.Wih = new Matrix(hiddenNodes, inputNodes);
    this.Who = new Matrix(outputNodes, hiddenNodes);
    this.Bih = new Matrix(hiddenNodes, 1);
    this.Bho = new Matrix(outputNodes, 1);
  }

  void setLearningRate(float lr){
    this.learningRate = lr;
  }

  void randomiseWeights(float lowest, float highest, int digitsAfterComma) {
    this.Wih.randomise(lowest, highest, digitsAfterComma);
    this.Who.randomise(lowest, highest, digitsAfterComma);
    this.Bih.randomise(lowest, highest, digitsAfterComma);
    this.Bho.randomise(lowest, highest, digitsAfterComma);
  }

  public float[] feedForward(float[] input) {
    this.I.define(input);
    this.H = multiply(Wih, I);
    this.H.add(this.Bih);
    this.H.sigmoidEach();
    this.O = multiply(Who, H);
    this.O.add(this.Bho);
    this.O.sigmoidEach();
    return this.O.arr[0];
  }

   public float[] feedForwardDebug(float[] input, boolean onlyOutput) {
    this.I.define(input);
    this.H = multiply(Wih, I);
    this.H.add(this.Bih);
    this.H.sigmoidEach();
    this.O = multiply(Who, H);
    this.O.add(this.Bho);
    this.O.sigmoidEach();
    
    if (!onlyOutput) {
      println("Input");
      this.I.debug();
      println("Weights between Input and Hidden");
      this.Wih.debug();
      println("Biases between Input and Hidden");
      this.Bih.debug();
      println("Hidden");
      this.H.debug();
      println("Weights between Hidden and Output");
      this.Who.debug();
      println("Biases between Hidden and Output");
      this.Bho.debug();
    }
    println("Output");
    this.O.debug();
    return this.O.arr[0];
  }

  void backpropagation(float[] input, float[] expectedOutput) {

    //feed forward
    this.I.define(input);
    this.H = multiply(Wih, I);
    this.H.add(this.Bih);
    this.H.sigmoidEach();
    this.O = multiply(Who, H);
    this.O.add(this.Bho);
    this.O.sigmoidEach();

    //initialize matrices
    Matrix EO = new Matrix(this.O.rows, 1);
    //matrix outputErrors = new matrix(this.O.rows,1);
    EO.define(expectedOutput);

    //calculate output errors
    Matrix outputErrors = substract(EO, O);

    //calculate the output gradient
    Matrix outputGradient = this.O.dSigmoidEach();
    outputGradient.multiply(outputErrors);
    outputGradient.scl(this.learningRate);

    //calculate the deltas of the outputHidden weights
    Matrix WhoDeltas = multiply(outputGradient, this.H.transpose());

    //adjust the HO weights by the calculated deltas
    this.Who.add(WhoDeltas);
    //adjust the HO biases by its deltas (basically just the output gradient...)
    this.Bho.add(outputGradient);

    //calculate hidden errors
    Matrix hiddenErrors = multiply(this.Who.transpose(), outputErrors);

    //calculate the hidden gradient
    Matrix hiddenGradient = this.H.dSigmoidEach();
    hiddenGradient.multiply(hiddenErrors);
    hiddenGradient.scl(this.learningRate);

    //calculate the deltas of the inputHidden weights
    Matrix WihDeltas = multiply(hiddenGradient, this.I.transpose());

    //adjust the IH weights by the calculated deltas
    this.Wih.add(WihDeltas);
    //adjust the IH biases by its deltas (basically just the output gradient...)
    this.Bih.add(hiddenGradient);
  }
}
