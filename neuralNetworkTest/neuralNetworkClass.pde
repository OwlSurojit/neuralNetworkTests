class neuralNetwork {
  Matrix I, H, O, Wih, Who, Bih, Bho;
  float learningRate;

  neuralNetwork(int inputNodes, int hiddenNodes, int outputNodes) {
    this.I = new Matrix(inputNodes, 1);
    this.H = new Matrix(hiddenNodes, 1);
    this.O = new Matrix(outputNodes, 1);
    this.Wih = new Matrix(hiddenNodes, inputNodes);
    this.Who = new Matrix(outputNodes, hiddenNodes);
    this.Bih = new Matrix(hiddenNodes, 1);
    this.Bho = new Matrix(outputNodes, 1);
    this.learningRate = 0.05;
  }

  void randomizeWeights(float lowest, float highest, int digitsAfterComma) {
    this.Wih.randomize(lowest, highest, digitsAfterComma);
    this.Who.randomize(lowest, highest, digitsAfterComma);
    this.Bih.randomize(lowest, highest, digitsAfterComma);
    this.Bho.randomize(lowest, highest, digitsAfterComma);
  }

  void feedForward(float[] input) {
    this.I.define(input);
    this.H = multiply(Wih, I);
    this.H.add(this.Bih);
    this.H.sigmoidEach();
    this.O = multiply(Who, H);
    this.O.add(this.Bho);
    this.O.sigmoidEach();
    //println("Output");
    //this.O.debug();
  }

  void feedForwardDebug(float[] input) {
    this.I.define(input);
    println("Input");
    this.I.debug();
    println("Weights between Input and Hidden");
    this.Wih.debug();
    println("Biases between Input and Hidden");
    this.Bih.debug();
    this.H = multiply(Wih, I);
    this.H.add(this.Bih);
    this.H.sigmoidEach();
    println("Hidden");
    this.H.debug();
    println("Weights between Hidden and Output");
    this.Who.debug();
    println("Biases between Hidden and Output");
    this.Bho.debug();
    this.O = multiply(Who, H);
    this.O.add(this.Bho);
    this.O.sigmoidEach();
    println("Output");
    this.O.debug();
  }

  void backpropagation(float[] input, float[] expectedOutput) {

    //feed forward
    this.I.define(input);
    println("1. multiply");
    this.H = multiply(Wih, I);
    this.H.add(this.Bih);
    this.H.sigmoidEach();
    println("2. multiply");
    this.O = multiply(Who, H);
    this.O.add(this.Bho);
    this.O.sigmoidEach();

    //initialize matricies
    Matrix EO = new Matrix(this.O.rows, 1);
    //matrix outputErrors = new matrix(this.O.rows,1);
    EO.define(expectedOutput);

    //calculate output errors
    Matrix outputErrors = substract(EO, O);

    //calculate the output gradient
    println("3. multiply");
    Matrix outputGradient = multiply(this.O.dSigmoidEach(), outputErrors.transpose());
    outputGradient.scl(this.learningRate);

    //calculate the deltas of the outputHidden weights
    //HERE WOULD BE AN ERROR IF O WEREN'T A 1x1 MATRIX
    println("4. multiply");
    Matrix WhoDeltas = multiply(outputGradient, this.H.transpose());

    //adjust the HO weights by the calculated deltas
    this.Who.add(WhoDeltas);
    //adjust the HO biases by its deltas (basically just the output gradient...)
    this.Bho.add(outputGradient);

    //calculate hidden errors
    println("5. multiply");
    Matrix hiddenErrors = multiply(this.Who.transpose(), outputErrors);
    println(this.H.dSigmoidEach().cols);
    println(hiddenErrors.transpose().rows);
    println("now");

    //calculate the hidden gradient
    //HERE IS THE ERROR
    println("6. multiply");
    Matrix hiddenGradient = multiply(this.H.dSigmoidEach(), hiddenErrors.transpose());
    hiddenGradient.scl(this.learningRate);

    //calculate the deltas of the inputHidden weights
    //HERE IS THE ERROR
    println("7. multiply");
    Matrix WihDeltas = multiply(hiddenGradient, this.I.transpose());
    println("don't print this");

    //adjust the IH weights by the calculated deltas
    this.Wih.add(WihDeltas);
    //adjust the IH biases by its deltas (basically just the output gradient...)
    this.Bih.add(hiddenGradient);
  }
}