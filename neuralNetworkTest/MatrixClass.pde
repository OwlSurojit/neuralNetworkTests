class Matrix {

  int rows, cols, digitsAfterComma;
  float[][] arr;

  Matrix(int rows, int cols) {
    this.rows = rows; 
    this.cols = cols;
    this.arr = new float[rows][cols];
  }

  void define(float[] def) {
    for (int i=0; i<this.rows; i++) {
      for (int j=0; j<this.cols; j++) {
        this.arr[i][j] = def[i+j];
      }
    }
  }

  void randomize(float lowest, float highest, int digitsAfterComma) {
    this.digitsAfterComma = digitsAfterComma;
    for (int i=0; i<this.rows; i++) {
      for (int j=0; j<this.cols; j++) {
        this.arr[i][j] = roundDigits((float) (Math.random()*highest+lowest), this.digitsAfterComma);
      }
    }
  }

  void scl(float scale) {
    for (int i=0; i<this.rows; i++) {
      for (int j=0; j<this.cols; j++) {
        this.arr[i][j] *= scale;
      }
    }
  }

  Matrix transpose() {
    Matrix transposed = new Matrix(this.cols, this.rows);
    for (int i=0; i<this.rows; i++) {
      for (int j=0; j<this.cols; j++) {
        transposed.arr[j][i] = this.arr[i][j];
      }
    }
    return transposed;
  }

  void debug() {
    for (float[] i : this.arr) {
      print("[  ");
      for (float j : i) {
        print(nf(j, 0, this.digitsAfterComma) + "  ");
      }
      print("]\n");
    }
    println();
  }

  void sigmoidEach() {
    for (int i=0; i<this.rows; i++) {
      for (int j=0; j<this.cols; j++) {
        this.arr[i][j] = sigmoid(this.arr[i][j]);
      }
    }
  }

  Matrix dSigmoidEach() {
    Matrix result = new Matrix(this.rows, this.cols);
    for (int i=0; i<this.rows; i++) {
      for (int j=0; j<this.cols; j++) {
        result.arr[i][j] = dSigmoid(this.arr[i][j]);
      }
    }
    return result;
  }

  void add(Matrix b) throws IllegalArgumentException {
    if (this.rows!=b.rows || this.cols!=b.cols) {
      throw new IllegalArgumentException("The matricies to add haven't got the same number of rows and columns!");
    }
    for (int i=0; i<this.rows; i++) {
      for (int j=0; j<this.cols; j++) {
        this.arr[i][j] += b.arr[i][j];
      }
    }
  }
  private float roundDigits(float num, int digitsAfterComma) {
    int factor = (int) pow(10, digitsAfterComma);
    num = round(num*factor);
    num/=factor;
    return num;
  }
}

Matrix add(Matrix a, Matrix b) throws IllegalArgumentException {
  if (a.rows!=b.rows || a.cols!=b.cols) {
    throw new IllegalArgumentException("The matricies to add haven't got the same number of rows and columns!");
  }
  Matrix result = new Matrix(a.rows, a.cols);
  result.digitsAfterComma = a.digitsAfterComma;
  for (int i=0; i<a.rows; i++) {
    for (int j=0; j<a.cols; j++) {
      result.arr[i][j] = a.arr[i][j] + b.arr[i][j];
    }
  }
  return result;
}

Matrix substract(Matrix a, Matrix b) throws IllegalArgumentException {
  if (a.rows!=b.rows || a.cols!=b.cols) {
    throw new IllegalArgumentException("The matricies haven't got the same number of rows and columns!");
  }
  Matrix result = new Matrix(a.rows, a.cols);
  result.digitsAfterComma = a.digitsAfterComma;
  for (int i=0; i<a.rows; i++) {
    for (int j=0; j<a.cols; j++) {
      result.arr[i][j] = a.arr[i][j] - b.arr[i][j];
    }
  }
  return result;
}

Matrix multiply(Matrix a, Matrix b) throws IllegalArgumentException {
  println(a.rows);
  println(a.cols);
  println(b.rows);
  println(b.cols);
  if (a.cols!=b.rows) {
    throw new IllegalArgumentException("The first matrix' number of columns isn't equal to the second matrix' number of rows!");
  }
  Matrix result = new Matrix(a.rows, b.cols);
  result.digitsAfterComma = a.digitsAfterComma;
  for (int i=0; i<result.rows; i++) {
    for (int j=0; j<result.cols; j++) {
      for (int k=0; k<a.cols; k++) {
        result.arr[i][j]+=a.arr[i][k]*b.arr[k][j];
      }
    }
  }
  return result;
}





public static float sigmoid(float x) {
  return 1/(1+exp(-x));
}

public static float dSigmoid(float x) {
  //sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
  //return sigmoid(x) * (1 - sigmoid(x)) ;

  //but here the output is already "sigmoided", therefore:
  return x * (1 - x);
}