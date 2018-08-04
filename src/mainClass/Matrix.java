package mainClass;

public class Matrix {

  int rows, cols, digitsAfterComma;
  double[][] arr;

  public Matrix(int rows, int cols) {
    this.rows = rows; 
    this.cols = cols;
    this.arr = new double[rows][cols];
  }

  public void define(double[] def) {
    for (int i=0; i<this.rows; i++) {
      for (int j=0; j<this.cols; j++) {
        this.arr[i][j] = def[i*cols+j];
      }
    }
  }

  public void randomise(double lowest, double highest, int digitsAfterComma) {
    this.digitsAfterComma = digitsAfterComma;
    for (int i=0; i<this.rows; i++) {
      for (int j=0; j<this.cols; j++) {
        this.arr[i][j] = roundDigits((double) (Math.random()*highest+lowest), this.digitsAfterComma);
      }
    }
  }

  public void scl(double scale) {
    for (int i=0; i<this.rows; i++) {
      for (int j=0; j<this.cols; j++) {
        this.arr[i][j] *= scale;
      }
    }
  }

  public Matrix transpose() {
    Matrix transposed = new Matrix(this.cols, this.rows);
    for (int i=0; i<this.rows; i++) {
      for (int j=0; j<this.cols; j++) {
        transposed.arr[j][i] = this.arr[i][j];
      }
    }
    return transposed;
  }

  public void debug() {
    for (double[] i : this.arr) {
      System.out.print("[  ");
      for (double j : i) {
        System.out.print(roundDigits(j, this.digitsAfterComma) + "  ");
      }
      System.out.print("]\n");
    }
    System.out.println();
  }

  public void sigmoidEach() {
    for (int i=0; i<this.rows; i++) {
      for (int j=0; j<this.cols; j++) {
        this.arr[i][j] = sigmoid(this.arr[i][j]);
      }
    }
  }

  public Matrix dSigmoidEach() {
    Matrix result = new Matrix(this.rows, this.cols);
    for (int i=0; i<this.rows; i++) {
      for (int j=0; j<this.cols; j++) {
        result.arr[i][j] = dSigmoid(this.arr[i][j]);
      }
    }
    return result;
  }

  public void add(Matrix b) throws IllegalArgumentException {
    if (this.rows!=b.rows || this.cols!=b.cols) {
      throw new IllegalArgumentException("The matricies to add haven't got the same number of rows and columns!");
    }
    for (int i=0; i<this.rows; i++) {
      for (int j=0; j<this.cols; j++) {
        this.arr[i][j] += b.arr[i][j];
      }
    }
  }

  public void multiply(Matrix b) {
    // hadamard product
    for (int i=0; i<this.rows; i++) {
      for (int j=0; j<this.cols; j++) {
        this.arr[i][j] *= b.arr[i][j];
      }
    }
  }

  private double roundDigits(double num, int digitsAfterComma) {
    int factor = (int) Math.pow(10, digitsAfterComma);
    num = Math.round(num*factor);
    num/=factor;
    return num;
  }


public static Matrix add(Matrix a, Matrix b) throws IllegalArgumentException {
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


public static Matrix substract(Matrix a, Matrix b) throws IllegalArgumentException {
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


public static Matrix multiply(Matrix a, Matrix b) throws IllegalArgumentException {
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


public static double sigmoid(double x) {
  return (1/(1 + Math.exp(-x)));
}

public static double dSigmoid(double x) {
  //sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
  //return sigmoid(x) * (1 - sigmoid(x)) ;

  //but here the output is already "sigmoided", therefore:
  return x * (1 - x);
}
}
