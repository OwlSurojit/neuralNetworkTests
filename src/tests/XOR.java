package tests;

import processing.core.*;

import mainClass.neuralNetwork;

public class XOR extends PApplet {

	neuralNetwork nn;
	final int precision = 10;

	public void setup() {

		nn = new neuralNetwork(2, 8, 1);
		nn.randomiseWeights(0, 1, 2);
		/*
		 * nn.feedForwardDebug(new double[]{0, 0}, true); nn.feedForwardDebug(new
		 * double[]{1, 0}, true); nn.feedForwardDebug(new double[]{1, 1}, true);
		 * nn.feedForwardDebug(new double[]{0, 1}, true);
		 */
	}

	final float rectSize = width / precision;

	public void draw() {

		for (int i = 0; i < 1000; i++) {
			switch (PApplet.parseInt(random(0, 4))) {
			case 0:
				nn.backpropagation(new double[] { 1, 0 }, new double[] { 1 });
				break;
			case 1:
				nn.backpropagation(new double[] { 0, 1 }, new double[] { 1 });
				break;
			case 2:
				nn.backpropagation(new double[] { 0, 0 }, new double[] { 0 });
				break;
			default:
				nn.backpropagation(new double[] { 1, 1 }, new double[] { 0 });
			}
		}

		noStroke();
		background(255);
		for (int i = 0; i < width; i += rectSize) {
			for (int j = 0; j < height; j += rectSize) {
				double res = nn.feedForward(new double[] { i / (width - rectSize), j / (height - rectSize) })[0];
				fill((int) (res * 255));
				rect(i, j, rectSize, rectSize);
			}
		}
	}

	public void mousePressed() {
		setup();
	}

	public void settings() {
		size(400, 400);
	}

	static public void main(String[] passedArgs) {
		String[] appletArgs = new String[] { "tests.XOR" };
		if (passedArgs != null) {
			PApplet.main(concat(appletArgs, passedArgs));
		} else {
			PApplet.main(appletArgs);
		}
	}
}
