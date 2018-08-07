package tests;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import javax.swing.JButton;
import javax.swing.JFrame;

import mainClass.neuralNetwork;

final public class Converters implements ActionListener {

	public static final int STOP_ON_INPUT = -1, INFINITELY = -1;

	private static double[] decToBin(int dec, int preferredLength) {
		/*
		 * int maxExp = 0;
		 * while (Math.pow(2, maxExp) < dec) maxExp++;
		 */

		double[] res = new double[preferredLength];
		for (int i = preferredLength - 1; i > 0; i--) {
			res[preferredLength - i - 1] = (int) (dec / Math.pow(2, i));
			if (res[preferredLength - i - 1] == 1)
				dec -= Math.pow(2, i);
		}

		return res;
	}

	private static boolean stop = false;

	public static neuralNetwork binToDec(double learningRate, int hiddenNodes, int maxBinDigits,
			int numOfTrainingSessions) {

		/**
		 * There are two modes: 1 - numOfTrainingSessions is defined as a natural number; 
		 * 	The neural network is then trained that many times. 
		 * 2 - numOfTrainingSessions is defined as STOP_ON_INPUT or INFINITELY; 
		 * 	The neural network is then trained until the user interrupts by pressing ENTER.
		 */

		int maxDec = (int) Math.pow(2, maxBinDigits);
		neuralNetwork nn = new neuralNetwork(maxBinDigits, hiddenNodes, maxDec);
		nn.setLearningRate(learningRate);
		nn.randomiseWeights(0, 1, 2);
		double[] dec = new double[maxDec];
		int x;

		if (numOfTrainingSessions == STOP_ON_INPUT || numOfTrainingSessions == INFINITELY) {
			JFrame frame = new JFrame();
			frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
			frame.setResizable(false);
			frame.setSize(160, 90);
			frame.setTitle("Stop the training");
			frame.setLocation(320, 180);
			//frame.addMou
			JButton b = new JButton("Stop");
			b.setActionCommand("stop");
			b.addActionListener(this);
			frame.getContentPane().add(b);
			frame.setVisible(true);
			int counter = 0;
			while (!stop) {
				x = (int) (Math.random() * maxDec);
				dec[x] = 1;
				nn.backpropagation(decToBin(x, 6), dec);
				dec[x] = 0;
				counter++;
			}
			System.out.println("trained " + counter + " times");
		} else {

			for (int i = 0; i < numOfTrainingSessions; i++) {
				if (i % (numOfTrainingSessions / 100) == 0)
					System.out.println((int) (100 * i / numOfTrainingSessions) + "% done");
				x = (int) (Math.random() * maxDec);
				dec[x] = 1;
				/*
				 * if (x == 52) System.out.println(dec[52] + " " + dec[53] + " " + decToBin(x,
				 * 6)[0] + " " + decToBin(x, 6)[1] + " " + decToBin(x, 6)[2] + " " + decToBin(x,
				 * 6)[3] + " " + decToBin(x, 6)[4] + " " + decToBin(x, 6)[5]);
				 */
				nn.backpropagation(decToBin(x, 6), dec);
				dec[x] = 0;
			}
		}

		return nn;
	}

	public static void main(String[] args) {
		// System.out.println(decToBin(52, 1250).length);
		binToDec(.1, 128, 4, STOP_ON_INPUT).feedForwardDebug(new double[] { 0, 1, 0, 0 }, true);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getActionCommand().equals("stop")) {
			stop = true;
			System.out.println("now");
		}
	}

}
