package com.liweigu.dl.study;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.WindowConstants;

import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.RefineryUtilities;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import com.liweigu.dl.util.DLTool;
import com.liweigu.spark.SparkToolBase;

public class LSTMStudy extends SparkToolBase {
	private static boolean showUI = true;
	private static boolean showPlot = true;

	public LSTMStudy(String appName, String master) {
		super(appName, master);
	}

	public static void main(String[] args) {
		System.out.println("[dl] LSTMStudy");
		String appName = "LSTMStudy";
		String master = null;
		LSTMStudy sparkTool = new LSTMStudy(appName, master);
		try {
			 sparkTool.runSingle();
//			sparkTool.runMulty();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void runSingle() throws Exception {
		// 1，跟文件内容相同的测试数据
		double[] trainInput = new double[] { 112.00, 118.00, 132.00, 129.00, 121.00, 135.00, 148.00, 148.00, 136.00, 119.00, 104.00, 118.00, 115.00, 126.00,
				141.00, 135.00, 125.00, 149.00, 170.00, 170.00, 158.00, 133.00, 114.00, 140.00, 145.00, 150.00, 178.00, 163.00, 172.00, 178.00, 199.00, 199.00,
				184.00, 162.00, 146.00, 166.00, 171.00, 180.00, 193.00, 181.00, 183.00, 218.00, 230.00, 242.00, 209.00, 191.00, 172.00, 194.00, 196.00, 196.00,
				236.00, 235.00, 229.00, 243.00, 264.00, 272.00, 237.00, 211.00, 180.00, 201.00, 204.00, 188.00, 235.00, 227.00, 234.00, 264.00, 302.00, 293.00,
				259.00, 229.00, 203.00, 229.00, 242.00, 233.00, 267.00, 269.00, 270.00, 315.00, 364.00, 347.00, 312.00, 274.00, 237.00, 278.00, 284.00, 277.00,
				317.00, 313.00, 318.00, 374.00, 413.00, 405.00, 355.00, 306.00, 271.00, 306.00, 315.00, 301.00, 356.00, 348.00 };
		double[] trainOutput = new double[] { 118.00, 132.00, 129.00, 121.00, 135.00, 148.00, 148.00, 136.00, 119.00, 104.00, 118.00, 115.00, 126.00, 141.00,
				135.00, 125.00, 149.00, 170.00, 170.00, 158.00, 133.00, 114.00, 140.00, 145.00, 150.00, 178.00, 163.00, 172.00, 178.00, 199.00, 199.00, 184.00,
				162.00, 146.00, 166.00, 171.00, 180.00, 193.00, 181.00, 183.00, 218.00, 230.00, 242.00, 209.00, 191.00, 172.00, 194.00, 196.00, 196.00, 236.00,
				235.00, 229.00, 243.00, 264.00, 272.00, 237.00, 211.00, 180.00, 201.00, 204.00, 188.00, 235.00, 227.00, 234.00, 264.00, 302.00, 293.00, 259.00,
				229.00, 203.00, 229.00, 242.00, 233.00, 267.00, 269.00, 270.00, 315.00, 364.00, 347.00, 312.00, 274.00, 237.00, 278.00, 284.00, 277.00, 317.00,
				313.00, 318.00, 374.00, 413.00, 405.00, 355.00, 306.00, 271.00, 306.00, 315.00, 301.00, 356.00, 348.00, 355.00 };
		double[] testInput = new double[] { 355.00, 422.00, 465.00, 467.00, 404.00, 347.00, 305.00, 336.00, 340.00, 318.00, 362.00, 348.00, 363.00, 435.00,
				491.00, 505.00, 404.00, 359.00, 310.00, 337.00, 360.00, 342.00, 406.00, 396.00, 420.00, 472.00, 548.00, 559.00, 463.00, 407.00, 362.00, 405.00,
				417.00, 391.00, 419.00, 461.00, 472.00, 535.00, 622.00, 606.00, 508.00, 461.00, 390.00 };
		double[] testOutput = new double[] { 422.00, 465.00, 467.00, 404.00, 347.00, 305.00, 336.00, 340.00, 318.00, 362.00, 348.00, 363.00, 435.00, 491.00,
				505.00, 404.00, 359.00, 310.00, 337.00, 360.00, 342.00, 406.00, 396.00, 420.00, 472.00, 548.00, 559.00, 463.00, 407.00, 362.00, 405.00, 417.00,
				391.00, 419.00, 461.00, 472.00, 535.00, 622.00, 606.00, 508.00, 461.00, 390.00, 432.00 };
		// 2，模拟测试数据
		double[] testArrData = new double[150];
		for (int i = 0; i < testArrData.length; i++) {
			double data;
			// data = i * 2 - 1;
			if (i % 3 == 0) {
				data = i;
			} else if (i % 3 == 1) {
				data = i + 2;
			} else if (i % 3 == 2) {
				data = i * 2;
			} else {
				data = i;
			}
			testArrData[i] = data;
		}
		trainInput = new double[100];
		trainOutput = new double[100];
		for (int i = 0; i < trainInput.length; i++) {
			trainInput[i] = testArrData[i];
			trainOutput[i] = testArrData[i + 1];
		}
		testInput = new double[30];
		testOutput = new double[30];
		for (int i = 0; i < testInput.length; i++) {
			testInput[i] = testArrData[99 + i];
			testOutput[i] = testArrData[99 + i + 1];
		}

		// 计算
		INDArray trainInputNDArray = Nd4j.create(trainInput, new int[] { 1, 1, trainInput.length });
		// INDArray inputNDArray = Nd4j.hstack(inputNDArrays);
		INDArray trainOutputNDArray = Nd4j.create(trainOutput, new int[] { 1, 1, trainOutput.length });
		DataSet trainData = new DataSet(trainInputNDArray, trainOutputNDArray);
		// System.out.println(trainData);
		// System.out.println(trainData.getFeatures().rank());
		INDArray testInputNDArray = Nd4j.create(testInput, new int[] { 1, 1, testInput.length });
		INDArray testOutputNDArray = Nd4j.create(testOutput, new int[] { 1, 1, testOutput.length });
		DataSet testData = new DataSet(testInputNDArray, testOutputNDArray);
		// System.out.println(testData);
		// System.out.println(testData.getFeatures().rank());

		// Normalize data, including labels (fitLabel=true)
		NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
		normalizer.fitLabel(true);
		normalizer.fit(trainData); // Collect training data statistics

		normalizer.transform(trainData);
		normalizer.transform(testData);

		// ----- Configure the network -----
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(140).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.iterations(1).weightInit(WeightInit.XAVIER).updater(Updater.NESTEROVS)
//				.learningRate(0.0015)
				.list()
				.layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build())
				.layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(10).nOut(1).build()).build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();

		if (showUI) {
			DLTool.showUI(net);
		}

		// net.setListeners(new ScoreIterationListener(20));

		// ----- Train the network, evaluating the test set performance at each epoch -----
		int nEpochs = 300;

		for (int i = 0; i < nEpochs; i++) {
			net.fit(trainData);
			System.out.println("Epoch " + i + " complete. Time series evaluation:");

			// Run regression evaluation on our single column input
			RegressionEvaluation evaluation = new RegressionEvaluation(1);
			INDArray features = testData.getFeatureMatrix();

			INDArray lables = testData.getLabels();
			INDArray predicted = net.output(features, false);

			evaluation.evalTimeSeries(lables, predicted);

			// Just do sout here since the logger will shift the shift the columns of the stats
			System.out.println(evaluation.stats());
		}

		// Init rrnTimeStemp with train data and predict test data
		net.rnnTimeStep(trainData.getFeatureMatrix());
		// net.rnnClearPreviousState();
		INDArray predicted = net.rnnTimeStep(testData.getFeatureMatrix());

		// net.predict(d)

		// Revert data back to original values for plotting
		normalizer.revert(trainData);
		normalizer.revert(testData);
		normalizer.revertLabels(predicted);

		boolean show = true;
		if (show) {
			INDArray trainFeatures = trainData.getFeatures();
			INDArray testFeatures = testData.getFeatures();
			// Create plot with out data
			XYSeriesCollection c = new XYSeriesCollection();
			createSeries(c, trainFeatures, 0, "Train data");
			createSeries(c, testFeatures, 99, "Actual test data");
			createSeries(c, predicted, 100, "Predicted test data");

			plotDataset(c);
		}

		System.out.println("----- Example Complete -----");
	}

	public void runMulty() throws Exception {
		System.out.println("runMulty()");
		// 模拟测试数据
		int dataSize = 1600;
		double[] testArrDataX = new double[dataSize];
		for (int i = 0; i < testArrDataX.length; i++) {
			testArrDataX[i] = i + 2;
		}
		double[] testArrDataY = new double[dataSize];
		for (int i = 0; i < testArrDataY.length; i++) {
			testArrDataY[i] = i;
		}
		// 按dl4j要求的格式构造测试数据
		int trainSize = 100;
		int demension = 2;
		int groupCount = 1200;
		double[] trainInput = new double[trainSize * demension * groupCount];
		double[] trainOutput = new double[trainSize * demension * groupCount];
		for (int j = 0; j < groupCount; j++) {
			for (int i = 0; i < trainSize; i++) {
				trainInput[j * trainSize * demension + i] = testArrDataX[j + i];
				if (i < trainSize - 1) {
					trainOutput[j * trainSize * demension + i] = 0;
				} else {
					trainOutput[j * trainSize * demension + i] = testArrDataX[j + trainSize];
					// System.out.println("trainOutput[" + i + "] = " + trainOutput[i]);
				}
			}
			for (int i = 0; i < trainSize; i++) {
				trainInput[j * trainSize * demension + trainSize + i] = testArrDataY[j + trainSize + i];
				if (i < trainSize - 1) {
					trainOutput[j * trainSize * demension + trainSize + i] = 0;
				} else {
					trainOutput[j * trainSize * demension + trainSize + i] = testArrDataY[j + trainSize];
				}
			}
		}
		int testSize = 60;
		int testStartIndex = 99;
		double[] testInput = new double[testSize * demension];
		double[] testOutput = new double[testSize * demension];
		for (int i = 0; i < testSize; i++) {
			testInput[i] = testArrDataX[testStartIndex + i];
			if (i < testSize - 1) {
				testOutput[i] = 0;
			} else {
				testOutput[i] = testArrDataX[testStartIndex + testSize];
			}
		}
		for (int i = 0; i < testSize; i++) {
			testInput[testSize + i] = testArrDataY[testSize + testStartIndex + i];
			if (i < testSize - 1) {
				testOutput[testSize + i] = 0;
			} else {
				testOutput[testSize + i] = testArrDataY[testSize * 2 + testStartIndex];
			}
		}

		// 计算
		INDArray trainInputNDArray = Nd4j.create(trainInput, new int[] { groupCount, demension, trainSize });
		// INDArray inputNDArray = Nd4j.hstack(inputNDArrays);
		INDArray trainOutputNDArray = Nd4j.create(trainOutput, new int[] { groupCount, demension, trainSize });
		DataSet trainData = new DataSet(trainInputNDArray, trainOutputNDArray);
		// System.out.println("trainData = " + trainData);
		INDArray testInputNDArray = Nd4j.create(testInput, new int[] { 1, demension, testSize });
		INDArray testOutputNDArray = Nd4j.create(testOutput, new int[] { 1, demension, testSize });
		DataSet testData = new DataSet(testInputNDArray, testOutputNDArray);
		// System.out.println("testData = " + testData);

		// Normalize data, including labels (fitLabel=true)
		NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
		normalizer.fitLabel(true);
		normalizer.fit(trainData); // Collect training data statistics

		normalizer.transform(trainData);
		normalizer.transform(testData);

		// ----- Configure the network -----
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(140).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.iterations(1).weightInit(WeightInit.XAVIER).updater(Updater.NESTEROVS)
//				.learningRate(0.0015)
				.list()
				.layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(demension).nOut(10).build())
				.layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(10).nOut(demension).build()).build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();

		if (showUI) {
			DLTool.showUI(net);
		}

		// net.setListeners(new ScoreIterationListener(20));

		// ----- Train the network, evaluating the test set performance at each epoch -----
		int nEpochs = 300 / 3;

		for (int i = 0; i < nEpochs; i++) {
			net.fit(trainData);
			System.out.println("Epoch " + i + " complete. Time series evaluation:");

			// labels.size(1)的值是demension
			RegressionEvaluation evaluation = new RegressionEvaluation(demension);
			INDArray features = testData.getFeatureMatrix();

			INDArray labels = testData.getLabels();
			INDArray predicted = net.output(features, false);

			// System.out.println("labels.size(1) = " + labels.size(1));
			// System.out.println("predicted.size(1) = " + predicted.size(1));
			// System.out.println("labels = " + labels);
			// System.out.println("predicted = " + predicted);

			evaluation.evalTimeSeries(labels, predicted);

			// Just do sout here since the logger will shift the shift the columns of the stats
			System.out.println(evaluation.stats());
		}

		// Init rrnTimeStemp with train data and predict test data
		// net.rnnTimeStep(trainData.getFeatureMatrix());
		// net.rnnClearPreviousState();
		INDArray predicted = net.rnnTimeStep(testData.getFeatureMatrix());

		// Revert data back to original values for plotting
		normalizer.revert(trainData);
		normalizer.revert(testData);
		normalizer.revertLabels(predicted);

		INDArray d = testInputNDArray;
		System.out.println("d: " + d);
		int[] predictResult = net.predict(d);
		System.out.println("predictResult: " + predictResult.length);
		for (int i : predictResult) {
			System.out.println(i);
		}

		if (showPlot) {
			INDArray trainFeatures = trainData.getFeatures();
			INDArray testFeatures = testData.getFeatures();
			// Create plot with out data
			XYSeriesCollection c = new XYSeriesCollection();
			createSeries(c, trainFeatures, 0, "Train data");
			createSeries(c, testFeatures, 99, "Actual test data");
			createSeries(c, predicted, 100, "Predicted test data");

			plotDataset(c);
		}

		System.out.println("----- LSTMStudy Complete -----");
	}

	private static void createSeries(XYSeriesCollection seriesCollection, INDArray data, int offset, String name) {
		int nRows = data.shape()[2];
		XYSeries series = new XYSeries(name);
		for (int i = 0; i < nRows; i++) {
			series.add(i + offset, data.getDouble(i));
		}
		seriesCollection.addSeries(series);
	}

	/**
	 * Generate an xy plot of the datasets provided.
	 */
	private static void plotDataset(XYSeriesCollection c) {
		String title = "Regression example";
		String xAxisLabel = "Timestep";
		String yAxisLabel = "Number of passengers";
		PlotOrientation orientation = PlotOrientation.VERTICAL;
		boolean legend = true;
		boolean tooltips = false;
		boolean urls = false;
		JFreeChart chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, c, orientation, legend, tooltips, urls);

		// get a reference to the plot for further customisation...
		final XYPlot plot = chart.getXYPlot();

		// Auto zoom to fit time series in initial window
		final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
		rangeAxis.setAutoRange(true);

		JPanel panel = new ChartPanel(chart);

		JFrame f = new JFrame();
		f.add(panel);
		f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		f.pack();
		f.setTitle("Training Data");

		RefineryUtilities.centerFrameOnScreen(f);
		f.setVisible(true);
	}
}
