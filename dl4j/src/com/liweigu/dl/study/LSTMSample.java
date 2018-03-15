package com.liweigu.dl.study;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * 基于数值的LSTM示例。
 * 
 * @author liweigu
 *
 */
public class LSTMSample {

	public static void main(String[] args) {
		MultiLayerNetwork net = getNet();

		// 训练数据有1500个样本，每个样本是长度为10的double数组
		DataSet trainData = getRandomData(10, 1500);
		// 测试数据有1500个样本，每个样本是长度为10的double数组
		DataSet testData = getRandomData(10, 1);

		// 训练1000次
		int epochs = 1000;
		for (int i = 0; i < epochs; i++) {
			net.fit(trainData);
			net.rnnClearPreviousState();
		}

		// 是否需要rnnClearPreviousState？
		 net.rnnClearPreviousState();
		// 用测试数据预测，并查看结果。
		INDArray predicted = net.rnnTimeStep(testData.getFeatureMatrix());
		System.out.println("testData:");
		System.out.println(testData);
		System.out.println("result:");
		System.out.println(predicted);
	}

	/**
	 * 生成DataSet。方法内部有随机数，每次返回的结果不同。
	 * 
	 * @param size 长度
	 * @param sampleCount 样本个数
	 * @return DataSet
	 */
	public static DataSet getRandomData(int size, int sampleCount) {
		INDArray stackedInputNDArray = null;
		INDArray stackedOutputNDArray = null;
		// 维度是1
		int demension = 1;
		// 以数组保存数据
		double[] input = new double[size * sampleCount * demension];
		double[] output = new double[size * sampleCount * demension];
		for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
			// 随机生成起点值
			double startValue = Math.random() * 100;
			input[sampleIndex * size] = startValue;
			output[sampleIndex * size] = calculateNextValue(startValue);
			for (int i = 1; i < size; i++) {
				input[i + sampleIndex * size] = output[i - 1];
				output[i + sampleIndex * size] = calculateNextValue(input[i]);
			}
		}
		INDArray inputNDArray = Nd4j.create(input, new int[] { sampleCount, demension, size });
		INDArray outputNDArray = Nd4j.create(output, new int[] { sampleCount, demension, size });
		if (stackedInputNDArray == null) {
			stackedInputNDArray = inputNDArray;
		} else {
			// 预留，用于支持多维度数据
			stackedInputNDArray = Nd4j.hstack(inputNDArray, stackedInputNDArray);
		}
		if (stackedOutputNDArray == null) {
			stackedOutputNDArray = outputNDArray;
		} else {
			// 预留，用于支持多维度数据
			stackedOutputNDArray = Nd4j.hstack(outputNDArray, stackedOutputNDArray);
		}
		DataSet dataSet = new DataSet(stackedInputNDArray, stackedOutputNDArray);
		return dataSet;
	}

	/**
	 * 计算下一个值
	 * 
	 * @param x 当前值
	 * @return 下一个值
	 */
	public static double calculateNextValue(double x) {
		return x * 1.1;
		// int n = 1000;
		// return x > n ? x % n : x * 2 + 1;
	}

	/**
	 * 获取网络
	 * 
	 * @return 网络
	 */
	public static MultiLayerNetwork getNet() {
		double learningRate = 0.0005;
		int inNum = 1;
		int hiddenCount = 10;
		int outNum = 1;
		NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
		builder.seed(140);
		builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
		builder.weightInit(WeightInit.XAVIER);
		builder.updater(Updater.NESTEROVS);
		builder.learningRate(learningRate);
		ListBuilder listBuilder = builder.list();
		listBuilder.layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(inNum).nOut(hiddenCount).build());
		listBuilder.layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(hiddenCount).nOut(outNum).build());
		MultiLayerConfiguration conf = listBuilder.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();

		// showUI
		UIServer uiServer = UIServer.getInstance();
		StatsStorage statsStorage = new InMemoryStatsStorage();
		uiServer.attach(statsStorage);
		net.setListeners(new StatsListener(statsStorage));

		return net;
	}
}
