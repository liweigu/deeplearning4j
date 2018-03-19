package com.liweigu.dls.offical.nlp;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
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
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

public class LSTMSample {

	public static void run() {
		MultiLayerNetwork net = getNet();

		// 训练数据有10000个样本，每个样本是长度为10的double数组
		DataSet trainData = getRandomData(10, 10000);
		// 测试数据有1个样本，每个样本是长度为10的double数组
		DataSet testData = getRandomData(10, 1);

		NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
		normalizer.fitLabel(true);
		normalizer.fit(trainData);

		normalizer.transform(trainData);
		normalizer.transform(testData);

		// 训练
		int epochs = 20000; // 30000
		for (int i = 0; i < epochs; i++) {
			net.fit(trainData);
			net.rnnClearPreviousState();
		}

		// 用测试数据预测，并查看结果。
		INDArray predicted = net.rnnTimeStep(testData.getFeatureMatrix());

		normalizer.revert(testData);
		normalizer.revertLabels(predicted);

		System.out.println("testData:");
		System.out.println(testData);
		System.out.println("result:");
		System.out.println(predicted);
		try {
			System.in.read();
		} catch (IOException e) {
			e.printStackTrace();
		}
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
			double startValue = Math.random() * 1;
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
		return Math.sin(x) * 1;
	}

	/**
	 * 获取网络
	 * 
	 * @return 网络
	 */
	public static MultiLayerNetwork getNet() {
		// double learningRate = 1e-6;
		Map<Integer, Double> lrSchedule = new HashMap<Integer, Double>();
		lrSchedule = new HashMap<Integer, Double>();
		lrSchedule.put(0, 1e-3);
		lrSchedule.put(16000, 5e-4);
		// lrSchedule.put(24000, 2e-4);

		ISchedule mapSchedule = new MapSchedule(ScheduleType.ITERATION, lrSchedule);
		// double l2 = 1e-6;
		int inNum = 1;
		int hiddenCount = 20;
		int outNum = 1;
		NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
		builder.seed(140);
		builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
		builder.weightInit(WeightInit.XAVIER);
		builder.updater(new Nesterovs(mapSchedule, 0.9)); // NESTEROVS, RMSPROP, ADAGRAD
		// builder.l2(l2);
		ListBuilder listBuilder = builder.list();
		listBuilder.layer(0, new GravesLSTM.Builder().activation(Activation.TANH) // SOFTSIGN, TANH
				.nIn(inNum).nOut(hiddenCount).build());
		listBuilder.layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(hiddenCount).nOut(outNum).build());
		listBuilder.pretrain(false);
		listBuilder.backprop(true);
		MultiLayerConfiguration conf = listBuilder.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();

		// showUI: http://localhost:9000
		UIServer uiServer = UIServer.getInstance();
		StatsStorage statsStorage = new InMemoryStatsStorage();
		uiServer.attach(statsStorage);
		net.setListeners(new StatsListener(statsStorage));

		return net;
	}
}
