package com.liweigu.dls.offical.nlp;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
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
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

/**
 * LSTM示例。4维输入值预测2维输出值，使用labelsMask。
 * 
 * @author liweigu
 *
 */
public class LSTMSample3 {
	static int InDemension = 4;
	static int OutDemension = 2;

	/**
	 * 训练模型和测试
	 */
	public static void run() {
		System.out.println("LSTMSample3.");
		MultiLayerNetwork net = getNet();

		// 训练数据有10000个样本，每个样本是长度为10的double数组
		DataSet trainData = getRandomData(50, 10000, InDemension, OutDemension);
		// 测试数据有1个样本，每个样本是长度为10的double数组
		DataSet testData = getRandomData(50, 1, InDemension, OutDemension);

		// 将数据标准化到0~1之间
		NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
		normalizer.fitLabel(true);
		normalizer.fit(trainData);
		normalizer.transform(trainData);
		normalizer.transform(testData);

		// 训练，设置迭代次数。
		int epochs = 300; // 30000
		for (int i = 0; i < epochs; i++) {
			net.fit(trainData);
			// LSTM需要清理状态，再进行下轮训练或测试。
			net.rnnClearPreviousState();
		}

		// 用测试数据预测，并查看结果。
		INDArray labelMaskArray = testData.getLabelsMaskArray();
		INDArray lastTimeStepIndices = Nd4j.argMax(labelMaskArray, 1);

		INDArray predicted = net.rnnTimeStep(testData.getFeatureMatrix());

		// 还原数据以便于查看
		normalizer.revert(testData);
		normalizer.revertLabels(predicted);

		System.out.println("testData:");
		System.out.println(testData);
		System.out.println("result:");
		// System.out.println(predicted);
		int numExamples = testData.getFeatureMatrix().size(0);
		for (int i = 0; i < numExamples; i++) {
			int thisTimeSeriesLastIndex = lastTimeStepIndices.getInt(i);
			INDArray thisExampleProbabilities = predicted.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(thisTimeSeriesLastIndex));
			System.out.println(thisExampleProbabilities);
		}

		try {
			// 让程序挂起不退出
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
	 * @return DataSet 随机数据
	 */
	public static DataSet getRandomData(int size, int sampleCount, int inDemension, int outDemension) {
		// 以数组保存数据
		// [miniBatchSize,inputSize,timeSeriesLength]
		double[] input = new double[size * sampleCount * inDemension];
		double[] output = new double[size * sampleCount * outDemension];
		double[] outputMask = new double[sampleCount * size];
		for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
			List<double[]> nextValues = new ArrayList<double[]>();
			double[] nextValue = null;
			for (int sizeIndex = 0; sizeIndex < size * inDemension; sizeIndex++) {
				nextValue = calculateNextValue(nextValue, inDemension);
				nextValues.add(nextValue);
			}
			double[] labelValue = calculateLabel(nextValue, outDemension);
			for (int demensionIndex = 0; demensionIndex < inDemension; demensionIndex++) {
				for (int sizeIndex = 0; sizeIndex < size; sizeIndex++) {
					input[sizeIndex + demensionIndex * size + sampleIndex * size * inDemension] = nextValues
							.get(sizeIndex + demensionIndex * size)[demensionIndex];
				}
			}
			for (int demensionIndex = 0; demensionIndex < outDemension; demensionIndex++) {
				for (int sizeIndex = 0; sizeIndex < size; sizeIndex++) {
					if (sizeIndex == size - 1) {
						output[sizeIndex + demensionIndex * size + sampleIndex * size * outDemension] = labelValue[demensionIndex];
					} else {
						output[sizeIndex + demensionIndex * size + sampleIndex * size * outDemension] = 0;
					}
				}
			}
			for (int sizeIndex = 0; sizeIndex < size; sizeIndex++) {
				outputMask[sizeIndex + sampleIndex * size] = sizeIndex == size - 1 ? 1 : 0;
			}
		}
		INDArray inputNDArray = Nd4j.create(input, new int[] { sampleCount, inDemension, size });
		INDArray outputNDArray = Nd4j.create(output, new int[] { sampleCount, outDemension, size });
		// [miniBatchSize, timeSeriesLength]
		INDArray featuresMask = null;
		INDArray labelsMask = Nd4j.create(outputMask, new int[] { sampleCount, size });
		DataSet dataSet = new DataSet(inputNDArray, outputNDArray, featuresMask, labelsMask);
		return dataSet;
	}

	/**
	 * 计算下一个值
	 * 
	 * @param x 当前值
	 * @return 下一个值
	 */
	public static double[] calculateNextValue(double[] x, int demension) {
		double[] result = new double[demension];

		if (demension == 4) {
			if (x == null || x.length == 0) {
				// 生成值在 0.5~1 之间
				result[0] = Math.random() / 2 + 0.5;
				result[1] = Math.random();
				result[2] = Math.random();
				result[3] = Math.random();
			} else {
				result[0] = Math.sin(x[0]);
				result[1] = x[1] + 2;
				result[2] = x[2] - 1;
				result[3] = x[3];
			}
		} else {
			throw new IllegalArgumentException("x = " + Arrays.toString(x) + ", demension = " + demension);
		}

		return result;
	}

	/**
	 * 计算标签值
	 * 
	 * @param x 当前值
	 * @return 标签值
	 */
	public static double[] calculateLabel(double[] x, int outDemension) {
		double[] result = new double[outDemension];

		if (outDemension == 2) {
			result[0] = Math.sin(x[0]);
			result[1] = x[1] + x[2] + x[3];
		} else {
			throw new IllegalArgumentException("x = " + Arrays.toString(x) + ", outDemension = " + outDemension);
		}

		return result;
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
		double l2 = 1e-6;
		int inNum = InDemension;
		int hiddenCount = 20;
		int outNum = OutDemension;
		NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
		builder.seed(140);
		builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
		builder.weightInit(WeightInit.XAVIER);
		builder.updater(new Nesterovs(mapSchedule, 0.9)); // NESTEROVS, RMSPROP, ADAGRAD
		builder.l2(l2);
		ListBuilder listBuilder = builder.list();
		listBuilder.layer(0, new GravesLSTM.Builder().activation(Activation.TANH) // SOFTSIGN, TANH
				.nIn(inNum).nOut(hiddenCount).build());
		listBuilder.layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(hiddenCount).nOut(outNum).build());
		listBuilder.pretrain(false);
		listBuilder.backprop(true);
		MultiLayerConfiguration conf = listBuilder.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();

		// 显示训练过程: http://localhost:9000
		UIServer uiServer = UIServer.getInstance();
		StatsStorage statsStorage = new InMemoryStatsStorage();
		uiServer.attach(statsStorage);
		net.setListeners(new StatsListener(statsStorage));

		return net;
	}
}
