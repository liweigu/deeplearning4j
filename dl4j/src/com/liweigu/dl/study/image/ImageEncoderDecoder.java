package com.liweigu.dl.study.image;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.Deconvolution2D;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.Upsampling2D;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * （灰度）图片编码解码，输出跟输入相同。
 * 测试代码。
 * 
 * @author liweigu
 *
 */
public class ImageEncoderDecoder {
	private static Logger LOGGER = LoggerFactory.getLogger(ImageEncoderDecoder.class);
	static String BasePath = "E:/data/img/";
	// 灰度图的channels是1，彩色图片的channels是3。
	private static int channels = 1;
	// 图片宽高
	private static int size = 501;

	public static void run() throws IOException {
		LOGGER.info("ImageEncoderDecoder");

		MultiLayerNetwork multiLayerNetwork = getNetwork();

		File trainDataFile = new File(BasePath);
		FileSplit train = new FileSplit(trainDataFile, NativeImageLoader.ALLOWED_FORMATS);
		LOGGER.info("train.length() = " + train.length());

		ImageRecordReader trainRecordReader = new ImageRecordReader(size, size, channels);
		trainRecordReader.initialize(train);

		int batchSize = 1; // 128
		LOGGER.info("batchSize = " + batchSize);
		RecordReaderDataSetIterator iterTrain = new RecordReaderDataSetIterator(trainRecordReader, batchSize);

		LOGGER.info("scaler.fit");
		DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
		scaler.fit(iterTrain);
		iterTrain.setPreProcessor(scaler);

		iterTrain.setCollectMetaData(true);

		LOGGER.info("training...");
		int numEpochs = 100;
//		numEpochs = 1;
		for (int i = 0; i < numEpochs; i++) {
			if (i % 10 == 0) {
				LOGGER.info("i = " + i);
			}
			iterTrain.reset();
			while (iterTrain.hasNext()) {
				// 读取图片数据
				DataSet batchData = iterTrain.next(batchSize);
				// 转换DataSet，让 feature和label都是图像数据，即结果要跟输入一致。
				if (batchData != null) {
					if (batchData.numExamples() > 1) {
						List<DataSet> dataSetList = batchData.asList();
						List<DataSet> trainDataSetList = new ArrayList<DataSet>();
						for (DataSet dataSet : dataSetList) {
							INDArray features = dataSet.getFeatures();
							INDArray labels = features;
							DataSet trainDataSet = new DataSet(features, labels);
							trainDataSetList.add(trainDataSet);
						}
						DataSet trainDataSets = DataSet.merge(trainDataSetList);
						multiLayerNetwork.fit(trainDataSets);
					} else {
						DataSet dataSet = batchData;
						INDArray features = dataSet.getFeatures();
						INDArray labels = features;
						DataSet trainDataSet = new DataSet(features, labels);
						multiLayerNetwork.fit(trainDataSet);
					}
				} else {
					LOGGER.info("batchData is null.");
				}
			}
		}
		LOGGER.info("training is over.");

		LOGGER.info("testing...");
		iterTrain.reset();
		while (iterTrain.hasNext()) {
			DataSet ds = iterTrain.next();
			INDArray features = ds.getFeatures();
			int totalShape = 1;
			for (int shape : features.shape()) {
				totalShape *= shape;
			}
			INDArray label = features.reshape(1, totalShape);
			DataSet testDataSet = new DataSet(features, label);
			double d = multiLayerNetwork.score(testDataSet);
			System.out.println(d);
		}

		System.in.read();
	}

	public static MultiLayerNetwork getNetwork() {
		Map<Integer, Double> lrSchedule = new HashMap<>();
		lrSchedule.put(0, 1e-4);

		long seed = 42;
		// double l2 = 1e-6;
		ISchedule mapSchedule = new MapSchedule(ScheduleType.ITERATION, lrSchedule);

		Upsampling2D unmaxool2 = new Upsampling2D();
		unmaxool2.setSize(new int[] { 2, 2 });
		unmaxool2.setLayerName("unmaxool2");

		Upsampling2D unmaxool1 = new Upsampling2D();
		unmaxool1.setSize(new int[] { 2, 2 });
		unmaxool1.setLayerName("unmaxool1");

		ListBuilder listBuilder = new NeuralNetConfiguration.Builder().seed(seed).weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.activation(Activation.RELU)
				.updater(new Nesterovs(mapSchedule, 0.9))
				// .updater(new Adam.Builder().learningRateSchedule(mapSchedule).build())
				// .l2(l2)
				// .dropOut(0.5)
				.list();
		int index = 0;
		listBuilder = listBuilder.layer(index++, new ConvolutionLayer.Builder(new int[] { 5, 5 }, new int[] { 1, 1 }, new int[] { 0, 0 }).name("cnn1")
				.nIn(channels).nOut(50).biasInit(0).build());
		listBuilder = listBuilder.layer(index++, new SubsamplingLayer.Builder(new int[] { 2, 2 }, new int[] { 2, 2 }).name("maxpool1").build());
		listBuilder = listBuilder.layer(index++,
				new ConvolutionLayer.Builder(new int[] { 5, 5 }, new int[] { 5, 5 }, new int[] { 1, 1 }).name("cnn2").nOut(100).biasInit(0).build());
		// listBuilder = listBuilder.layer(index++, new SubsamplingLayer.Builder(new int[] { 2, 2 }, new int[] { 2, 2 }).name("maxool2").build());
		// listBuilder = listBuilder.layer(index++, unmaxool2);
		listBuilder = listBuilder.layer(index++,
				new Deconvolution2D.Builder(new int[] { 5, 5 }, new int[] { 5, 5 }, new int[] { 1, 1 }).name("decnn2").nOut(100).biasInit(0).build());
		listBuilder = listBuilder.layer(index++, unmaxool1);
		listBuilder = listBuilder.layer(index++,
				new Deconvolution2D.Builder(new int[] { 5, 5 }, new int[] { 1, 1 }, new int[] { 0, 0 }).name("decnn1").nOut(1).biasInit(0).build());
		listBuilder = listBuilder.layer(index++, new DenseLayer.Builder().nOut(244036 / 100).build());
		listBuilder = listBuilder.layer(index++,
				new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nOut(size * size * channels).activation(Activation.IDENTITY).build());
		MultiLayerConfiguration multiLayerConfiguration = listBuilder.backprop(true).pretrain(false).setInputType(InputType.convolutional(size, size, channels))
				.inputPreProcessor(6, new CnnToFeedForwardPreProcessor(500, 500, 1)).build();

		MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(multiLayerConfiguration);
		multiLayerNetwork.init();

		System.out.println(multiLayerNetwork.summary());

		UIServer uiServer = UIServer.getInstance();
		StatsStorage statsStorage = new InMemoryStatsStorage();
		uiServer.attach(statsStorage);
		multiLayerNetwork.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));

		return multiLayerNetwork;
	}

	public static void main(String[] args) {
		try {
			ImageEncoderDecoder.run();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
