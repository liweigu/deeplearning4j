package com.liweigu.dl.study;

import java.util.Random;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.google.gson.Gson;

/**
 * 
 * @author QQ:474483925(格物致知)
 *
 */
public class VariationalAutoencoderStudy {
	public static void main(String[] args) {
	       Random random = new Random();

	       MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	               .seed(-1)
	               .learningRate(1e-2)
	               .weightInit(WeightInit.XAVIER)
	               .activation(Activation.SIGMOID)
	               .regularization(true).l2(1e-4)
	               .updater(Updater.RMSPROP)
	               .list().layer(0, new VariationalAutoencoder.Builder()
	                       .activation(Activation.LEAKYRELU)
	                       .encoderLayerSizes(16, 8)
	                       .decoderLayerSizes(8, 16)
	                       .pzxActivationFunction(Activation.IDENTITY)  //p(z|data) activation function
	                       .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID.getActivationFunction()))     //Bernoulli distribution for p(data|z) (binary or 0 to 1 data only)
	                       .nIn(18)                       //Input size: 28x28
	                       .nOut(4)                            //Size of the latent variable space: p(z|x). 2 dimensions here for plotting, use more in general
	                       .build())
	               .pretrain(true).backprop(false)
	               .build();

	       MultiLayerNetwork net = new MultiLayerNetwork(conf);
	       net.init();

	       double[][] data = new double[2][19 - 1 + 2 - 3 + 1];
	       for (int j = 0; j < 2; j++){
	           for (int i = 0; i < 18; i++) {
	               data[j][i] = random.nextDouble();
	           }
	       }

	       INDArray trainData = Nd4j.create(data);

	       for (int i = 0; i < 20000; i++) {
	           net.fit(trainData);
	       }

	       org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net.getLayer(0);

	       System.out.println("原始数据1：" + new Gson().toJson(data[0]));
	       System.out.println("原始数据1被编码：" + vae.activate(trainData.getRow(0)));
	       System.out.println("原始数据1被解码：" + vae.generateAtMeanGivenZ(vae.activate(trainData.getRow(0))));

	       System.out.println("原始数据2：" + new Gson().toJson(data[1]));
	       System.out.println("原始数据22被编码：" + vae.activate(trainData.getRow(1)));
	       System.out.println("原始数据2被解码：" + vae.generateAtMeanGivenZ(vae.activate(trainData.getRow(1))));
	   }
}
