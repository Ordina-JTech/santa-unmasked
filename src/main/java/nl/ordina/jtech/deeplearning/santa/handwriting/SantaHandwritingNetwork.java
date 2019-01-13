package nl.ordina.jtech.deeplearning.santa.handwriting;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator.Set;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

public class SantaHandwritingNetwork {
	private static final Logger log = org.slf4j.LoggerFactory.getLogger(SantaHandwritingNetwork.class);

	public static final int SEED = 123;
	public static final Set EMNIST_SET = EmnistDataSetIterator.Set.MNIST;
	public static final int NUM_CLASSES = EmnistDataSetIterator.numLabels(EMNIST_SET);
	public static final List<String> LABELS = EmnistDataSetIterator.getLabels(EMNIST_SET);

	public static final int BATCH_SIZE = 256;
	public static final int HEIGHT = 28;
	public static final int WIDTH = 28;
	public static final int CHANNELS = 1;
	public static EmnistDataSetIterator emnistTrain, emnistTest;
	public MultiLayerNetwork network;
	
	public SantaHandwritingNetwork() {}
	
	public void setup() throws IOException {
		// TODO assignment 1.1

		
		initializeNetwork();
	}
	
	public void initializeNetwork() {
		// TODO Assignment 1.2a -> 1.2d
		
	}
	
	public void train() throws IOException {
		// TODO Assignment 1.3
		network.init();
		
		
		// TODO Assignment 1.4
		
	}
	
	public static void main(String args[]) {
		try {
			SantaHandwritingNetwork network = new SantaHandwritingNetwork();
			network.setup();
			network.train();
		} catch (IOException ioe) {
			ioe.printStackTrace();
		}
	}
	
}
