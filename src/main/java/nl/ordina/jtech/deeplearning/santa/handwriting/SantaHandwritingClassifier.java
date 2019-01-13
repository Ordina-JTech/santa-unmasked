package nl.ordina.jtech.deeplearning.santa.handwriting;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.springframework.stereotype.Component;

import nl.ordina.jtech.deeplearning.santa.handwriting.model.Prediction;

@Component
public class SantaHandwritingClassifier {
	private MultiLayerNetwork network;
	private List<String> labels;
	
	public SantaHandwritingClassifier() {
		// TODO Assignment 1.5
	}

	List<Prediction> classify(InputStream inputStream) throws IOException {
		// TODO Assignment 1.7
		
		return null; //replace this line
	}
	
	private INDArray passThroughNetwork(final InputStream inputStream) throws IOException {
		// TODO Assignment 1.6
		return null; // replcae this line
	}
	
}

