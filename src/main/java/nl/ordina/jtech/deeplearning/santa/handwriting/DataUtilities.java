package nl.ordina.jtech.deeplearning.santa.handwriting;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

/**
 * Common data utility functions.
 * 
 * @author fvaleri
 */
public class DataUtilities {
	private static final String DATASETS_MNIST = "/datasets/mnist/";
	public static final int HEIGHT = 28, WIDTH = 28, CHANNELS = 1, BATCH_SIZE = 256, NUM_OUTPUTS = 10;
	private static final Random random = new Random(1337);	
	private static List<String> labels;
	
	public static DataSetIterator getTrainDataSetIterator() {
		try {
			File dataDir = loadResource("training");
			return makeIterator(dataDir);
		} catch (IOException ioe) {
			throw new NullPointerException("Specified traindata directory is null");
		}
	}
	
	public static DataSetIterator getTestDataSetIterator() {
		try {
			File dataDir = loadResource("testing");
			return makeIterator(dataDir);
		} catch (IOException ioe) {
			throw new NullPointerException("Specified testdata directory is null");
		}
	}
	
	public static List<String> getLabels() {
		if(null != labels) {
			return labels;
		} else {
			getTrainDataSetIterator();
			return getLabels();
		}
	}
	
	private static File loadResource(String subDir) {
		File output = null;
        try {
            output = new File(DataUtilities.class.getResource(DATASETS_MNIST + subDir).toURI());
        } catch (URISyntaxException e) {
            e.printStackTrace();
        }
        return output;
	}
	
	private static DataSetIterator makeIterator(File data) throws IOException {
		FileSplit fileSplit = new FileSplit(data, NativeImageLoader.ALLOWED_FORMATS, random);
	    ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // parent path as the image label
	    ImageRecordReader recordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
	    recordReader.initialize(fileSplit);
	    labels = recordReader.getLabels();
	    DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, BATCH_SIZE, 1, NUM_OUTPUTS);
	    
	    // pixel values from 0-255 to 0-1 (min-max scaling)
	    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
	    scaler.fit(iterator);
	    iterator.setPreProcessor(scaler);
	    
	    return iterator;
	}

}