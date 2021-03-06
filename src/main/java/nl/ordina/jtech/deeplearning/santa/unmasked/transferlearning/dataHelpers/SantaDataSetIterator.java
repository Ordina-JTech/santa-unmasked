package nl.ordina.jtech.deeplearning.santa.unmasked.transferlearning.dataHelpers;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

public class SantaDataSetIterator {
    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static final Random rng  = new Random(1663);

    private static final int HEIGHT = 224;
    private static final int WIDTH = 224;
    private static final int CHANNELS = 3;
    private static final int NUM_CLASSES = 2;
    private static final String DATASETS_SANTA = "/datasets/santa";

    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static InputSplit trainData,testData;
    private static int batchSize;

    public static DataSetIterator trainIterator() throws IOException {
        return makeIterator(trainData);

    }

    public static DataSetIterator testIterator() throws IOException {
        return makeIterator(testData);

    }

    public static List<String> getLabels(){
        return Arrays.asList( "no_santa_hat", "santa_hat");
    }

    public static void setup(int batchSizeArg, int trainPerc) {
        batchSize = batchSizeArg;
        File parentDir = null;
        try {
            parentDir = new File(SantaDataSetIterator.class.getResource(DATASETS_SANTA).toURI());
        } catch (URISyntaxException e) {
            e.printStackTrace();
        }

        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        if (trainPerc >= 100) {
            throw new IllegalArgumentException("Percentage of data set aside for training has to be less than 100%. Test percentage = 100 - training percentage, has to be greater than 0");
        }
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, trainPerc, 100-trainPerc);
        trainData = filesInDirSplit[0];
        testData = filesInDirSplit[1];
    }

    private static DataSetIterator makeIterator(InputSplit split) throws IOException {
        DataSetIterator iter;
        try (ImageRecordReader recordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker)) {
            recordReader.initialize(split);
            iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, NUM_CLASSES);
        }
        iter.setPreProcessor( new VGG16ImagePreProcessor());
        return iter;
    }

}
