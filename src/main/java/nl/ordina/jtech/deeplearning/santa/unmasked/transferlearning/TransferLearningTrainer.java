package nl.ordina.jtech.deeplearning.santa.unmasked.transferlearning;

import static org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;

import java.io.File;
import java.io.IOException;

import nl.ordina.jtech.deeplearning.santa.unmasked.transferlearning.dataHelpers.SantaDataSetIterator;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.slf4j.Logger;

/**
 * Based on code by susaneraly on 3/9/17.
 */
public class TransferLearningTrainer {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(TransferLearningTrainer.class);

    private static final int NUM_CLASSES = 2; //Santa / No Santa
    private static final long SEED = 12345; //Should normally be random, but is set in this case to be get reproducible results

    private static final int TRAIN_PERC = 80; // Percentage of images that should be included in the trainings set, the rest is included in the test set
    private static final int BATCH_SIZE = 12;

    public static void main(String[] args) throws IOException {

        log.info("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");

        //TODO: Assignment 2.2: Load VGG16 Network
        ComputationGraph vgg16 = null;//Replace this line


        //Shows the
        log.info(vgg16.summary());

        //TODO: Assignemnt 2.3: Create outputlayer



        //TODO: Assignemnt 2.4: Include the new outputlayer in the model
        ComputationGraph vgg16Transfer = null; //Replace this line

        log.info(vgg16Transfer.summary());

        //TODO: Assignment 2.5: Create test and train iterator


        //TODO: Assignent 2.6: Evaluate the network



        //TODO: Assignment 2.7: Train the network


        log.info("Model build complete");

        //TODO: Assignment 2.9: Store the trained network


        log.info("Model saved");
    }

    @NotNull
    private static NormalDistribution getDist() {
        return new NormalDistribution(0, 0.2 * (2.0 / (4096 + NUM_CLASSES)));
    }

    private static FineTuneConfiguration getFineTuneConfiguration() {
        return new FineTuneConfiguration.Builder()//
                .updater(new Nesterovs(5e-5))//
                .seed(SEED).build();
    }
}
