package org.deeplearning4j.nlpexample;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.EvaluationBinary;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;

/**
 * Created by susaneraly on 7/11/19.
 */
public class ImbdExample {

    public static void main(String args[]) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        String modelPath = new ClassPathResource("nlpexample/fasttext/saved_model_bigram.h5").getFile().getPath();
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(modelPath, true);
        model.setListeners(new ScoreIterationListener(100));
        System.out.println(model.summary());
        System.out.println(model.conf().toJson());

        /*
        INDArray input = Nd4j.readNumpy(new ClassPathResource("nlpexample/fasttext/input_bigram.csv").getFile().getPath(),",");
        System.out.println(input);
        INDArray output = model.output(input);
        System.out.println(output );
        INDArray saveOutput = Nd4j.readNumpy(new ClassPathResource("nlpexample/fasttext/output_bigram.csv").getFile().getPath(),",");
        System.out.println(saveOutput);
        */

        DataSetIterator trainIter = new CustomDataSetIterator(true);
        DataSetIterator testIter = new CustomDataSetIterator(false);
        EvaluationBinary eval = new EvaluationBinary();
        int nContinueEpochs = 5;
        while (nContinueEpochs > 0) {
            if (nContinueEpochs == 5) {
                model.doEvaluation(testIter,eval);
                testIter.reset();
                System.out.println("Evaluation before any training " + nContinueEpochs + ":\n" + eval.stats());
                eval.reset();
            }
            model.fit(trainIter);
            trainIter.reset();
            model.doEvaluation(testIter,eval);
            testIter.reset();
            System.out.println("Evaluation at epoch " + nContinueEpochs + ":\n" + eval.stats());
            eval.reset();
            nContinueEpochs--;
        }
    }
}