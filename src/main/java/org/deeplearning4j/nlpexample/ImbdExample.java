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

        DataSetIterator trainIter = new CustomDataSetIterator(true);
        DataSetIterator testIter = new CustomDataSetIterator(false);
        EvaluationBinary eval = new EvaluationBinary();
        int nContinueEpochs = 15;
        int epoch = 0;
        while (epoch < nContinueEpochs) {
            if (epoch == 0) {
                model.doEvaluation(testIter,eval);
                testIter.reset();
                System.out.println("Evaluation before any training " + nContinueEpochs + ":\n" + eval.stats());
                eval.reset();
            }
            model.fit(trainIter);
            trainIter.reset();
            model.doEvaluation(testIter,eval);
            testIter.reset();
            System.out.println("Evaluation at epoch " + epoch + ":\n" + eval.stats());
            eval.reset();
            epoch--;
        }
    }
}
