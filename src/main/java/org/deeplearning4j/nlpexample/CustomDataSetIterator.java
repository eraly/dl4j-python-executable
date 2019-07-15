package org.deeplearning4j.nlpexample;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;
import java.util.List;

/**
 * Created by susaneraly on 7/15/19.
 */
public class CustomDataSetIterator implements DataSetIterator {
    private static final int NUMFILES = 781;
    private String trainOrTest = "train/";
    private int currentBatch = 0;

    public CustomDataSetIterator() {
        new CustomDataSetIterator(true);
    }

    public CustomDataSetIterator(boolean train) {
        if (!train) trainOrTest = "test/";
    }

    @Override
    public DataSet next(int i) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public int inputColumns() {
        return 0;
    }

    @Override
    public int totalOutcomes() {
        return 0;
    }

    @Override
    public boolean resetSupported() {
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {
        currentBatch = 0;
    }

    @Override
    public int batch() {
        return currentBatch;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {

    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        if (currentBatch <= NUMFILES) return true;
        return false;
    }

    @Override
    public DataSet next() {
        try {
            INDArray features = Nd4j.readNumpy(new ClassPathResource(constructedPath("data",currentBatch)).getFile().getPath(), ",");
            INDArray label = Nd4j.readNumpy(new ClassPathResource(constructedPath("label",currentBatch)).getFile().getPath(), ",");
            currentBatch++;
            return new DataSet(features,label);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    private String constructedPath(String dataOrLabel, int fileNum) {
        String baseDir = "nlpexample/fasttext/";
        return baseDir + trainOrTest + dataOrLabel + "_" + fileNum + ".csv";
    }
}
