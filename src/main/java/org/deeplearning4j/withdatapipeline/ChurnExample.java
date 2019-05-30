package org.deeplearning4j.withdatapipeline;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.StringColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.normalize.Normalize;
import org.datavec.api.transform.ui.HtmlAnalysis;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.AnalyzeSpark;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.EvaluationBinary;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

/**
 * Created by susaneraly on 5/30/19.
 */
public class ChurnExample {

    @Parameter(names = {"--newDataPath"}, description = "Path/directory to more training data", required = true)
    private String newDataPath = null;

    @Parameter(names = {"--newDataFeaturizedPath"}, description = "Path/directory to save new training data after featurization", required = true)
    private String newDataFeaturizedPath = null;

    @Parameter(names = {"--pathToSaveModel"}, description = "Path/directory to save new training data after featurization", required = true)
    private String pathToSaveModel = null;

    public static void main(String... args) throws Exception {
        new ChurnExample().entryPoint(args);
    }

    /**
     * JCommander entry point
     *
     * @param args
     * @throws Exception
     */
    protected void entryPoint(String[] args) throws Exception {

        //Handle command line arguments
        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            throw e;
        }

        //=====================================================================
        //                 Step 1: Clean dataset and featurize
        //=====================================================================

        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.setAppName("End to End Example with DataVec");

        JavaSparkContext sc = new JavaSparkContext(conf);
        JavaRDD<String> stringData = sc.textFile(newDataPath);

        RecordReader rr = new CSVRecordReader();
        JavaRDD<List<Writable>> parsedInputData = stringData.map(new StringToWritablesFunction(rr));

        //Let's define the schema of the data that we want to import
        //The order in which columns are defined here should match the order in which they appear in the input data
        //RowNumber,CustomerId,Surname,CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited
        Schema inputDataSchema = new Schema.Builder()
                //We will drop these first three cols so doesn't matter that we treat them as strings
                .addColumnsString("RowNumber", "CustomerID", "Surname")
                .addColumnInteger("CreditScore")
                .addColumnCategorical("Geography", Arrays.asList("France", "Germany", "Spain"))
                .addColumnCategorical("Gender", Arrays.asList("Female", "Male"))
                .addColumnsInteger("Age", "Tenure")
                //Some columns have restrictions on the allowable values, that we consider valid:
                .addColumnDouble("Balance", 0.0, null, false, false)   //$0.0 or more, no maximum limit, no NaN and no Infinite values
                .addColumnsInteger("NumOfProducts", "HasCrCard", "IsActiveMember")
                .addColumnDouble("EstimatedSalary", 0.0, null, false, false)   //$0.0 or more, no maximum limit, no NaN and no Infinite values
                .addColumnInteger("Exited")
                .build();

        //Print out the schema:
        System.out.println("Input data schema details:");
        System.out.println(inputDataSchema);

        TransformProcess tpClean = new TransformProcess.Builder(inputDataSchema)
                .filter(new ConditionFilter(new StringColumnCondition("RowNumber", ConditionOp.Equal, "RowNumber")))
                .removeColumns("RowNumber", "CustomerID", "Surname")
                .categoricalToOneHot("Geography")
                .categoricalToInteger("Gender")
                .build();
        JavaRDD<List<Writable>> cleanData = SparkTransformExecutor.execute(parsedInputData, tpClean);

        int maxHistogramBuckets = 10;
        DataAnalysis dataAnalysis = AnalyzeSpark.analyze(tpClean.getFinalSchema(), cleanData, maxHistogramBuckets);
        HtmlAnalysis.createHtmlAnalysisFile(dataAnalysis, new File("DataVecAnalysis.html"));


        TransformProcess.Builder tpFeaturizeBuilder = new TransformProcess.Builder(tpClean.getFinalSchema());
        //.normalize("CreditScore", Normalize.Standardize, dataAnalysis)

        for (String aCol : tpClean.getFinalSchema().getColumnNames()) {
            if (!aCol.equals("Exited")) tpFeaturizeBuilder.normalize(aCol, Normalize.Standardize, dataAnalysis);
        }
        TransformProcess tpFeaturize = tpFeaturizeBuilder.build();
        System.out.println(tpFeaturize.toJson());
        JavaRDD<List<Writable>> featurizedData = SparkTransformExecutor.execute(cleanData, tpFeaturize);
        List<String> featurizedCSV = featurizedData.map(new WritablesToStringFunction(",")).collect();
        Files.write(Paths.get(newDataFeaturizedPath), featurizedCSV, Charset.defaultCharset());

        //=====================================================================
        //                 Step 2: Import Keras Model
        //=====================================================================
        String modelPath = new ClassPathResource("datapipeline/churn/churn.h5").getFile().getPath();
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(modelPath, true);
        model.setListeners(new ScoreIterationListener(1000));
        System.out.println(model.summary());
        System.out.println(model.conf().toJson());

        //=====================================================================
        //                 Step 3: Setup dataset from clean data
        //=====================================================================
        int labelIndex = 12;
        int batchSizeTraining = 10;
        DataSetIterator trainingDataIter = makeCSVDataSetIterator(
                newDataFeaturizedPath,
                batchSizeTraining, labelIndex);

        //=====================================================================
        //                 Step 4: Continue training
        //=====================================================================
        EvaluationBinary eval = new EvaluationBinary();
        model.doEvaluation(trainingDataIter, eval);
        System.out.println(eval.stats());
        eval.reset();
        int nEpochs = 25;
        for (int epoch = 0; epoch < nEpochs; epoch++) {
            model.fit(trainingDataIter);
            System.out.printf("======== EPOCH %d complete ==========\n", epoch);
        }
        System.out.println("======== COMPLETE ==========");
        model.doEvaluation(trainingDataIter, eval);
        System.out.println(eval.stats());

        //=====================================================================
        //                 Step 5: Save Model
        //=====================================================================
        model.save(new File(pathToSaveModel),true);

    }

    private static DataSetIterator makeCSVDataSetIterator(
            String csvFileClasspath, int batchSize, int labelIndex)
            throws IOException, InterruptedException {

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(csvFileClasspath)));
        int numClasses = 1;
        return new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);
    }

    private static DataSetIterator makeCSVDataSetIterator(
            String csvFileClasspath, int batchSize, int labelIndexStart, int labelIndexStop)
            throws IOException, InterruptedException {

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(csvFileClasspath)));
        return new RecordReaderDataSetIterator(rr, batchSize, labelIndexStart, labelIndexStop, true);
    }
}
