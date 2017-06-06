package DL

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.nd4j.linalg.io.ClassPathResource
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.apache.spark.sql.execution.datasources.RecordReaderIterator
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import java.io.File
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.SplitTestAndTrain
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.scalnet.models.NeuralNet
import org.deeplearning4j.scalnet.layers.Dense
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.scalnet.regularizers.WeightRegularizer
import org.deeplearning4j.scalnet.regularizers.L2
import org.deeplearning4j.scalnet.layers.DenseOutput
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.deeplearning4j.scalnet.optimizers.SGD
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.eval.Evaluation
import org.nd4j.linalg.api.ndarray.INDArray

object Iris {

    private val rngseed: Int = 123
    private val learning_rate: Double = 0.00015
    private val mometum: Double = 0.98
    private val batch_size: Int = 64
    private val n_rows: Int = 28
    private val n_cols: Int = 28
    private val n_hidden_1: Int = 512
    private val n_hidden_2: Int = 128
    private val n_out: Int = 10
    private val lambda: Double = 0.0005
    private val n_epochs: Int = 15

    def main(agrs: Array[String]) {
        val recordIris = new CSVRecordReader(1, ",")
        recordIris.initialize(new FileSplit(new File("C://Users//cuongdm30//Downloads//train.csv")))
        val iterator: DataSetIterator = new RecordReaderDataSetIterator(recordIris, 42000, 0, 10)
        val data: DataSet = iterator.next()
        data.shuffle()

        /*Chia dữ liệu thành 2 phần: train(80%), test(20%)*/
        val testAndTrain: SplitTestAndTrain = data.splitTestAndTrain(0.80)
        val test_data = testAndTrain.getTest
        val train_ = testAndTrain.getTrain.asList()
        val train_data = new ListDataSetIterator(train_, train_.size)

        /*Xây dựng mô hình*/
        println("Build Model...")
        val model = new NeuralNet()
        model.add(new Dense(nOut = n_hidden_1, nIn = n_rows * n_cols, weightInit = WeightInit.XAVIER, activation = "relu", regularizer = L2(lambda)))
        model.add(new Dense(nOut = n_hidden_2, nIn = n_hidden_1, weightInit = WeightInit.XAVIER, activation = "relu", regularizer = L2(lambda)))
        model.add(new DenseOutput(nOut = n_out, weightInit = WeightInit.XAVIER, lossFunction = LossFunction.NEGATIVELOGLIKELIHOOD, activation = "relu", regularizer = L2(lambda)))
        model.compile(optimizer = new SGD(learning_rate, mometum, true))
        println("Done.")
        
        /*Huấn luyện*/
        println("Training...")
        model.fit(train_data, nbEpoch = n_epochs, List(new ScoreIterationListener(5)))
        println("Done.")
        
        /*Đánh giá mô hình*/
        println("Evaluating..")
        val eval: Evaluation = new Evaluation(n_out)
        val output : INDArray = model.predict(test_data.getFeatureMatrix)
        eval.eval(test_data.getLabels, output)
        println(eval.stats())
        
    }
}