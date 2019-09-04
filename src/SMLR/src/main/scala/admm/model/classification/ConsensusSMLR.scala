package admm.model.classification

import admm.optimizer.smlr.ConsensusSMLROptimizer
import org.apache.spark.rdd.RDD
import breeze.linalg.{*, argmax, DenseMatrix=>BDM, DenseVector => BDV}

/**
  * Created by Evan on 2018/8/13.
  */
class ConsensusSMLR(paraMap:Map[String,String]=Map()) {

	private val ConsensusSMLROptimizer:ConsensusSMLROptimizer = new ConsensusSMLROptimizer(paraMap)
	private var coef:BDM[Double] = _

	def fit(data:RDD[(Double,Array[Double])]):BDM[Double] = {
		coef = ConsensusSMLROptimizer.optimize(data)
		coef
	}
	def predict(data:RDD[Array[Double]]):BDV[Int]={
		val n_samples = data.count().toInt
		val A: BDM[Double] = BDM.horzcat(BDM(data.collect(): _*), BDM.ones[Double](n_samples, 1))

		val result = A*coef
		argmax(result(*,::)) + 1
	}
}
