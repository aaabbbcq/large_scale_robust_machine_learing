package admm.model.classification

import breeze.linalg.{*, argmax, DenseMatrix => BDM, DenseVector=>BDV}
import admm.optimizer.smlr.SharingSMLROptimizer
import org.apache.spark.rdd.RDD

/**
  * Created by Evan on 2018/8/19.
  */
class SharingSMLR(paraMap:Map[String,String]=Map()) {
	private val SharingSMLROptimizer:SharingSMLROptimizer = new SharingSMLROptimizer(paraMap)
	private var coef:BDM[Double] = _
	def fit(data:RDD[(Double,Array[Double])]):BDM[Double] = {
		coef = SharingSMLROptimizer.optimize(data)
		coef
	}
	def predict(data:RDD[Array[Double]]):BDV[Int]={
		val n_samples = data.count().toInt
		//val A: BDM[Double] = BDM.horzcat(BDM(data.collect(): _*), BDM.ones[Double](n_samples, 1))
		val A: BDM[Double] = BDM(data.collect(): _*)
		val result = A*coef
		(argmax(result(*,::)) + 1)
	}
}

