package admm.model.regression

import admm.optimizer.Lasso.SharingLassoOptimizer
import breeze.linalg.{DenseMatrix => BDM}
import org.apache.spark.rdd.RDD

/**
  * Created by Evan on 2018/8/8.
  */

class SharingLasso(paraMap:Map[String,String]=Map()) {
	private val lassoOptimizer:SharingLassoOptimizer = new SharingLassoOptimizer(paraMap)
	def fit(data:RDD[(Double,Array[Double])]):BDM[Double] = lassoOptimizer.optimize(data)
	def predict(data:RDD[(Double,Array[Double])]):BDM[Double]={
		???
	}
}