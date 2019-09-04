package admm.model.regression

import admm.optimizer.Lasso.ConsensusLassoOptimizer
import breeze.linalg.{DenseMatrix => BDM}
import org.apache.spark.rdd.RDD

/**
  * Created by Evan on 2018/8/6.
  */
class ConsensusLasso(paraMap:Map[String,String]=Map()) {
	private val lassoOptimizer:ConsensusLassoOptimizer = new ConsensusLassoOptimizer(paraMap)
	def fit(data:RDD[(Double,Array[Double])]):BDM[Double] = lassoOptimizer.optimize(data)
	def predict(data:RDD[(Double,Array[Double])]):BDM[Double]={
		???
	}
}
