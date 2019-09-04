package admm.optimizer.smlr

import admm.core.{AdmmOptimizer, AdmmState}
import breeze.linalg.{*, norm, DenseMatrix => BDM, DenseVector => BDV, max => bmax, sum => bsum, Axis}
import breeze.numerics.{exp => bexp, sqrt}
import org.apache.spark.rdd.RDD

import math.{exp, max}
import scala.util.Random.shuffle

/**
  * Created by Evan on 2018/8/13.
  */
private[admm] class ConsensusSMLROptimizer(paraMap: Map[String, String]) extends AdmmOptimizer(paraMap) {
	private val smlrParameter: Map[String, String] = Map(
		"lambda" -> "0.5",
		"alpha" -> "0.001"
	) ++ paraMap

	kind = "consensus"

	private var n_features: Int = _
	private var n_class: Int = _
	private val alpha: Double = smlrParameter("alpha").toDouble
	private val lamb: Double = smlrParameter("lambda").toDouble
	private val k: Double = lamb / (numPartition * rho)

	private def shrinkage(a: Double): Double = max(0, a - k) - max(0, -1 * (a + k))

	def initialize(data: RDD[(Double, Array[Double])]): RDD[AdmmState] = {
		n_features = data.first()._2.length
		n_class = data.map(_._1.toInt).collect().distinct.length
//		n_features = 1024
//		n_class = 20
		val states: RDD[AdmmState] = AdmmOptimizer.splitDataForConsensus(data, numPartition).map(d => ConsensusSMLRState(d._2.toArray, rho, alpha, n_features, n_class))
		//val states: RDD[AdmmState] = AdmmOptimizer.experiment_split(data, numPartition).map(d => ConsensusSMLRState(d._2.toArray, rho, alpha, n_features, n_class))
		gZ = BDM.zeros[Double](n_features + 1, n_class)
		states
	}

	def runRound(states: RDD[AdmmState]): RDD[AdmmState] = {
		var admmStates = states
		admmStates = states.map(_.xUpdate(gZ)).cache()

//		val x_bar: BDM[Double] = admmStates.map(_.x).collect().reduce((m1, m2) => m1 + m2) / numPartition.toDouble
//		val u_bar: BDM[Double] = admmStates.map(_.u).collect().reduce((m1, m2) => m1 + m2) / numPartition.toDouble
//		gZ = (x_bar + u_bar).map(shrinkage)

//		val x_u_bar: BDM[Double] = admmStates.map(state => state.x + state.u).collect().reduce((m1, m2) => m1 + m2) / numPartition.toDouble
//		gZ = x_u_bar.map(shrinkage)

		val x_u_median: BDM[Double] = median(admmStates)
		//val u_bar: BDM[Double] = admmStates.map(_.u).collect().reduce((m1, m2) => m1 + m2) / numPartition.toDouble
		gZ = (x_u_median).map(shrinkage)

		admmStates.map(_.uUpdate(gZ))
	}

	def median(states: RDD[AdmmState]): BDM[Double] = {
		val medianMatrix: BDM[Double] = BDM(states.map(state => (state.x + state.u).toArray).collect(): _*)
		//val medianMatrix: BDM[Double] = BDM(states.map(_.x.toArray).collect(): _*)
		var x_median: BDV[Double] = BDV.zeros[Double]((n_features + 1) * n_class)
		val idx: Int = medianMatrix.rows.toInt / 2
		for (c <- 0 until medianMatrix.cols) {
			val vecSorted = medianMatrix(::, c).toArray.sorted
			x_median(c) = vecSorted(idx)
		}
		x_median.asDenseMatrix.reshape(n_features + 1, n_class)
	}

	override def getCurrentWeights(states: RDD[AdmmState]): BDM[Double] = gZ
}


private[admm] class ConsensusSMLRState(initX: BDM[Double], initU: BDM[Double], val A: BDM[Double], val labelCodeMatrix: BDM[Double], val rho: Double, val alpha: Double)
  extends AdmmState(initX, u = initU) {

	override def xUpdate(para: BDM[Double]*): AdmmState = {
		//println("adam optimezer~~")
		adam(para(0)); this
		//sgd(para(0)); this
	}

	override def uUpdate(para: BDM[Double]*): AdmmState = {
		this.u += this.x - para(0); this
	}

	def sgd(newZ: BDM[Double]): Unit = {
		var count = 0
		val rows = this.A.rows
		var iterSeries = shuffle((0 until rows).toList)
		var flag: Boolean = true
		var newX: BDM[Double] = this.x.copy
		while (flag) {
			if (count % rows == 0) {
				iterSeries = shuffle((0 until rows).toList)
				count = 0
			}
			var softmax: BDM[Double] = this.A(count, ::).inner.asDenseMatrix * this.x
			val max_base: BDV[Double] = bmax(softmax(*, ::))
			for (c <- 0 until softmax.cols) softmax(0, c) -= max_base(0)
			softmax = softmax.map(exp)
			val sum_base: BDV[Double] = bsum(softmax(*, ::))
			for (c <- 0 until softmax.cols) softmax(0, c) /= sum_base(0)
			val gradient: BDM[Double] = this.A(count, ::).t * (softmax - this.labelCodeMatrix(count, ::).inner.asDenseMatrix) + rho * (this.x - newZ + this.u)
			this.x -= alpha * gradient
			flag = if (norm((this.x - newX).toDenseVector, 2) <= 1e-3) false else true
			newX = this.x.copy
			count += 1
		}
	}

	def adam(newZ: BDM[Double]): Unit = {
		val (cols, rows, nbExapmles) = (newZ.cols, newZ.rows, this.A.rows)
		var mt: BDM[Double] = BDM.zeros(rows, cols) //momentum item
		var vt: BDM[Double] = BDM.zeros(rows, cols) //accumulation gradient square
		val (beta1, beta2, epsilon, maxIters) = (0.9, 0.999, 1e-8, 50) // default parameters of adam

		var isConverage: Boolean = false // conditions for stopping train
		var reachMaxItertimes: Boolean = true

		var t = 1
		while (reachMaxItertimes || isConverage) { //batch gradient decent with all examples
			val margin: BDM[Double] = this.A * this.x
			val max_margin = bmax(margin, Axis._1)
			val norm_margin: BDM[Double] = bexp(margin(::, *) - max_margin) // minus the max item of their rows for numerical stability
			val softmax = norm_margin(::, *) / bsum(norm_margin, Axis._1)
			val gradient = (this.A.t * (softmax - this.labelCodeMatrix)) + (rho * (this.x - newZ + this.u))

			mt = (mt * beta1) + (1 - beta1) * gradient
			vt = (vt * beta2) + (1 - beta2) * (gradient :* gradient)

			val alpha_t = alpha * math.pow(1 - math.pow(beta2, t), 0.5) / (1 - math.pow(beta1, t))
			this.x :-= (alpha_t * mt / (sqrt(vt) + epsilon))

			t += 1; if(t==maxIters) reachMaxItertimes = false
		}
	}

	def bgd(newZ: BDM[Double]): Unit = {
		var flag: Boolean = true
		var newX: BDM[Double] = this.x.copy
		while (flag) {
			var softmax: BDM[Double] = this.A * this.x
			val maxBase: Array[Double] = bmax(softmax(*, ::)).data
			for (r <- 0 until softmax.rows; c <- 0 until softmax.cols)
				softmax(r, c) -= maxBase(r)
			softmax = softmax.map(exp)
			val sumBase: Array[Double] = bsum(softmax(*, ::)).data
			for (r <- 0 until softmax.rows; c <- 0 until softmax.cols)
				softmax(r, c) /= sumBase(r)
			val gradient: BDM[Double] = this.A.t * (softmax - this.labelCodeMatrix) + rho * (this.x - newZ + this.u)
			this.x -= alpha * gradient
			flag = if (norm((this.x - newX).toDenseVector, 2) <= 1e-2) false else true
			newX = this.x.copy
		}
	}
}

private[admm] object ConsensusSMLRState {
	def apply(data: Array[(Double, Array[Double])], rho: Double, alpha: Double, n_features: Int, n_class: Int): ConsensusSMLRState = {
		val labels = data.map(_._1.toInt)
		val n_samples = labels.length
		val initX = BDM.zeros[Double](n_features + 1, n_class)
		val initU = BDM.zeros[Double](n_features + 1, n_class)
		val A: BDM[Double] = BDM.horzcat(BDM(data.map(_._2): _*), BDM.ones[Double](n_samples, 1))
		var labelCodeMatrix = BDM.zeros[Double](n_samples, n_class)
		for (i <- 0 until n_samples)
			labelCodeMatrix(i, labels(i) - 1) = 1.0
		new ConsensusSMLRState(initX = initX, initU = initU, A = A, labelCodeMatrix = labelCodeMatrix, rho = rho, alpha = alpha)
	}
}