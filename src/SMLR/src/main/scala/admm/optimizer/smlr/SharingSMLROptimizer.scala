package admm.optimizer.smlr

import org.apache.spark.rdd.RDD
import admm.core.{AdmmOptimizer, AdmmState}
import breeze.linalg.{*, Axis, inv, norm, DenseMatrix => BDM, DenseVector => BDV, max => bmax, sum => bsum}
import breeze.numerics.{sqrt, exp => bexp}

import scala.math.{exp, max}

/**
  * Created by Evan on 2018/8/18.
  */

private[admm] class SharingSMLROptimizer(paraMap: Map[String, String]) extends AdmmOptimizer(paraMap) {
	private val smlrParameter: Map[String, String] = Map(
		"lambda" -> "0.5",
		"alpha" -> "0.01"
		) ++ paraMap

	kind = "sharing"

	private val lamb: Double = smlrParameter("lambda").toDouble
	private val alpha: Double = smlrParameter("alpha").toDouble

	def initialize(data: RDD[(Double, Array[Double])]): RDD[AdmmState] = {
		val labels = data.map(_._1.toInt).collect()
		val n_samples = data.count().toInt
		val n_class = labels.distinct.length                                                  //RDD[Iterable[(Int, BDM[Double])]]
		val states: RDD[AdmmState] = AdmmOptimizer.splitDataForSharing(data, numPartition).map(d => SharingSMLRState(d.toArray, rho, lamb, alpha, n_class))
		gAiXi = states.map(s => s.AiXi).reduce((m1, m2) => m1 + m2) / states.count().toDouble
		gZ = BDM.zeros(n_samples, n_class)
		gU = BDM.zeros(n_samples, n_class)
		var labelCodeMatrix = BDM.zeros[Double](n_samples, n_class)
		for (i <- 0 until n_samples)
			labelCodeMatrix(i, labels(i) - 1) = 1.0
		targetMatrix = labelCodeMatrix
		states
	}

	def runRound(states: RDD[AdmmState]): RDD[AdmmState] = {
		var admmState = states
		admmState = states.map(_.xUpdate(gZ, gAiXi, gU)).cache()
		gAiXi = admmState.map(_.AiXi).reduce((m1, m2) => m1 + m2) / states.count().toDouble
		adam()
		gU += gAiXi - gZ
		//admmState = states.map(_.xUpdate(gZ, gAiXi, gU)).cache()
		admmState
	}

	def adam(): Unit = {
		val (rows, cols) = (gZ.rows, gZ.cols)
		var mt: BDM[Double] = BDM.zeros(rows, cols) //momentum item
		var vt: BDM[Double] = BDM.zeros(rows, cols) //accumulation gradient square
		val (beta1, beta2, epsilon, maxIters) = (0.9, 0.999, 1e-8, 200) // default parameters of adam

		var isConverage: Boolean = false // conditions for stopping train
		var reachMaxItertimes: Boolean = true

		var t = 1
		while (reachMaxItertimes || isConverage) { //batch gradient decent with all examples 未采用收敛条件，所以采用 或运算 ||
			val margin: BDM[Double] = this.numPartition.toDouble * gZ
			val max_margin = bmax(margin, Axis._1)
			val norm_margin: BDM[Double] = bexp(margin(::, *) - max_margin) // minus the max item of their rows for numerical stability
			val softmax = norm_margin(::, *) / bsum(norm_margin, Axis._1)
			val gradient = this.numPartition.toDouble * ((softmax - targetMatrix) + rho * (gZ - gAiXi - gU))

			mt = (mt * beta1) + (1 - beta1) * gradient
			vt = (vt * beta2) + (1 - beta2) * (gradient :* gradient)

			val alpha_t = alpha * math.pow(1 - math.pow(beta2, t), 0.5) / (1 - math.pow(beta1, t))
			gZ :-= (alpha_t * mt / (sqrt(vt) + epsilon))

			t += 1; if(t==maxIters) reachMaxItertimes = false
		}
	}
}


private[admm] class SharingSMLRState(initX: BDM[Double], val inver: BDM[Double], val Ai: BDM[Double], val rho: Double, val alpha: Double, val lambDrho: Double, val lamb:Double) //lambDrho 参数多余
  extends AdmmState(initX) {
	this.AiXi = this.Ai * this.x
//	private def shrink(a: Double, b: Double): Double = if (a > 0) a else if (b < 0) b else 0.0
//
//		override def xUpdate(para: BDM[Double]*): AdmmState = {
//			val (z_bar, aixi_bar, u_bar) = (para(0), para(1), para(2))
//			val a: BDM[Double] = this.AiXi + z_bar - aixi_bar - u_bar
//			for (c <- 0 until this.x.cols) {
//				val part: BDV[Double] = this.rho * this.Ai.t * a(::, c)
//				for(r <- 0 until this.x.rows) {
//					val newx: Double = shrink(
//						inver(r, ::) * (part - this.lambDrho),
//						inver(r, ::) * (part + this.lambDrho)
//					)
//					this.x.update(r, c, newx)
//				}
//			}
//			this.AiXi = Ai * this.x
//			this
//		}

	val (rho_in, iter_in) = (1.0, 50) //the default parameters of admm
	val I: BDM[Double] = BDM.eye[Double](Ai.cols)
	val cons: BDM[Double] = inv(rho * Ai.t * Ai + I * rho_in)
	val (rows, cols) = (this.x.rows,this.x.cols)

	val k = lamb / rho_in
	private def shrinkage(a: Double): Double = max(0, a - k) - max(0, -1 * (a + k))

	override def xUpdate(para: BDM[Double]*): AdmmState = { //updated by admm
		val (z_bar, aixi_bar, u_bar) = (para(0), para(1), para(2))
		var x_in: BDM[Double] = BDM.zeros[Double](rows, cols)
		var z_in: BDM[Double] = BDM.zeros[Double](rows, cols)
		var u_in: BDM[Double] = BDM.zeros[Double](rows, cols)
		var z_old = z_in.copy
		val t = Ai * this.x + z_bar - aixi_bar - u_bar

		var iter_count = 0
		var (reachMaxItertimes, isConverage) = (true, true)
		while (reachMaxItertimes && isConverage) {
			val right: BDM[Double] = rho * Ai.t * t + rho_in * (z_in - u_in)
			x_in = cons * right
			z_in = (x_in + u_in).map(shrinkage)
			u_in = u_in + x_in - z_in

			iter_count += 1;
			if (iter_count == iter_in) reachMaxItertimes = false
			if (norm((z_in - z_old).toDenseVector, 2) < 1e-3) isConverage = false
		}
		this.x = z_in
		this.AiXi = Ai * this.x
		this
	}
}

private[admm] object SharingSMLRState {
	def apply(data: Array[(Int, BDM[Double])], rho: Double, lamb: Double, alpha: Double, n_class: Int): SharingSMLRState = {
		val Ai: BDM[Double] = data.map(_._2).reduce((m1: BDM[Double], m2: BDM[Double]) => BDM.horzcat(m1, m2)).t
		val n_features: Int = Ai.cols
		val initXi: BDM[Double] = BDM.zeros(n_features, n_class)
		val inver: BDM[Double] = inv(Ai.t * Ai)
		val lambDrho: Double = lamb / rho
		new SharingSMLRState(initX = initXi, inver = inver, Ai = Ai, rho = rho, alpha = alpha, lambDrho = lambDrho, lamb=lamb)
	}
}
