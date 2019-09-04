package admm.optimizer.Lasso

import breeze.linalg.{inv, norm, DenseMatrix => BDM}
import org.apache.spark.rdd.RDD
import admm.core.{AdmmOptimizer, AdmmState}

/**
  * Created by Evan on 2018/8/6.
  */
private[admm] class SharingLassoOptimizer(paraMap: Map[String, String]) extends AdmmOptimizer(paraMap) {
	private val lassoParameter: Map[String, String] = Map(
		"lambda" -> "0.5"
		) ++ paraMap

	kind = "sharing"

	private val lamb: Double = lassoParameter("lambda").toDouble
	private val cons: Double = 1 / (numPartition + rho)

	def initialize(data: RDD[(Double, Array[Double])]): RDD[AdmmState] = {
		val n_samples:Int = data.count().toInt
		val states: RDD[AdmmState] = AdmmOptimizer.splitDataForSharing(data, numPartition).map(d => SharingLassoState(d.toArray, n_samples, rho, lamb))
		gAiXi = states.map(s => s.AiXi).reduce((m1, m2) => m1 + m2) / states.count().toDouble
		gZ = BDM.zeros(n_samples, 1)
		gU = BDM.zeros(n_samples, 1)
		targetMatrix = BDM(data.map(_._1).collect(): _*)
		states
	}

	def isConverged(newStates: RDD[AdmmState], status: Array[Array[BDM[Double]]]): Boolean = {
		val l = newStates.count()
		val primalResidual: Double = norm((gAiXi - gZ).toDenseVector, 2) * l
		val dualResidual: Double = newStates.map(_.x).collect().zip(status(0)).map(m => norm((m._1 - m._2).toDenseVector, 2)).sum
		if (primalResidual <= threshold && dualResidual <= threshold) true else false
	}

	def runRound(states: RDD[AdmmState]): RDD[AdmmState] = {
		var admmState = states
		admmState = states.map(_.xUpdate(gZ, gAiXi, gU)).cache()
		gAiXi = admmState.map(_.AiXi).reduce((m1, m2) => m1 + m2) / states.count().toDouble
		gZ = this.cons * (targetMatrix + rho * (gAiXi + gU))
		gU += gAiXi-gZ
		admmState
	}
}

private[admm] class SharingLassoState(initX: BDM[Double], val inver: BDM[Double], val Ai: BDM[Double], val lambDrho: Double)
  extends AdmmState(initX) {
	this.AiXi = this.Ai*this.x

	private def shrink(a: Double, b: Double) :Double = if (a > 0) a else if (b < 0) b else 0.0

	override def xUpdate(para: BDM[Double]*): AdmmState = {
		val (z_bar,aixi_bar,u_bar) = (para(0),para(1),para(2))
		var count = -1
		val a:BDM[Double] = this.Ai.t*(this.AiXi + z_bar - aixi_bar - u_bar)
		this.x = this.x.map(d=>{
			count+=1
			shrink(
				(inver(count,::)*(a-lambDrho)).inner.data(0),
				(inver(count,::)*(a+lambDrho)).inner.data(0)
			)})
		this.AiXi = Ai*this.x
		this
	}
}

private[admm] object SharingLassoState {
	def apply(data: Array[(Int, BDM[Double])], n_samples:Int, rho: Double, lamb: Double): SharingLassoState = {
		val Ai: BDM[Double] = data.map(_._2).reduce((m1: BDM[Double], m2: BDM[Double]) => BDM.horzcat(m1, m2)).t
//		println(Ai)
//		println("--------------------------------------------------------------------")
		val n_features: Int = Ai.cols
		val initXi: BDM[Double] = BDM.zeros(n_features, 1)
		val inver: BDM[Double] = inv(Ai.t * Ai)
		val lambDrho: Double = lamb / rho
		new SharingLassoState(initX = initXi, inver = inver, Ai = Ai, lambDrho = lambDrho)
	}
}