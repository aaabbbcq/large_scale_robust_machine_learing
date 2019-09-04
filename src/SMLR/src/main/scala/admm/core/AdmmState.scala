package admm.core

import breeze.linalg.{DenseMatrix => BDM}

/**
  * Created by Evan on 2018/8/21.
  */
abstract class AdmmState(var x: BDM[Double], var u: BDM[Double] = BDM.zeros[Double](1, 1)) extends Serializable {
	var AiXi: BDM[Double] = _

	def xUpdate(para: BDM[Double]*): AdmmState = this

	def zUpdate(para: BDM[Double]*): AdmmState = this

	def uUpdate(para: BDM[Double]*): AdmmState = this
}
