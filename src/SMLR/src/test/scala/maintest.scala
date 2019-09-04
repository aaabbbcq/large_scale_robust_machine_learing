/**
  * Created by Evan on 2018/10/24.
  */

import breeze.linalg.{*, norm, DenseMatrix => BDM, DenseVector => BDV, max => bmax, sum => bsum, softmax,Axis}

object maintest {

	//implicit def int2Double(x: Int):Double = x.toDouble

	def main(args: Array[String]): Unit = {
		//val c:BDM[Double] = BDM.eye(5)
		val margin:BDM[Double] = BDM(Array(Array[Double](1,2,3),Array[Double](2,3,4)):_*)

		print(Array[Double](1,2,3).init)
		//val maxMargin = bmax(margin,Axis._1)

		//val normedMargin = margin(::,*) - maxMargin
		//val sunMargin = bsum(normedMargin,Axis._1)
		//println(margin)
		//println(margin :* margin)
//		println(margin)
//		println("---")
//		println(maxMargin)
//		println("----")
//		println(normedMargin)
//		println("---")
//		println(sunMargin)
//		println("---")
//		println(normedMargin(::,*) / sunMargin)
		//println(bmax(c,Axis._0))
		//val norm_margin: BDM[Double] = bexp(margin(::, *) - max_margin)

	}
}
