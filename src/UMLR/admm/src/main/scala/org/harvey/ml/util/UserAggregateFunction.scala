package org.harvey.ml.util

import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types.{ArrayType, DataType, StructField, StructType}

import scala.collection.mutable

class UserAggregateFunction extends UserDefinedAggregateFunction {
  // This is the input fields for your aggregate function.
  override def inputSchema: org.apache.spark.sql.types.StructType = StructType(
    StructField("value", VectorType) :: Nil
  )

  // This is the internal fields you keep for computing your aggregate.
  override def bufferSchema: StructType = StructType(
    StructField("count", ArrayType(VectorType)) :: Nil
  )

  // This is the output type of your aggregatation function.
  override def dataType: DataType = ArrayType(VectorType)

  override def deterministic: Boolean = true

  // This is the initial value for your buffer schema.
  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer(0) = Array[Vector]()
  }

  // This is how to update your buffer schema given an input.
  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    buffer(0) = buffer.getAs[mutable.WrappedArray[Vector]](0).+:(input.getAs[mutable.WrappedArray[Vector]](0))
    //    println(buffer.getAs[mutable.WrappedArray[Double]](0))
  }

  // This is how to merge two objects with the bufferSchema type.
  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    //    println("1 " + buffer1(0))
    //    println("2 " + buffer2(0))
    buffer1(0) = buffer1.getAs[mutable.WrappedArray[Vector]](0).++(buffer2.getAs[mutable.WrappedArray[Vector]](0))
    println(buffer1.getAs[mutable.WrappedArray[Vector]](0).zipWithIndex)
    //    buffer1(0) = buffer1.getAs[mutable.WrappedArray[Vector]](0)
    //    println(buffer2(0))
  }

  // This is where you output the final value, given the final value of your bufferSchema.
  override def evaluate(buffer: Row): Any = {
    buffer.getAs[mutable.WrappedArray[Vector]](0)
    //    println(buffer)
    //    1.0
  }
}