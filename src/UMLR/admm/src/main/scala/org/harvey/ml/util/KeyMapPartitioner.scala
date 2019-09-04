package org.harvey.ml.util

import org.apache.spark.Partitioner

class KeyMapPartitioner(partitions: Int) extends Partitioner{

  require(partitions >= 0, s"Number of partitions ($partitions) cannot be negative.")

  override def numPartitions: Int = partitions

  override def getPartition(key: Any): Int = key match {
    case null => 0
    case _ => key.asInstanceOf[Int] % numPartitions
  }

  override def equals(other: Any): Boolean = other match {
    case h: KeyMapPartitioner =>
      h.numPartitions == numPartitions
    case _ =>
      false
  }

  override def hashCode: Int = numPartitions
}
