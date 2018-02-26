package org.utils
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.DataFrame
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.Row
import org.apache.hadoop.fs.Path
import org.apache.hadoop.fs.FileUtil
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.conf.Configuration


object SparkUtil {
  
  // Spark Configuration
  private lazy val conf = new SparkConf().setAppName("app").setMaster("local[*]")
  private lazy val sc:SparkContext = new SparkContext(conf)
  private lazy val sql:SparkSession = SparkSession.builder().config(sc.getConf).getOrCreate()
  
  // Spark Utils
  def rddToDf(rdd: RDD[Row], schema: StructType): DataFrame = sql.createDataFrame(rdd, schema)
  def readRDDText(path:String,minPartitions:Int = sc.defaultMinPartitions):RDD[String] = sc.textFile(path, minPartitions)
  def writeJSON(df:DataFrame,path:String,saveMode:String = "append"):Unit = df.write.mode(saveMode).json(path)
  def writeCSV(df:DataFrame,path:String,saveMode:String = "append"):Unit = df.write.mode(saveMode).options(Map("header"-> "true")).csv(path)
  
  def mergeFiles(inputPath: String, outputPath: String): Unit =  {
     val hadoopConfig = new Configuration()
     val hdfs = FileSystem.get(hadoopConfig)
     FileUtil.copyMerge(hdfs, new Path(inputPath), hdfs, new Path(outputPath), true, hadoopConfig, null) 
  }
  
  def writeCSVmerged(df:DataFrame,path:String): Unit = {
    writeCSV(df, "output/0.csv"  )
    mergeFiles( "output/0.csv" ,  path )
  }
  
  def writeJSONmerged(df:DataFrame,path:String): Unit = {
    writeJSON(df, "output/0.json" )
    mergeFiles( "output/0.json" ,  path )
  }  
 
  
}