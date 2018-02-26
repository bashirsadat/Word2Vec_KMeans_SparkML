package org.mining
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StringType
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.clustering.KMeans
import org.utils.SparkUtil

object TextClustering {
    
  def main(args: Array[String]) {    
    
    //############ Get Dataframe of "sentiments" 
    
    // get txt file of "sentiments" as Rdd of rows
    val inputPath = "resources/sentiments.txt"
    val rddData= SparkUtil.readRDDText(inputPath).map { x => Row(x) }
    
    // set the schema of dataframe
    val schema =
      StructType(
        Array(
          StructField("Sentiment", StringType, nullable=false)
        )
      )

    // get Dataframe of "sentiments"
    val dfData = SparkUtil.rddToDf(rddData, schema)
    
    
    //############ Tokenizer Process
    
    // create tokenizer object and set parameters
    val tokenizer = new Tokenizer().setInputCol("Sentiment").setOutputCol("words_tk")
    
    // apply tokenizer and create new tokenized dataframe
    val tokenized = tokenizer.transform(dfData)
    // apply udf of countTokens
    val dfTokenized = tokenized.withColumn("tokens", countTokens(col("words_tk")))
    
    //############ StopWordsRemover Process

    // create remover object and set parameters
    val remover = new StopWordsRemover()
                  .setInputCol("words_tk")
                  .setOutputCol("words_filtered")
    // apply remover and create new removed dataframe
    val dfRemoved = remover.transform(dfTokenized)
    
    
    //############ Second StopWordsRemover Process (my own criteria of removement)

    // apply udf of second removement and join all the final tokens in a string
    val dfSecondRemoved = dfRemoved.withColumn("words_tk_relevant", filterArrayElem(dfRemoved("words_filtered")))
                            .withColumn("words_tk_relevant_joined", filterJoinArrayElem(dfRemoved("words_filtered")))                    
    
                            
    //############ Word2Vec Process
    
    // create word2vec object and set parameters                        
    val word2vec = new Word2Vec()
      .setInputCol("words_tk_relevant")
      .setOutputCol("result_w2v")
      .setVectorSize(5) // a word will be represented as vector of 5 dimensions
      .setMinCount(0)
      
    // fit word2vec object to dataframe for create a word2vec model
    val model_word2vec = word2vec.fit(dfSecondRemoved)
    // apply word2vec model to dataframe
    val dfWord2Vec = model_word2vec.transform(dfSecondRemoved)

    
    //############ Clustering Process
    
    // create KMeans object and set parameters
    val kmeans = new KMeans()
          .setK(400) // get 400 clusters
          .setSeed(1L)
          .setFeaturesCol("result_w2v")
          .setPredictionCol("predicted_cluster")
    // fit KMeans object to dataframe for create a KMeans model
    val model_kmeans = kmeans.fit(dfWord2Vec)
    
    // apply KMeans model to dataframe
    val dfClustered = model_kmeans.transform(dfWord2Vec)
    
    
    //############ Save Result Process
    
    val outputPath = "output/"
    // save files
    SparkUtil.writeCSVmerged(dfClustered.select("Sentiment","words_tk_relevant_joined" , "predicted_cluster") ,
                              outputPath + "Sentiments_Clustered.csv"  )
    SparkUtil.writeJSONmerged(dfClustered , 
                              outputPath + "Sentiments_Clustered.json")
    
    
   }
  
  // udf to count tokens
  val countTokens = udf { (words: Seq[String]) => words.length }
  
  // udf to remove tokens of length < = 2
  val filterArrayElem = udf((array : Seq[String]) => {
                             // val x
                             array.filter { x => x.length() > 2 }
                          })
  
  // udf to remove tokens of length < = 2 and join them in a string
  val filterJoinArrayElem = udf((array : Seq[String]) => {
                             // val x
                             array.filter { x => x.length() > 2 }.mkString(" ")
                          })
  
  
}