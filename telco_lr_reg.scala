import org.apache.spark.ml.classification.{GBTClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, regexp_extract, regexp_replace, when}
import org.apache.spark.sql.types.DecimalType
object telco_lr_reg {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .master("local[*]")
      .config("spark.executor.memory", "70g")
      .config("spark.driver.memory", "50g")
      .config("spark.memory.offHeap.enabled", true)
      .config("spark.memory.offHeap.size", "16g")
      .config("spark.kryoserializer.buffer.max", "400M")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .appName("Telecom")
      .getOrCreate()
//• Read the CSV file into spark Dataframe to perform dataframe related operations.

    spark.sparkContext.setLogLevel("WARN")
    val dataspath = "D:\\data_files\\Telco-Customer-Churn.csv"

    // Keep the actual header and schema.
    val telco_DF = spark.read.option("header", "true")
      .option("inferSchema", true)
      .option("mode", "DROPMALFORMED")
      .csv(dataspath)

//Print the schema of the data in tree format.
  println("telco schema")
    telco_DF.printSchema()
/*
root
 |-- customerID: string (nullable = true)
 |-- gender: string (nullable = true)
 |-- SeniorCitizen: integer (nullable = true)
 |-- Partner: string (nullable = true)
 |-- Dependents: string (nullable = true)
 |-- tenure: integer (nullable = true)
 |-- PhoneService: string (nullable = true)
 |-- MultipleLines: string (nullable = true)
 |-- InternetService: string (nullable = true)
 |-- OnlineSecurity: string (nullable = true)
 |-- OnlineBackup: string (nullable = true)
 |-- DeviceProtection: string (nullable = true)
 |-- TechSupport: string (nullable = true)
 |-- StreamingTV: string (nullable = true)
 |-- StreamingMovies: string (nullable = true)
 |-- Contract: string (nullable = true)
 |-- PaperlessBilling: string (nullable = true)
 |-- PaymentMethod: string (nullable = true)
 |-- MonthlyCharges: double (nullable = true)
 |-- TotalCharges: string (nullable = true)
 |-- Churn: string (nullable = true)

 */
//Print the top 5 rows of the dataframe to get some initial understanding of the data.

   println("telco_DF show upto 5")
    telco_DF.createOrReplaceTempView("main_view")
    telco_DF.show(5,false)

/*
+----------+------+-------------+-------+----------+------+------------+----------------+---------------+--------------+------------+----------------+-----------+-----------+---------------+--------------+----------------+-------------------------+--------------+------------+-----+
|customerID|gender|SeniorCitizen|Partner|Dependents|tenure|PhoneService|MultipleLines   |InternetService|OnlineSecurity|OnlineBackup|DeviceProtection|TechSupport|StreamingTV|StreamingMovies|Contract      |PaperlessBilling|PaymentMethod            |MonthlyCharges|TotalCharges|Churn|
+----------+------+-------------+-------+----------+------+------------+----------------+---------------+--------------+------------+----------------+-----------+-----------+---------------+--------------+----------------+-------------------------+--------------+------------+-----+
|7590-VHVEG|Female|0            |Yes    |No        |1     |No          |No phone service|DSL            |No            |Yes         |No              |No         |No         |No             |Month-to-month|Yes             |Electronic check         |29.85         |29.85       |No   |
|5575-GNVDE|Male  |0            |No     |No        |34    |Yes         |No              |DSL            |Yes           |No          |Yes             |No         |No         |No             |One year      |No              |Mailed check             |56.95         |1889.5      |No   |
|3668-QPYBK|Male  |0            |No     |No        |2     |Yes         |No              |DSL            |Yes           |Yes         |No              |No         |No         |No             |Month-to-month|Yes             |Mailed check             |53.85         |108.15      |Yes  |
|7795-CFOCW|Male  |0            |No     |No        |45    |No          |No phone service|DSL            |Yes           |No          |Yes             |Yes        |No         |No             |One year      |No              |Bank transfer (automatic)|42.3          |1840.75     |No   |
|9237-HQITU|Female|0            |No     |No        |2     |Yes         |No              |Fiber optic    |No            |No          |No              |No         |No         |No             |Month-to-month|Yes             |Electronic check         |70.7          |151.65      |Yes  |
+----------+------+-------------+-------+----------+------+------------+----------------+---------------+--------------+------------+----------------+-----------+-----------+---------------+--------------+----------------+-------------------------+--------------+------------+-----+
 */
    println("first part of main data base")
    spark.sql("select customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService," +
      "OnlineSecurity,DeviceProtection from main_view").show(5,false)
/*
+----------+------+-------------+-------+----------+------+------------+----------------+---------------+--------------+----------------+
|customerID|gender|SeniorCitizen|Partner|Dependents|tenure|PhoneService|MultipleLines   |InternetService|OnlineSecurity|DeviceProtection|
+----------+------+-------------+-------+----------+------+------------+----------------+---------------+--------------+----------------+
|7590-VHVEG|Female|0            |Yes    |No        |1     |No          |No phone service|DSL            |No            |No              |
|5575-GNVDE|Male  |0            |No     |No        |34    |Yes         |No              |DSL            |Yes           |Yes             |
|3668-QPYBK|Male  |0            |No     |No        |2     |Yes         |No              |DSL            |Yes           |No              |
|7795-CFOCW|Male  |0            |No     |No        |45    |No          |No phone service|DSL            |Yes           |Yes             |
|9237-HQITU|Female|0            |No     |No        |2     |Yes         |No              |Fiber optic    |No            |No              |
+----------+------+-------------+-------+----------+------+------------+----------------+---------------+--------------+----------------+
 */
    println("second part of main data base")
    spark.sql("select TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod," +
      "MonthlyCharges,TotalCharges,Churn from main_view").show(5,false)
/*
+-----------+-----------+---------------+--------------+----------------+-------------------------+--------------+------------+-----+
|TechSupport|StreamingTV|StreamingMovies|Contract      |PaperlessBilling|PaymentMethod            |MonthlyCharges|TotalCharges|Churn|
+-----------+-----------+---------------+--------------+----------------+-------------------------+--------------+------------+-----+
|No         |No         |No             |Month-to-month|Yes             |Electronic check         |29.85         |29.85       |No   |
|No         |No         |No             |One year      |No              |Mailed check             |56.95         |1889.5      |No   |
|No         |No         |No             |Month-to-month|Yes             |Mailed check             |53.85         |108.15      |Yes  |
|Yes        |No         |No             |One year      |No              |Bank transfer (automatic)|42.3          |1840.75     |No   |
|No         |No         |No             |Month-to-month|Yes             |Electronic check         |70.7          |151.65      |Yes  |
+-----------+-----------+---------------+--------------+----------------+-------------------------+--------------+------------+-----+
 */


    // Get the total counts of rows.
    println("Total rows=" + telco_DF.count())

/*
Total rows=7043
 */
    //Compute basic statistics for numeric columns - count, mean, standard deviation, min, and max.
    //Try to describe the columns and check how statistics are shown for numerical and categorical columns.
   println("telco describe")
    telco_DF.describe().show()

/*
+-------+----------+------+------------------+-------+----------+------------------+------------+-------------+---------------+--------------+------------+----------------+-----------+-----------+---------------+--------------+----------------+--------------------+------------------+------------------+-----+
|summary|customerID|gender|     SeniorCitizen|Partner|Dependents|            tenure|PhoneService|MultipleLines|InternetService|OnlineSecurity|OnlineBackup|DeviceProtection|TechSupport|StreamingTV|StreamingMovies|      Contract|PaperlessBilling|       PaymentMethod|    MonthlyCharges|      TotalCharges|Churn|
+-------+----------+------+------------------+-------+----------+------------------+------------+-------------+---------------+--------------+------------+----------------+-----------+-----------+---------------+--------------+----------------+--------------------+------------------+------------------+-----+
|  count|      7043|  7043|              7043|   7043|      7043|              7043|        7043|         7043|           7043|          7043|        7043|            7043|       7043|       7043|           7043|          7043|            7043|                7043|              7043|              7043| 7043|
|   mean|      null|  null|0.1621468124378816|   null|      null| 32.37114865824223|        null|         null|           null|          null|        null|            null|       null|       null|           null|          null|            null|                null| 64.76169246059922|2283.3004408418697| null|
| stddev|      null|  null|0.3686116056100135|   null|      null|24.559481023094442|        null|         null|           null|          null|        null|            null|       null|       null|           null|          null|            null|                null|30.090047097678482| 2266.771361883145| null|
|    min|0002-ORFBO|Female|                 0|     No|        No|                 0|          No|           No|            DSL|            No|          No|              No|         No|         No|             No|Month-to-month|              No|Bank transfer (au...|             18.25|                  |   No|
|    max|9995-HOTOH|  Male|                 1|    Yes|       Yes|                72|         Yes|          Yes|             No|           Yes|         Yes|             Yes|        Yes|        Yes|            Yes|      Two year|             Yes|        Mailed check|            118.75|             999.9|  Yes|
+-------+----------+------+------------------+-------+----------+------------------+------------+-------------+---------------+--------------+------------+----------------+-----------+-----------+---------------+--------------+----------------+--------------------+------------------+------------------+-----+
 */

//Check the “TotalCharges” column minutely. This should be a numerical column, but it is shown as a “string” type in schema.
    //  Try to describe only this column. It has a maximum value but no minimum value.

    println("TotalCharges describe")
    telco_DF.describe("TotalCharges").show()
/*
+-------+------------------+
|summary|      TotalCharges|
+-------+------------------+
|  count|              7043|
|   mean|2283.3004408418697|
| stddev| 2266.771361883145|
|    min|                  |
|    max|             999.9|
+-------+------------------+
 */
//Need to clean this “TotalCharges” column. There are missing values as space in this column. Replace the space values (“ “)
    //  of this column with null values and save the result into a new dataframe.

    println("Check and count space values present in TotalCharges column..")
    println(telco_DF.filter(col("TotalCharges")===" ").count())

    /*
       Check and count space values present in totalCharges column..
       11
        */

    //Replace empty string with null on selected columns

    val telco_DF_new = telco_DF
      .withColumn("TotalCharges", when(col("TotalCharges")===" ",null).otherwise(col("TotalCharges")))

    println("Check and count space values present in totalCharges column..")
    println( telco_DF_new.filter(col("TotalCharges").isNull || col("TotalCharges")==="").count())

    /*
    Check and count space values present in totalCharges column..
    0
     */

    //• Now, try to describe the “TotalCharges” column of the new dataframe and check if there is any difference.

    println("TotalCharges describe")
    telco_DF_new.describe("TotalCharges").show()

/*
+-------+------------------+
|summary|      TotalCharges|
+-------+------------------+
|  count|              7032|
|   mean|2283.3004408418697|
| stddev| 2266.771361883145|
|    min|             100.2|
|    max|             999.9|
+-------+------------------+
 */

//• Now, drop the null values from the “TotalCharges” column of the new dataframe and save the result into another dataframe.
    println("drop null values")
   val  telco_null_drop = telco_DF_new.na.drop(Seq("TotalCharges"))

    println("Check and count null values present in totalCharges column..")
    println( telco_null_drop.filter(col("TotalCharges").isNull || col("TotalCharges")==="").count())

/*
Check and count null values present in totalCharges column..
0
 */
    val telcoDf_new_deci = telco_null_drop
    .withColumn("TotalCharges", col("TotalCharges").cast(DecimalType(6,2)))

//Change the column type to double, describe the dataframe, and check if there is any difference.

    println("change to decimal")
    telcoDf_new_deci.printSchema()

/*
root
 |-- TotalCharges: decimal(6,2) (nullable = true)
 */


//• Let us check the customer attrition figure. Find out the total number of customers churned and not churned.

    telcoDf_new_deci.createOrReplaceTempView("perf_df_view")

    println("not churned")
    spark.sql("select count(*) from perf_df_view where churn = 'No' ").show()

/*
+--------+
|count(1)|
+--------+
|    5174|
+--------+
 */

    println("churned")
    spark.sql("select count(*) from perf_df_view where churn = 'Yes' ").show()
/*
+--------+
|count(1)|
+--------+
|    1869|
+--------+
 */
//• Now let us clean some categorical or String type columns. Check the distinct values of the following columns:
    //  'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies'.
    println("distinct values")
    spark.sql("select distinct OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV," +
      "StreamingMovies, churn from perf_df_view").show()

/*
+--------------+------------+----------------+-----------+-----------+---------------+-----+
|OnlineSecurity|OnlineBackup|DeviceProtection|TechSupport|StreamingTV|StreamingMovies|churn|
+--------------+------------+----------------+-----------+-----------+---------------+-----+
|           Yes|         Yes|             Yes|         No|        Yes|             No|  Yes|
|            No|          No|             Yes|        Yes|        Yes|            Yes|   No|
|           Yes|         Yes|              No|         No|        Yes|            Yes|  Yes|
|            No|         Yes|             Yes|        Yes|        Yes|             No|   No|
|            No|          No|             Yes|         No|        Yes|            Yes|  Yes|
|           Yes|          No|             Yes|        Yes|        Yes|             No|  Yes|
|            No|         Yes|             Yes|        Yes|        Yes|            Yes|   No|
|           Yes|         Yes|              No|        Yes|        Yes|            Yes|   No|
|           Yes|          No|             Yes|         No|         No|            Yes|  Yes|
|            No|          No|             Yes|        Yes|         No|             No|   No|
|            No|          No|             Yes|         No|        Yes|             No|  Yes|
 */

//Replace “No internet service” value with “No.”

   val telco_DF_renamed = telcoDf_new_deci
     .withColumn("OnlineSecurity", when(col("OnlineSecurity").equalTo("No internet service"), "No").otherwise(col("OnlineSecurity")))
        .withColumn("TechSupport", when(col("TechSupport").equalTo("No internet service"), "No").otherwise(col("TechSupport")))
    telco_DF_renamed.createOrReplaceTempView("renamed_view")
     println("renamed telco_df")
     spark.sql ("select gender, SeniorCitizen, Dependents, InternetService, OnlineSecurity, TechSupport, Contract, PaperlessBilling, churn from renamed_view where churn = 'No' ").show()


/*
+------+-------------+----------+---------------+--------------+-----------+--------------+----------------+-----+
|gender|SeniorCitizen|Dependents|InternetService|OnlineSecurity|TechSupport|      Contract|PaperlessBilling|churn|
+------+-------------+----------+---------------+--------------+-----------+--------------+----------------+-----+
|Female|            0|        No|            DSL|            No|         No|Month-to-month|             Yes|   No|
|  Male|            0|        No|            DSL|           Yes|         No|      One year|              No|   No|
|  Male|            0|        No|            DSL|           Yes|        Yes|      One year|              No|   No|
|  Male|            0|       Yes|    Fiber optic|            No|         No|Month-to-month|             Yes|   No|
|Female|            0|        No|    Fiber optic|            No|         No|Month-to-month|             Yes|   No|
 */

    spark.sql("select gender, SeniorCitizen, Dependents, InternetService, OnlineSecurity, TechSupport, Contract, PaperlessBilling, churn from renamed_view where churn = 'Yes' ").show()

/*
+------+-------------+----------+---------------+-------------------+-------------------+--------------+----------------+-----+
|gender|SeniorCitizen|Dependents|InternetService|     OnlineSecurity|        TechSupport|      Contract|PaperlessBilling|churn|
+------+-------------+----------+---------------+-------------------+-------------------+--------------+----------------+-----+
|  Male|            0|        No|            DSL|                Yes|                 No|Month-to-month|             Yes|  Yes|
|Female|            0|        No|    Fiber optic|                 No|                 No|Month-to-month|             Yes|  Yes|
|Female|            0|        No|    Fiber optic|                 No|                 No|Month-to-month|             Yes|  Yes|
|Female|            0|        No|    Fiber optic|                 No|                Yes|Month-to-month|             Yes|  Yes|
|  Male|            0|        No|    Fiber optic|                 No|                 No|Month-to-month|             Yes|  Yes|
|Female|            0|       Yes|            DSL|                 No|                Yes|Month-to-month|              No|  Yes|
|  Male|            1|        No|            DSL|                 No|                 No|Month-to-month|             Yes|  Yes|

 */

//• Find out the distribution of gender, senior citizen, dependents, Internet Service, Online Security, Tech Support,
    //  Contract, Paperless Billing with respect to churn. Show individual distribution in cross tabs.

    telco_DF_renamed.stat.crosstab("gender","churn").show()
/*
+------------+----+---+
|gender_churn|  No|Yes|
+------------+----+---+
|        Male|2625|930|
|      Female|2549|939|
+------------+----+---+
 */
    telco_DF_renamed.stat.crosstab("SeniorCitizen","churn").show()
/*
+-------------------+----+----+
|SeniorCitizen_churn|  No| Yes|
+-------------------+----+----+
|                  0|4508|1393|
|                  1| 666| 476|
+-------------------+----+----+
 */
    telco_DF_renamed.stat.crosstab("Dependents","churn").show()
/*
+----------------+----+----+
|Dependents_churn|  No| Yes|
+----------------+----+----+
|              No|3390|1543|
|             Yes|1784| 326|
+----------------+----+----+
 */
    telco_DF_renamed.stat.crosstab("InternetService","churn").show()
/*
+---------------------+----+----+
|InternetService_churn|  No| Yes|
+---------------------+----+----+
|                   No|1413| 113|
|          Fiber optic|1799|1297|
|                  DSL|1962| 459|
+---------------------+----+----+
 */
    telco_DF_renamed.stat.crosstab("OnlineSecurity","churn").show()

/*
+--------------------+----+----+
|OnlineSecurity_churn|  No| Yes|
+--------------------+----+----+
|                  No|3443|1574|
|                 Yes|1720| 295|
+--------------------+----+----+
 */

    telco_DF_renamed.stat.crosstab("TechSupport","churn").show()
/*
+-----------------+----+----+
|TechSupport_churn|  No| Yes|
+-----------------+----+----+
|               No|3433|1559|
|              Yes|1730| 310|
+-----------------+----+----+
 */
    telco_DF_renamed.stat.crosstab("Contract","churn").show()

/*
+--------------+----+----+
|Contract_churn|  No| Yes|
+--------------+----+----+
|Month-to-month|2220|1655|
|      One year|1307| 166|
|      Two year|1647|  48|
+--------------+----+----+
 */
    telco_DF_renamed.stat.crosstab("PaperlessBilling","churn").show()
/*
   +----------------------+----+----+
|PaperlessBilling_churn|  No| Yes|
+----------------------+----+----+
|                    No|2403| 469|
|                   Yes|2771|1400|
+----------------------+----+----+
 */

//• Select only monthly charges and total charges columns from the dataframe and filter them wit respect to churn values. Then,
//  explore their basic statistics for churned and not-churned customers.
   val month_total_no_df = spark.sql("select MonthlyCharges, TotalCharges,churn from renamed_view where churn = 'No'")
    month_total_no_df.describe().show()
/*
+-------+-----------------+-----------------+-----+
|summary|   MonthlyCharges|     TotalCharges|churn|
+-------+-----------------+-----------------+-----+
|  count|             5163|             5163| 5163|
|   mean|61.30740848343966|      2555.344141| null|
| stddev|31.09455690667256|2329.456983560435| null|
|    min|            18.25|            18.80|   No|
|    max|           118.75|          8672.45|   No|
+-------+-----------------+-----------------+-----+
 */
    month_total_no_df.show(5,false)
/*
+--------------+------------+-----+
|MonthlyCharges|TotalCharges|churn|
+--------------+------------+-----+
|29.85         |29.85       |No   |
|56.95         |1889.5      |No   |
|42.3          |1840.75     |No   |
|89.1          |1949.4      |No   |
|29.75         |301.9       |No   |
+--------------+------------+-----+
 */
val month_total_yes_df = spark.sql("select MonthlyCharges, TotalCharges,churn from renamed_view where churn = 'Yes'")
    month_total_yes_df.describe().show()

/*
+-------+------------------+------------------+-----+
|summary|    MonthlyCharges|      TotalCharges|churn|
+-------+------------------+------------------+-----+
|  count|              1869|              1869| 1869|
|   mean|  74.4413322632423|1531.7960941680035| null|
| stddev|24.666053259397422|1890.8229944644042| null|
|    min|             18.85|            100.25|  Yes|
|    max|            118.35|            999.45|  Yes|
+-------+------------------+------------------+-----+
 */
    month_total_yes_df.show(5,false)
/*
+--------------+------------+-----+
|MonthlyCharges|TotalCharges|churn|
+--------------+------------+-----+
|53.85         |108.15      |Yes  |
|70.7          |151.65      |Yes  |
|99.65         |820.5       |Yes  |
|104.8         |3046.05     |Yes  |
|103.7         |5036.3      |Yes  |
+--------------+------------+-----+
 */

//• Correlation is a statistical technique that can show whether and how strongly pairs of variables are related. Find out how
    //  “MonthlyCharges” and “TotalCharges” are correlated.

    println("corelation")
    println(telco_DF_renamed.stat.corr("MonthlyCharges","TotalCharges"))

   //print("corelation gender and monthly")
   println( telco_DF_renamed.stat.corr("SeniorCitizen","MonthlyCharges"))

   // print("corelation gender and total")
   println( telco_DF_renamed.stat.corr("SeniorCitizen","TotalCharges"))



    println("group by churn")
    telco_DF_renamed.groupBy("Churn").count().show()
    println("gender churn")
    telco_DF_renamed.stat.crosstab("gender","Churn").show()

//• Let us perform basic predictive analytics on this dataset. Select two features - "MonthlyCharges," "TotalCharges"
    //  from the last dataframe, and create a vector (VectorAssembler). To the features to be used by a machine learning algorithm,
    //  this vector needs to be added as a “feature” column into the Dataframe. Create this “feature” column in the dataframe.

   val for_per_df=
    spark.sql("select MonthlyCharges,TotalCharges,Churn from renamed_view")

    val cols = Array("MonthlyCharges","TotalCharges")
    val my_Assembler = new VectorAssembler()
      .setInputCols(cols)
      .setOutputCol("features")
    val telco_Feature = my_Assembler.transform(for_per_df)

    println("Feature Schema")

    telco_Feature.printSchema()
/*
root
 |-- MonthlyCharges: double (nullable = true)
 |-- TotalCharges: decimal(6,2) (nullable = true)
 |-- Churn: string (nullable = true)
 |-- features: vector (nullable = true)
 */

//• Next, we need to add a label column to the dataframe with the values of the churn column. StringIndexer can be used for that.
//  It will return a new dataframe by adding a label column with the value of the result column.

    val indexer = new StringIndexer()
    .setInputCol("Churn")
    .setOutputCol("label")

    val telco_Label = indexer
      .setHandleInvalid("keep")
      .fit(telco_Feature)
      .transform(telco_Feature)
    println("grade_label printSchema")
    telco_Label.printSchema()

/*
root
 |-- MonthlyCharges: double (nullable = true)
 |-- TotalCharges: decimal(6,2) (nullable = true)
 |-- Churn: string (nullable = true)
 |-- features: vector (nullable = true)
 |-- label: double (nullable = false)
 */

//• Split the dataframe into a training set and test set with a 70:30 ratio. Use randomForestClassifier to create a random forest
    //  classifier model with 50 trees and max depth of 10. Fit the training set to the model and make predictions on the test set.

    val seed = 5043
    val Array(trainData,testData)=telco_Label.randomSplit(Array(0.7,0.3),seed)

    val classifier = new GBTClassifier()
      .setMaxDepth(10)
      .setFeatureSubsetStrategy("auto")
      .setSeed(5043)

    val model = classifier.fit(trainData)

    val prediction_df = model.transform(testData)
    println("predictionDF dataframe")
    prediction_df.show(10,false)

/*
+--------------+------------+-----+--------------+-----+------------------------------------------+---------------------------------------------+----------+
|MonthlyCharges|TotalCharges|Churn|features      |label|rawPrediction                             |probability                                  |prediction|
+--------------+------------+-----+--------------+-----+------------------------------------------+---------------------------------------------+----------+
|18.7          |1005.70     |No   |[18.7,1005.7] |0.0  |[48.92426743396161,1.0757325660383754,0.0]|[0.9784853486792325,0.021514651320767516,0.0]|0.0       |
|18.8          |279.20      |No   |[18.8,279.2]  |0.0  |[45.89551376042884,4.104486239571147,0.0] |[0.917910275208577,0.08208972479142297,0.0]  |0.0       |
|18.85         |18.85       |Yes  |[18.85,18.85] |1.0  |[36.61169208970237,13.388307910297618,0.0]|[0.7322338417940476,0.2677661582059524,0.0]  |0.0       |
|18.85         |163.20      |No   |[18.85,163.2] |0.0  |[43.4297992763828,6.570200723617191,0.0]  |[0.8685959855276562,0.13140401447234384,0.0] |0.0       |
|18.85         |867.30      |No   |[18.85,867.3] |0.0  |[48.84947298187861,1.1505270181213734,0.0]|[0.9769894596375726,0.023010540362427473,0.0]|0.0       |
|18.9          |18.90       |No   |[18.9,18.9]   |0.0  |[36.61169208970237,13.388307910297618,0.0]|[0.7322338417940476,0.2677661582059524,0.0]  |0.0       |
|18.95         |110.15      |No   |[18.95,110.15]|0.0  |[44.61846044849231,5.381539551507692,0.0] |[0.8923692089698462,0.10763079103015384,0.0] |0.0
 */

//• Use MulticlassClassificationEvaluator and find out the accuracy in the prediction.

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(prediction_df)
    println("accuracy % = " + accuracy * 100 )

/*
accuracy % = 77.40585774058577
 */

    prediction_df.filter( "prediction = 1").show(10,false)

//• Find out the confusion matrix by showing the actual values of the “label” column and the value of the “prediction” column.

    println("confusion matix")

    prediction_df.groupBy(col("label"),col("prediction")).count().show

/*
   +-----+----------+-----+
|label|prediction|count|
+-----+----------+-----+
|  1.0|       1.0|  260|
|  0.0|       1.0|  162|
|  1.0|       0.0|  324|
|  0.0|       0.0| 1405|
+-----+----------+-----+
 */
  }
}
