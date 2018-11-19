# NOTES - Spark The Definitive Guide

---------------------------
## Production Applications - How SPARK run on Cluster (Chapter 15)
---------------------------

#### The Spark Driver (SD)
* Controller of the execution of Spark Application
* Maintain all the state of the Spark Cluster (the state & task of executors)
* Interface with __Cluster Manager__...
    - get physical resources
    - launch executors

#### The Spark Executors (EX)
* Processes performing tasks assigned by Spark driver
* take the task __==>__ run them __==>__ report back STATE & RESULTS 
* __Every Spark application__ has its __own seperate Executor processes__.

#### The Cluster Manager (CM)
* Spark Driver and Executors are tied together by a Cluster Manager
* responsible for maintaining the cluster fo machines on which the Spark appication is run
* A __cluster manager__ has its own "__driver__" (sometimes called as Master) & "__worker__" __abstractions__. Only difference being they __are tied to physical machines rather than processes__.
* Cluster Manager provide the resources for execution when asked for. Depending on how our application is configured this may include...
    - a place to run Spark driver
    - or just the resources for the executors for our Spark Application.
* 3 cluster manager supported -
    - simple built-in standalone cluster manager
    - Apache Mesos
    - Hadoop YARN

#### Execution Modes
* gives the power to decide where the resources are physically located
* 3 modes to choose from -
    - Cluster mode
    - Client mode
    - Local mode

##### Cluster Mode
* most common way to run Spark
* Processes -
    - User pre-submit the jar/python script to a CM
    - CM launches driver process on a worker node inside the cluster, in addition to the executor processes.
    - __CM is responsible for maintaining all Spark Appn related processes.__
    - __Both SD & EX can be started any of the worker node of the CM__

##### Client Mode
* Similar to Cluster mode, except the __SD remains on the client machine__ that submitted the application.
* __Client Machine__ is __responsible__ for maintaining the __SD process__, and __CM maintains__ the __executor process__.
* __Spark Application__ are __running__ on the machines that is __not colocated on the cluster__ (refer to as __gateway/edge nodes__)
* __Worker__ located __in the cluster__ and __manged by the CM__.

##### Local Mode
* Run entire Spark application on Single machine.
* achive parallelism thrugh Threading.
* Commonly used for learning purpose or for local development.

#### Life cycle of Spark Application (Outside Spark)
* Overall life cycle of the Spark Application "outside" the actual Spark code.
* Cluster - 4 nodes ===>> 1 CM driver & 3 CM worker nodes
* Full Process
    - __Client Request__
        + Submit the client request using "__spark-submit__"
        + Explicitly __ask__ for __resources__ for __Spark Driver process__ only
        + CM accepts the request and place the SD in the worker node
        + client process that submitted the request exists
    - __Launch__
        + Driver process already placed in the cluster, start running code
        + Must always include __SparkSession__(initializes Spark cluster, Drive + executor)
        + Num of executors and conf are set using command line in the initial call or through code.
        + __CM__ responds by __launching the Executor__ processes, and __inform__ their __location__ to the __Driver__.
    - __Execution__
        + Now code execution starts
        + driver and executors communicate among themselves, executing code and moving around data
        + Driver schedules task onto each worker
        + Wroker respond with status/result
    - __Completion__
        + After completion, SD process exits with either success or failure
        + CM shuts down the executors in the cluster for the driver 
        + Status can be seen using CM

#### Life cycle of Spark Application (Inside Spark)
* Full Process
    - The __SparkSession__
        + initialize this manually
        + To create spark session in Python
```

from pyspark.sql import SparkSession

spark = SparkSession.builder.master('local')\
                            .appName('<app name>')\
                            .config('spark.some.config.option', 'some-value')\
                            .getOrCreate()
```
        + Old Version used to create SparkContext (but this is not the way to be used now)
    - The __SparkContext__
        + represent the connection to the Spark cluster
        + helps to communicate with some of Spark's lower level API, such as RDDs
        + can create RDDs, accumulators & broadcast variables
```

from pyspark import SparkContext
logFile = "file:///local/file/path"
sc = SparkContext("local", "first app")
logData = sc.textFile(logFile).cache()

```
        + The Sparkcontext can also be created __using the SparkSession__. (_the step below need to be evaluated_)
```
from pyspark.sql import SparkSession

spark = SparkSession.builder.master('local')\
                            .appName('<app name>')\
                            .config('spark.some.config.option', 'some-value')\
                            .getOrCreate()

sc = spark.SparkContext
```
        + For Spark 2.x, the 2 apis SQLContext & SparkContext has been combined into SparkSession. You can also create Spark and SQL context from their respective getOrCreate() func, but it is not necessary now.

    - The Spark Job
        + Each __Spark job__ is started __for one action__.
        + Each __job breaks into stages__, depending on the number of __shuffle operations__ that need to take place


