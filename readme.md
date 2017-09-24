
<body marginheight="0"><h1>包含的文件</h1>
<p>包含的代码：rtb_train_spark.py, rtb_train_python.py, rtb_predict.py, utils.py

</p>
<p>配置文件： conf_train.py, conf_predict.py

</p>
<p>运行脚本： train.bash

</p>
<h4>conf_train :  配置内容如下</h4>
<p>path = './'

</p>
<p>ad = [114333,114368,114290]

</p>
<p>bid_path = 's3://logarchive.ym/hive/parquet_table/base/team=dsp/logtype=reqad/producttype=*/date=2017-08-10'

</p>
<p>session_path = 's3://logarchive.ym/hive/parquet_table/session/team=dsp/date=2017-08-10'

</p>
<h4>conf_predict :  配置内容如下</h4>
<p>path = './' # 是存放输出模型的路径

</p>
<p>ad_predict = [114333,114290]  # 这是准备预测的模型


</p>
<h6>训练的代码分为两部分，一部分需要使用pyspark，另一部分纯python。python依赖 xgboost，scipy1.18，numpy，sklearn。pyspark不需要另外安装东西。</h6>
<h4>rtb_train_spark.py :</h4>
<p>从 bid_path 和 session_path 拿数据。

</p>
<p>这个脚本会每两个小时跑一次。

</p>
<p>并不是每次都需要训练所有的axid，只有该axid的session数据增量超过5%或者这个模型12小时内没有训练过，才会进行新一次训练。

</p>
<p>对需要训练的每个axid，做onehot等操作，数据输出以libsvm的形式存成txt文件，留给下一步训练。然后把onehot后的column信息也存起来。

</p>
<p>这里有个问题就是libsvm格式的txt占空间比较大，跟parquet的储存效率相距甚远。


</p>
<h4>rtb_train_spark.py :</h4>
<p>对上一步训练过的axid，训练GBDT+LR模型，存起来。

</p>
<h4>train.bash：</h4>
<p>pyspark s3://datamining.ym/dmuser/jtan/script/rtb_train_spark.py

</p>
<p>python s3://datamining.ym/dmuser/jtan/script/rtb_train_python.py


</p>
<h4>rtb_predict.py :</h4>
<p>这个脚本是用来预测的，输入变量为一个dict，存放流量信息。

</p>
<p>脚本会先加载所有axid的模型，如果流量的axid有现成模型，则使用此模型来预测。

</p>
<p>否则，查看同一adid的其他axid模型。如果还是没有找到，则使用历史上最好的模型来预测。

</p>
<h4>部署建议 :</h4>
<p>在 aws上跑rtb_train_spark.py,生成一个很大的文件。

</p>
<p>因为不方便拉回来，所以最好也是在aws上的python机器上进行训练模型。

</p>
<p>最终输出的model可以拿来部署在竞价机子上。

Edit By <a href="http://mahua.jser.me">Tan Jin</a></p>
</body></html>
