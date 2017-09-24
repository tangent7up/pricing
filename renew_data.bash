session_path=/data/session
bid_path=/data/bid

#session_path=/home/youmi/workplace/rtb_online/session_aws
#bid_path=/home/youmi/workplace/rtb_online/bid_aws

# rm -rf $session_path
# rm -rf $bid_path
mkdir $session_path
mkdir $bid_path

for((i=0;i<=10;i++));
do
  de=`date -d "$i days ago" +%Y-%m-%d`
  rm -rf $bid_path/producttype=banner/date=$de/
  rm -rf $bid_path/producttype=feed/date=$de/
  rm -rf $bid_path/producttype=native/date=$de/
  rm -rf $bid_path/producttype=spot/date=$de/
  rm -rf $bid_path/producttype=video/date=$de/
  aws s3 cp --recursive s3://logarchive.ym/hive/parquet_table/base/team=dsp/logtype=reqad/producttype=banner/date=$de/ $bid_path/producttype=banner/date=$de/
  aws s3 cp --recursive s3://logarchive.ym/hive/parquet_table/base/team=dsp/logtype=reqad/producttype=feed/date=$de/ $bid_path/producttype=feed/date=$de/
  aws s3 cp --recursive s3://logarchive.ym/hive/parquet_table/base/team=dsp/logtype=reqad/producttype=native/date=$de/ $bid_path/producttype=native/date=$de/
  aws s3 cp --recursive s3://logarchive.ym/hive/parquet_table/base/team=dsp/logtype=reqad/producttype=spot/date=$de/ $bid_path/producttype=spot/date=$de/
  aws s3 cp --recursive s3://logarchive.ym/hive/parquet_table/base/team=dsp/logtype=reqad/producttype=video/date=$de/ $bid_path/producttype=video/date=$de/
done


for((i=0;i<=10;i++));
do
  de=`date -d "$i days ago" +%Y-%m-%d`
  rm -rf $session_path/date=$de/
  aws s3 cp --recursive s3://logarchive.ym/hive/parquet_table/session/team=dsp/date=$de/ $session_path/date=$de/
done
