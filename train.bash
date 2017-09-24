# bash getdata.bash
bash renew_data.bash
# spark-submit --master local[-1] rtb_train_allonce_bid.py
spark-submit --master local[8] rtb_train_allonce_cvr.py
spark-submit --master local[8] rtb_train_allonce_cvr_adid.py
spark-submit --master local[8] rtb_train_allonce_cvr_adid_old.py
spark-submit --master local[8] table.py

de=`date +%Y-%m-%d_%H:%M:%S`
aws s3 cp --recursive /home/ymserver/workplace/rtb_online/models s3://datamining.ym/dmuser/jtan/models/$de/models
aws s3 cp --recursive /home/ymserver/workplace/rtb_online/old_models s3://datamining.ym/dmuser/jtan/models/$de/old_models
aws s3 cp --recursive /home/ymserver/workplace/rtb_online/constants s3://datamining.ym/dmuser/jtan/constants/$de/constants


