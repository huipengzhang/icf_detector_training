一些代码片段用于训练测试

推荐看一下每个脚本的内容，并不复杂，弄懂之后知道用起来更加得心应手

1. 训练单尺度模型
```
sh get_neg_train_file.sh ~/Documents/ball ball #得到球的负样本文件名列表
同理,要得到球的正样本文件名列表
sh get_pos_train_file.sh ~/Documents/goal goal #得到球门的正样本文件名列表
同理,要得到球门的负样本文件名列表
```
训练之前总是要得到训练物体的正、负样本
其中正样本各个尺度的是小图片，负样本是没有待训练样本的大图片

```
sh train.sh ball_ ball_neg.txt 1 ball #训练球的模型
sh train.sh goal_ goal_neg.txt 4/3 goal #训练球门的模型
```
其中1, 4/3是长比宽的比值，可以看扣出的正样本图片，计算得到


2. 多个单一尺度模型生成多尺度模型
```
../build/main_get_multiscale_model model_ball_24_3.proto.bin model_ball_48_3.proto.bin model_ball_96_3.proto.bin model_ball_192_3.proto.bin
```
用的是每个尺度的最后一轮（第3轮）的模型生成一个multiscale_model，也是机器人最终使用的模型


3. 测试
```
../build/main_test multiscale_model.proto.bin test_images.txt true
```
用模型multiscale_model.proto.bin去检测test_images中的图片，最后一个参数设成true的话就只以白色区域为待检测的区域
