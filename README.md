#### 代码使用

安装好 torch 及相关 packages 后，直接运行 run.py 文件即可，相关超参数可在 run.py 文件中修改。

其中，

```python
para_kv = {
    "model": " --model UNet",  # UNet, ENet, deeplabv3plus_mobilenet,
    "dataset": " --dataset drive",
    "batch_size": " --batch_size 1",
    "loss": " --use_lovaszsoftmax",  # keep only one loss
    # "loss": " --use_label_smoothing",
    # "loss": " --use_ohem",
    "optim": " --optim sgd",  # adamw, sgd (refer to train.py)
    "traintype": " --train_type train"  # trainval, train
    # "resume": " --resume ./checkpoint/drive/UNetbs1gpu1_train/model_12.pth",
}

sh = "python ./train.py"

for _, v in para_kv.items():
    sh += v

os.system(sh)
```

为训练所用代码。

另外，

```python
model_epoch = 1
sh = f"python ./predict.py --model UNet --dataset drive --checkpoint ./checkpoint/drive/UNetbs1gpu1_train/model_{model_epoch}.pth"

os.system(sh)
```

为测试集代码。

测试完后，可在 predict 文件夹中查看可视化结果。

<img src="https://i.loli.net/2021/03/25/dszDSfb8qKFNagk.png" alt="image-20210325145549847" style="zoom: 80%;" />

目前模型性能一般，大家可以在 baseline 的基础上进行改动。(o゜▽゜)o☆ 有問題及時交流~

#### 比赛

比赛地址：https://drive.grand-challenge.org/DRIVE/

比赛仅提供了 20 张训练集样本，如需测试模型性能需要到官网进行提交。