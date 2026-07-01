# 论文原始结果

原始仓库：https://github.com/c-yn/BioIR

单一退化的可视化结果：[百度网盘](https://pan.baidu.com/s/18EIFlLx-xSQRIoLc62Qt6A?pwd=x65n)

<img src="img/README_img/image-20260630192144846.png" alt="image-20260630192144846" style="zoom:80%;" />



<img src="img/README_img/image-20260630192207889.png" alt="image-20260630192207889" style="zoom:80%;" />

# 复现

只复现Single_Composite中单一退化的LOLv2-syn，其他的复合退化、All-in-one没有关注



## 配置环境

配置conda环境：

```
conda create -n bioir python=3.9
conda activate bioir

# 安装依赖
conda install pytorch=2.4.0 torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
pip install opencv-python lmdb tqdm einops scipy scikit-image tensorboard natsort pyiqa joblib lpips scikit-learn pandas

# 安装basicsr
cd Single_Composite
python setup.py develop --no_cuda_ext
```



## 下载数据集

数据集：LOLv1、LOLv2

- BioIR只在LOL-v2-syn上训练，没有在LOLv1和LOLv2-real训练

```
python3 -m pip install -U gdown
gdown "https://drive.google.com/uc?id=1mAN3ll5wWwt1Xz0C7uio31-NJu-50S8Z"
gdown "https://drive.google.com/uc?id=1dzLJFz0svHXYHvAe-Tl52miChhF4BXXE"

apt install -y unzip
mkdir -p Single_Composite/datasets
unzip LOL-v1.zip -d Single_Composite/datasets
unzip LOL-v2.zip -d Single_Composite/datasets
```



目录结构

```
Single_Composite/
  datasets/
    LOL-v1/
      our485/
        low/
        high/
      eval15/
        low/
        high/
    LOL-v2/
      Synthetic/
        Train/
          Low/
          Normal/
        Test/
          Low/
          Normal/
      Real_captured/
        Train/
          Low/
          Normal/
        Test/
          Low/
          Normal/
```

**修改yaml配置**：

```
datasets:
  train:
    dataroot_gt: ./datasets/LOL-v2/Synthetic/Train/Normal
    dataroot_lq: ./datasets/LOL-v2/Synthetic/Train/Low、
  val:
    dataroot_gt: ./datasets/LOL-v2/Synthetic/Test/Normal
    dataroot_lq: ./datasets/LOL-v2/Synthetic/Test/Low
```



## 测试

下载预训练权重，放到`pretrained_models/`：[Google Drive](https://drive.google.com/drive/folders/1VrFxqox3fewPUmP-i0a9rJw3qCmT1Vnp?usp=sharing)、[百度网盘](https://pan.baidu.com/s/1AEieYLl5i-afkr-bF47a_g?pwd=ja58)

**原始的测试流程**✖：原脚本里的 `--data` 枚举没有区分 `LOL-v1`、`LOL-v2-syn`、`LOL-v2-real`，并且默认按 `pretrain_model/<data>.pth` 找权重；如果继续用原脚本，需要同时修改 `eval.py`、`metrics_score.py` 和数据集枚举。因此建议直接用下面的新脚本 `test_lol.py`。

```
# 可视化实验：输出增强图
python eval.py --data CSD
# 定量实验：计算PSNR、SSIM
python metrics_score.py --data CSD
```

新建的`test_lol.py`⭐：同时完成推理、保存增强图、按同名 GT 计算 PSNR/SSIM，并把每张图和平均指标写入 `metrics.csv`。

**下载的 BioIR 预训练权重**：放在`BioIR/Single_Composite/pretrained_models/`

```
cd Single_Composite

# 测试下载的预训练权重示例
python test_lol.py --opt options/LOL-v2-syn.yml --weights pretrained_models/LOL-v2-syn.pth
# 额外保存低光图/增强图/GT 的横向拼接对比图
python test_lol.py --opt options/LOL-v2-syn.yml --weights pretrained_models/LOL-v2-syn.pth --save_comparison

# 测试自己训练出的权重
# LOL-v1
python test_lol.py --opt options/LOL-v1.yml --weights experiments/BioIR-LOLv1/models/net_g_latest.pth
# LOL-v2-syn
python test_lol.py --opt options/LOL-v2-syn.yml --weights experiments/BioIR-LOLv2-syn/models/net_g_latest.pth
# LOL-v2-real
python test_lol.py --opt options/LOL-v2-real.yml --weights experiments/BioIR-LOLv2-real/models/net_g_latest.pth
```

输出位置：

```text
results_lol/<实验名>/
  restored/      # 增强后图片
  metrics.csv    # 每张图和平均 PSNR/SSIM
```

默认按 RGB 三通道计算 PSNR/SSIM。如果你要和只报 Y 通道的论文口径对齐，可以加：

```powershell
python test_lol.py --opt options/LOL-v1.yml --weights experiments/BioIR-LOLv1/models/net_g_latest.pth --test_y_channel
```







## 训练

**训练**：

```
cd Single_Composite

python basicsr/train.py -opt options/LOL-v2-syn.yml
python basicsr/train.py -opt options/LOL-v1.yml
python basicsr/train.py -opt options/LOL-v2-real.yml
```

原始 README 也可以用 `torchrun`。它是 PyTorch 分布式启动器，即使只有 1 张 GPU，也按“单进程分布式”方式跑。普通单卡实验不需要优先用它；原生 Windows 上还可能因为 `nccl` 分布式后端不可用而报错。

```
# win
torchrun --nproc_per_node=1 --master_port=4322 basicsr/train.py -opt options/LOL-v1.yml --launcher pytorch
torchrun --nproc_per_node=1 --master_port=4322 basicsr/train.py -opt options/LOL-v2-syn.yml --launcher pytorch
torchrun --nproc_per_node=1 --master_port=4322 basicsr/train.py -opt options/LOL-v2-real.yml --launcher pytorch

# linux
sh train.sh options/LOL-v1.yml
sh train.sh options/LOL-v2-syn.yml
sh train.sh options/LOL-v2-real.yml
```

**周期性输出评价指标、保存模型权重和断点状态**

- 训练中断后，原训练脚本会自动从 `experiments/<实验名>/training_states/` 里最新的 `.state` 恢复

```
val:
  val_freq: 1e3

logger:
  save_checkpoint_freq: 1e3
```

**保存目录**：

```
Single_Composite\experiments\<实验名>\
  models\
  training_states\
  
# 示例
Single_Composite\experiments\BioIR-LOLv1\models\net_g_1000.pth
Single_Composite\experiments\BioIR-LOLv1\models\net_g_latest.pth
Single_Composite\experiments\BioIR-LOLv1\training_states\1000.state
```



## 消融实验



