## 问题现象

在 BioIR 的 `Single_Composite` 目录中执行：

```powershell
python setup.py develop --no_cuda_ext
```

安装 `basicsr` 时出现报错：

```text
ModuleNotFoundError: No module named 'torch'
```

但当前 `bioir` conda 环境里其实已经安装了 `torch`，可以正常输出 `2.4.0`。

```powershell
python -c "import torch; print(torch.__version__)"
```

所以，这个错误不是“主环境没装 torch”，而是安装过程被新版 `pip/setuptools` 拉进了临时的隔离构建环境，临时环境里看不到当前 conda 环境已经安装好的 `torch`。



## 直接原因

BioIR 的 `Single_Composite/setup.py` 是老版 BasicSR 风格，文件开头直接导入了 `torch`，导致安装脚本在准备构建元数据时就需要 `torch`。

```python
import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)
```

旧版 `setuptools` 执行 `python setup.py develop` 时，通常直接在当前 conda 环境里运行，因此能找到当前环境中的 `torch`。

新版 `setuptools/pip` 更倾向于走 PEP 517 隔离构建流程，等价于内部调用类似：

```powershell
python -m pip install -e . --use-pep517
```

这时 pip 会创建一个临时构建环境。临时环境只安装它认为需要的构建依赖，不一定包含当前 conda 环境中的 `torch`，于是 `setup.py` 顶层 `import torch` 就失败了。



## 为什么 RetinexFormer 之前没有报错

本地对比发现，RetinexFormer 和 BioIR 的 `setup.py` 基本是同一套老 BasicSR 安装脚本，都在文件开头直接 `import torch`。

RetinexFormer 不是因为代码写法更稳才没报错，而是当时的运行环境不同；旧环境里 `setuptools` 大概率没有触发现在这种隔离构建流程，所以能直接使用当前环境中的 `torch`。

- RetinexFormer README 中建议的是 `python=3.7`、`pytorch=1.11`。
- BioIR 这次使用的是 `python=3.9`、`torch=2.4.0`、`pip=25.2`，并且一开始 `setuptools` 是较新的版本。



## BasicSR 官方仓库怎么说

BasicSR 官方安装文档：

- 要求 `Python >= 3.7`、`PyTorch >= 1.7`。
- 官方说明主要面向 Linux，并写明 Windows 未测试。
- 官方新版 BasicSR 的 `setup.py` 已经比 BioIR/RetinexFormer 内置的老 fork 更稳。官方新版不是一上来就导入 `torch`，而是在需要编译扩展时才导入。BioIR/RetinexFormer 这类老 fork 仍保留“顶层 import torch”的写法，所以更容易被新版构建流程卡住。
- 如果只是作为普通包使用，可以执行：

```powershell
pip install basicsr
```

- 如果要本地开发或修改 BasicSR，官方仍然给的是：

```powershell
python setup.py develop
```

- 如果要编译 C++/CUDA 扩展，官方新版 BasicSR 使用环境变量：

```bash
BASICSR_EXT=True python setup.py develop
```

官方文档参考：

- https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md
- https://github.com/XPixelGroup/BasicSR/blob/master/setup.py



## 推荐解决方案

对 BioIR、RetinexFormer 这种论文复现仓库，最稳的做法是：每个项目单独建 conda 环境，并把该环境里的 `setuptools` 固定到老版本。让 `python setup.py develop` 走更接近旧式的安装流程，避免新式隔离构建环境看不到 `torch`。

BioIR 推荐流程：

```powershell
conda create -n bioir python=3.9 -y
conda activate bioir

conda install pytorch=2.4.0 torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install opencv-python lmdb tqdm einops scipy scikit-image tensorboard natsort pyiqa joblib lpips scikit-learn pandas

cd D:\Code\02_论文复现\低照度图像增强\论文代码\BioIR\Single_Composite
python -m pip install "setuptools<64" wheel
python setup.py develop --no_cuda_ext
```



## 6.可选方案：修改 setup.py

理论上也可以修改 BioIR 的 `Single_Composite/setup.py`，参考官方新版 BasicSR 的写法：

- 不在文件顶部直接 `import torch`。
- 只有需要编译 CUDA/C++ 扩展时，才导入：

```python
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
```

这样可以减少对旧版 `setuptools` 的依赖。

但对论文复现来说，不建议优先改安装脚本。原因是：

- 改安装脚本会引入额外变量。
- 后续出错时不容易判断是环境问题、代码问题，还是自己修改脚本导致的问题。
- 降级 `setuptools` 是局部影响，只影响当前 conda 环境，风险更低。

因此当前推荐：先用 `setuptools<64` 解决安装，等模型训练、测试跑通后，再考虑是否清理安装脚本。

