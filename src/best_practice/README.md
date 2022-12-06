# PyTorch 实践


## 安装

PyTorch : 可按照[PyTorch官网](http://pytorch.org)的指南，根据自己的平台安装指定的版本

### 安装 pytorch

**CPU 版本**
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

**CUDA 版本**
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

### 安装指定依赖：

```
pip install -r requirements.txt
```

## 训练

使用如下命令启动训练：

```
python main_torch.py -a resnext101_32x8d --lr 0.008 -b 32 --epoch 150 <dataset_dir>
```


详细的使用命令 可使用
```
python main_torch.py --help
```
