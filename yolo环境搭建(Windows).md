---
typora-root-url: res
---

# Anaconda

### 下载Anaconda

```
https://www.anaconda.com/download/success
具体方法可以在网上找其他教程，这里不多赘述
```

![0.1](/0.1.png)

### 查看镜像源

```
conda config --show channels
```

### 删除恢复默认

```
conda config --remove-key channels
```

### 添加清华源

```
#添加镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2

#终端显示包从哪个channel下载，以及下载地址是什么
conda config --set show_channel_urls yes
```

### 创建虚拟环境

```
在开始菜单找到Anaconda prompt
```

![0.2](/0.2.png)

```
输入：conda create -n 环境名 python=X.X
```

![0.3](/0.3.png)

```
查看当前存在的虚拟环境：
conda env list 
*号代表当前所处的虚拟环境，此处为base
```

![0.4](/0.4.png)

### 激活/切换虚拟环境

```
输入：activate 环境名
```

![0.5](/0.5.png)

# 电脑配置

### **查看显卡型号**

```
win+R
输入cmd 跳出命令提示符
输入dxdiag 在DirectX诊断工具 找到显示这一栏
```

##### 集成显卡

![01](/01.png)

##### 独立显卡

![02](/02.png)

##### 下载/更新驱动

```
GeForce Game Ready:https://www.nvidia.cn/drivers/lookup/
```

![03](/03.png)

![04](/04.png)

### 查看显卡算力

```
https://developer.nvidia.com/cuda-gpus
```

![05](/05.png)

### 查看显卡支持的CUDA版本

##### 方法1

```
win+R
输入cmd 跳出命令提示符
输入nvidia-smi
```

![06](/06.png)

##### 方法2

```
在设置中找到NVIDIA Control Panel
打开后点击--系统信息--组件
找到--3D设置--NVCCUDA64.DLL
```

![07](/07.png)

![08](/08.png)

![09](/09.png)

# CUDA

### 安装CUDA

```
https://developer.nvidia.com/cuda-toolkit-archive
```

![10](/10.png)

![11](/11.png)

##### 查看是否安装成功

```
win+R
输入cmd 跳出命令提示符
查看当前CUDA版本，输入nvcc -V
```

![14](/14.png)

### 卸载CUDA（可选）

```
win+R
输入cmd 跳出命令提示符
查看当前CUDA版本，输入nvcc -V
```

![12](/12.png)

![13](/13.png)

# cuDNN

### 查看CUDA支持的cuDNN

```
https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html#support-matrix
```

![15](/15.png)

### 下载cuDNN

```
下载地址:https://developer.nvidia.com/cudnn-downloads
下载后解压，后续步骤可以查看官方文档:
https://docs.nvidia.com/deeplearning/cudnn/latest/installation/windows.html
也可以往下继续看
```

![16](/16.png)

##### 复制文件

```
解压后打开文件夹
复制bin\cudnn*.dll到。C:\Program Files\NVIDIA\CUDNN\v9.x\bin
复制include\cudnn*.h到。C:\Program Files\NVIDIA\CUDNN\v9.x\include
复制lib\x64\cudnn*.lib到。C:\Program Files\NVIDIA\CUDNN\v9.x\lib
注：后面的路径是自己安装CUDA Toolkit的路径根据自己的来，v9.x是版本号
```

![17](/17.png)

##### 配置环境变量

![18](/18.png)

##### 查看是否配置正确

```
在extras\demo_suite路径中输入cmd打开命令提示符
```

![19](/19.png)

![20](/20.png)

```
运行 bandwidthTest.exe 和 deviceQuery.exe
```

![21](/21.png)

![22](/22.png)

# PyTorch

### 下载PyTorch

```
https://pytorch.org/get-started/locally/
注：前面下载的CUDA版本大于PyTorch支持版本，所以就下载12.4
```

![23](/23.png)

```
切换到你需要的虚拟环境：activate 环境名
运行命令：conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
注：此处安装环境需要耐心等待
```

![24](/24.png)

# 下载yolov8源码

```
https://github.com/ultralytics/ultralytics
```

### 方法1

```
直接下载源码压缩包
```

![26](/26.png)

### 方法2

```
从github上克隆到pycharm
打开pycharm--file--Project from Version Control...
https://github.com/ultralytics/ultralytics.git
```

![27](/27.png)

# 配置python解释器环境

```
方法有很多 这里选择最方便的
右下角--选择Add New Interpreter--Add Local interpreter
```

![29](/29.png)

```
下面操作看图片
Environment：选择select existing（已存在的环境）
Type:选择conda
path to conda：在安装Anaconda的目录中找到--script--conda.exe
Environment：选择刚刚创建的虚拟环境（这里是yolov8）可以点击Reload environments重新加载一下环境
```

![30](/30.png)

# 安装yolov8所需依赖

```
pip install ultralytics
方法有很多这里选择最简单的一种
在yolov8项目下找到README.md
直接运行该代码会自动安装
具体操作看图片
注:下载速度过慢可以用pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ultralytics
```

![31](/31.png)

```
至此yolov8环境搭建成功 下面就可以开始寻找数据集 标注数据
```

