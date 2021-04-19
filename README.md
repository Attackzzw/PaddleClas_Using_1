# Windows：如何使用PaddleClas做一个完整的项目(一）
Windows下PaddleClas全流程文档 第一篇
## AiStudio:[AiStudio](https://aistudio.baidu.com/aistudio/projectdetail/1133588)
# PaddleClas简介：
**飞桨图像分类套件PaddleClas是飞桨为工业界和学术界所准备的一个图像分类任务的工具集，助力使用者训练出更好的视觉模型和应用落地。**

## 一、环境搭建
#### **Windows搭建环境请查看文章：[Windows-paddle-深度学习环境搭建](https://aistudio.baidu.com/aistudio/projectdetail/1147447)**
### 1.1 PaddleClas环境安装


**· 通过pip install paddleclas安装paddleclas whl包**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#### 包括ResNet、HRNet、ResNeSt、MobileNetV1/2/3、GhostNet等，所有可以使用的模型在import paddleclas的时候都会直接打印出来。


#### **pip安装**

`pip install paddleclas==2.0.0rc2`

#### **本地构建并安装**

```
python3 setup.py bdist_wheel
pip3 install dist/paddleclas-x.x.x-py3-none-any.whl # x.x.x是paddleclas的版本号
```

**· 克隆PaddleClas模型库：**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```
cd /d  path_to_clone_PaddleClas
git clone https://github.com/PaddlePaddle/PaddleClas.git
```

#### **安装Python依赖库：**

#### **Python依赖库在requirements.txt中给出，可通过如下命令安装：**

`pip install --upgrade -r requirements.txt`

#### **visualdl可能出现安装失败，请尝试**

`pip3 install --upgrade visualdl==2.0.0b3 -i https://mirror.baidu.com/pypi/simple`

#### **此外，visualdl目前只支持在python3下运行，因此如果希望使用visualdl，需要使用python3。**

**·  也可以再浏览器直接进入[github](https://github.com/PaddlePaddle/PaddleClas)**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#### **下载压缩包**

<br/>

<img src="https://ai-studio-static-online.cdn.bcebos.com/92294f2a1e0f40248607a90f1007423e92c163aa08364994bd081c4eb222ec45" width = "800" height = "400" align=center />

<br/>

#### **解压到指定目录，作为PaddleClas工作目录**

#### **运行cmd，通过如下命令安装Python依赖库**

`pip install --upgrade -r requirements.txt`

#### **待安装完成即可**




## 二、数据和模型准备
### 安装好paddle环境和paddleClas套件后，开始训练
### 2.1准备数据集
### 如何自定义分类任务数据集
- 首先，我们需要将数据图片按照类别都存放在各自文件夹里
- 把数据图片按照文件夹名称制作标签，并放在同一文件夹内以便读取
#### 这里用12猫分类作例子
- 共有12类图片，每一类包含180张并存放在各自文件夹内 

<img src="https://ai-studio-static-online.cdn.bcebos.com/b7bf0464958e4eb9b20109c16d515b5a6955db6d5cd84f32b743b5a203350f27" width = "800" height = "400" align=center />

<img src="https://ai-studio-static-online.cdn.bcebos.com/b34aee9899174821bdffbe92ab01b4a0254e2eb91b034be7b6f12a4e12271557" width = "800" height = "400" align=center />

```
import os 
import shutil
a =0
for i in os.walk("E:/cat_12/1"):   #读取目录内子文件夹名称，以及文件  [0] 文件夹  [1] 子文件夹 [2] 文件夹和子文件夹内文件名
    if a>0:
        f = open("E:/cat_12/train_list_1.txt",'a')    # 追加模式打开标注文件
        isExists=os.path.exists("E:/cat_12/cat_12_train")    # 判断路径是否存在
        if not isExists:    # 如果路径不存在则创建
            os.makedirs("E:/cat_12/cat_12_train")
        for j in range(0,len(i[2])):    # 统一文件到一个目录，并做标注
            line = i[0]
            line = './cat_12_train/'+i[2][j]    if i[0][-2]=='1' else './cat_12_train/'+i[2][j]            
            f.write(line+' '+i[0][-2:]+'\n')  if i[0][-2]=='1'  else  f.write(line+' '+i[0][-1]+'\n')    #写入标注文件
            try:
                line = i[0]
                line = line[:11]+'/'+line[-2:]+'/'+i[2][j]    if i[0][-2]=='1' else line[:11]+'/'+line[-1]+'/'+i[2][j]
                shutil.move(line, "E:/cat_12/cat_12_train")    # 移动文件
            except:
                print(line+'图片已移动或不存在，请检查')
        f.close()
    a += 1 
```

<img src="https://ai-studio-static-online.cdn.bcebos.com/fc456b94fa3644a58803423fd1c0c27254950c0b94914f8b8e3a5bd1c658a0bd" width = "800" height = "400" align=center />

- 运行脚本，图片和标注文件会保存在'E:/cat_12/cat_12_train'目录

<img src="https://ai-studio-static-online.cdn.bcebos.com/b89c5cb226624b56824b3c8241028cedb44a3ae3229c4f3da9d67fab3f3113f6" width = "800" height = "400" align=center />

<img src="https://ai-studio-static-online.cdn.bcebos.com/21955bd9ceae456ebc554f6722337dc518ba29916f79487a8f2fde3f667cba53" width = "800" height = "400" align=center />

- 整理完成后，还需要划分训练集和测试集

```
import os
import random
import numpy as np

val_percent = 0.1
picfilepath = 'E:/cat_12/cat_12_train'

f = open("E:/cat_12/train_list_1.txt","r")
line = f.readlines()

# 打乱文件顺序
np.random.shuffle(line)
# 划分训练集、测试集
train = line[:int(len(line)*(1-val_percent))]
test = line[int(len(line)*(1-val_percent)):]

# 分别写入train.txt, test.txt	
with open('train_list.txt', 'w') as f1, open('test_list.txt', 'w') as f2:
    for i in train:
        f1.write(i)
    for j in test:
        f2.write(j)

print('完成')
```
- 运行完脚本可以发现 train_list.txt 和 test_list.txt 两个文件
- 此时 cat_12_train文件夹 和 train_list.txt 和 test_list.txt 就是我们需要的数据（测试数据集和训练数据集都存放在cat_12_train内）

<img src="https://ai-studio-static-online.cdn.bcebos.com/23c26e7c313b4a8db2e0f33a5c3a3efa87dda6a0890a4d5fa8142aaed02b96c4" width = "800" height = "400" align=center />

### 可能出现的问题：数据集内图片出错

- 数据集内的图片看似完好，都可以查看，而实际训练的时候，可能会出现dataloader错误，原因在于读取的图片位深是8，一般三通道的图片位深是24，位深为8的图也可以是彩色图。

<img src="https://ai-studio-static-online.cdn.bcebos.com/2e2bf21520204cadbc81ffb13523979cdc628c6b84d24fb7b02e0bb889d5bc1f" width = "800" height = "400" align=center />

<br/>

- 如果出现以上图片问题，可以运行以下代码，找出坏图：

<br/>

```
import os
import cv2
import shutil
 
dirName = 'E:\PaddleClas-release-2.0\dataset\cat_12\cat_12_train'
# 将dirName路径下的所有文件路径全部存入all_path列表
all_path = []
for root, dirs, files in os.walk(dirName):
        for file in files:
            if "jpg" in file:
                    all_path.append(os.path.join(root, file))
all_path.sort()
 
bad = []
# 坏图片存放路径
badpath = 'E:\PaddleClas-release-2.0\dataset\\bad'
 
for i in range(len(all_path)):
    org = all_path[i]
    # print(all_path[i].split('/')[-1])
    try:
        img = cv2.imread(org)
        ss = img.shape
    except:
        bad.append(all_path[i])
        shutil.move(all_path[i],badpath)
        continue
 
print('共有%s张坏图'%(len(bad)))
print(bad)
```
<br/>

- 坏图片会自动移动到指定目录，使用该数据前，需要再标注文件中找出这些坏图片的标注并删掉，否则会出问题

<br/>

<img src="https://ai-studio-static-online.cdn.bcebos.com/f6687f966842433f9d07583790eeee85110625f5e7254b069db8068f109aca72" width = "800" height = "400" align=center />


<br/>

- 若使用自定义数据集训练，将数据集存放在dataset目录下即可

- 该目录下默认会有一个flowers102的文件夹

<img src="https://ai-studio-static-online.cdn.bcebos.com/373c1d708a4a4d90aa5e6f7437e7ce5b6104c0519603430aa683338b6959648c" width = "800" height = "400" align=center />

<br/>

<br/>

- 这次我们使用12猫分类数据集来进行训练
- 将cat12数据集移动到dataset目录下


<img src="https://ai-studio-static-online.cdn.bcebos.com/74c8a6551cec497898cfe7e99b0e8b6e7f2f6d963f68485ba435db93187dab6a" width = "800" height = "400" align=center />

- **注：数据集训练和测试都已经做好标注，命名为train_list.txt  test_list.txt**

### 2.2 设置PYTHONPATH环境变量

- 为什么设置pythonpath环境变量：只需设置PYTHONPATH，从而可以从正在用的目录（也就是正在交互模式下使用的当前目录，或者包含顶层文件的目录）以外的其他目录进行导入
- 数据集准备好后，进入anaconda（命令提示符，可以再win运行内输入anaconda找到）

<img src="https://ai-studio-static-online.cdn.bcebos.com/2bacf7c42d724eb1b322c5b775ececb39c96feee7a62465eb142e90273e83a09" width = "800" height = "400" align=center />

<br/>

- 激活anaconda环境（该环境已经再文章[Windows-paddle-深度学习环境搭建](http://)2.5章节中配置过）
- 进入paddleClas套件目录，设置环境变量

```
activate paddle
cd /d PaddleClas目录
set PYTHONPATH=./:$PYTHONPATH
```

<img src="https://ai-studio-static-online.cdn.bcebos.com/4f70f21afd464aecb061440dd4a835456fad20e33574489fb5d859c9014e10a8" width = "800" height = "400" align=center />

<br/>

### 2.3 下载预训练模型
- 使用预训练模型作为初始化，不仅加速训练，可以使用更少的训练epoch；预训练模型还可以避免陷入局部最优点或鞍点。
- 更多PaddleClas预训练模型关注[GitHub](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.0/docs/zh_CN/models/models_intro.md)哈
**通过tools/download.py下载所需要的预训练模型。**
**在cmd输入以下命令**
```
python tools/download.py -a ResNet50_vd -p ./pretrained -d True
python tools/download.py -a ResNet50_vd_ssld -p ./pretrained -d True
python tools/download.py -a MobileNetV3_large_x1_0 -p ./pretrained -d True
```
* **参数说明：**

- architecture（简写 a）：模型结构
- path（简写 p）：下载路径
- decompress （简写 d）：是否解压

<img src="https://ai-studio-static-online.cdn.bcebos.com/82c71f00a17648368c6d69dd10f5644c3be5eb1f201f4883a59425617062a92d" width = "800" height = "400" align=center />

<br/><br/>
- 待显示download .....finished  表示预训练模型完成下载
- 下载完成后不要关闭窗口

## 三、模型训练

### 3.1 训练前设置
 

#### **设置gpu卡号**
- 设置gpu卡号是为了指定使用哪张显卡训练，win下目前仅支持一张显卡训练，卡号设为0

```
set CUDA_VISIBLE_DEVICES=0
```

<img src="https://ai-studio-static-online.cdn.bcebos.com/853617e5de0849fab5f11e996e533e5cdd4eca111691409b9d796a8ae2ffb9bd" width = "800" height = "400" align=center />

<br/><br/>
#### **Windows下不支持多卡训练，如果有多卡请在在设备管理器（右键计算机图标）里面禁用未使用的显卡**

<br/>

<img src="https://ai-studio-static-online.cdn.bcebos.com/dc9db7b95eeb4927b4e498c866259c7678b8fd8874e54ccbbc31e07767bbd25b" width = "800" height = "400" align=center />
<br/><br/>

#### **注：在Windows系统下，不支持多进程读取数据，训练前，需要修改yaml文件**
#### **用记事本打开configs/quick_start 目录下 ResNet50_vd.yaml 文件**


<img src="https://ai-studio-static-online.cdn.bcebos.com/4e7ec9764dbe42209c3bb04f81f19873c1f62333226e4b13a3cc0b94c4127678" width = "800" height = "400" align=center />

<br/><br/>
#### **更改TRAIN模块和 VALID模块**
#### **num_workers 为0表示单进程读取数据**
#### **file_list  是标注文件，例如train_list.txt**
#### **data_dir  数据集所在目录**
#### **batch_size 可以根据自己gpu显存来调~**

<img src="https://ai-studio-static-online.cdn.bcebos.com/8b083a75d31a4f0fbc8f7550ae82e5e821605376118b4078a30750c9efc8c969" width = "800" height = "400" align=center />
<br/>

<img src="https://ai-studio-static-online.cdn.bcebos.com/b983ba341f3d40efae750415e6bed373889d68b2efc6457fb0a9c2a90b1c6fcf" width = "800" height = "400" align=center />
<br/>
### 3.2 零基础训练：不加载预训练模型的训练
#### **基于ResNet50_vd模型训练**
- **脚本：**

`Python tools/train.py -c ./configs/quick_start/ResNet50_vd.yaml`

<br/>

<img src="https://ai-studio-static-online.cdn.bcebos.com/97fc4672658d428b895b6648c9221c50b262ff05c5934379b5434dd5f73ea36a" width = "800" height = "400" align=center />
<br/>

#### **训练时间可能会有些长，期间注意不要关闭窗口，尽量不要做其他事（Windows很容易崩，不要问俺为什么）**

<br/>

#### **训练log为以下输出：**

<img src="https://ai-studio-static-online.cdn.bcebos.com/a3d2729b5e2e49c3b2aba0833c4e9348ea5c893f2202484695345cfcdbbe64f4" width = "800" height = "400" align=center />

<br/>

#### 待训练完成， 训练的模型将会存放在output目录里面，并以ppcls为名

<br/>

<img src="https://ai-studio-static-online.cdn.bcebos.com/2883e2111e504634ba8fc21870fbc25d5b210de28cc441788cd9aed9f30d1834" width = "800" height = "400" align=center />

<br/>

### 3.3 模型微调

根据自己的数据集路径设置好配置文件后，可以通过加载预训练模型的方式进行微调，如下所示。

```
python tools/train.py \
    -c ./configs/quick_start/ResNet50_vd.yaml \
    -o pretrained_model="./pretrained/MobileNetV3_large_x1_0_pretrained" \
    -o use_gpu=True
```

**其中-o pretrained_model用于设置加载预训练模型权重文件的地址，使用时需要换成自己的预训练模型权重文件的路径，也可以直接在配置文件中修改该路径。**

**我们也提供了大量基于ImageNet-1k数据集的预训练模型，模型列表及下载地址详见模型库概览。**

### 3.4 模型恢复训练

#### **如果训练任务因为其他原因被终止，也可以加载断点权重文件，继续训练：**

```
python tools/train.py \
    -c ./configs/quick_start/ResNet50_vd.yaml \
    -o checkpoints="./output/ResNet50_vd/10/ppcls" \
    -o last_epoch=10 \
    -o use_gpu=True
```

**其中配置文件不需要做任何修改，只需要在继续训练时设置checkpoints参数即可，表示加载的断点权重文件路径，使用该参数会同时加载保存的断点权重和学习率、优化器等信息。**

#### **注意：**

- -o last_epoch=5表示将上一次训练轮次数记为10，即本次训练轮次数从11开始计算，该值默认为-1，表示本次训练轮次数从0开始计算。

- -o checkpoints参数无需包含断点权重文件的后缀名，上述训练命令会在训练过程中生成如下所示的断点权重文件，若想从断点5继续训练，则checkpoints参数只需设置为"./output/MobileNetV3_large_x1_0_gpupaddle/5/ppcls"，PaddleClas会自动补充后缀名。

### 3.5 模型评估
#### 同样，在评估前设置一下我们的eval.yaml文件
#### 在configs目录下用记事本打开eval.yaml将圈出的改为以下内容

<img src="https://ai-studio-static-online.cdn.bcebos.com/cd2a522ed359417798e0d169dfa250905e38d8425bcb41c88cea85a17319ca6c" width = "800" height = "400" align=center />
<br/>

#### 说明：
- num_workers设置成0表示使用单进程
- file_list为测试标注文件
- data_dir为测试集

#### **输入评估命令：**


```
python -m paddle.distributed.launch 
	--selected_gpus="0" tools/eval.py 
   -c ./configs/eval.yaml 
   -o ARCHITECTURE.name="ResNet50_vd" 
   -o pretrained_model="./output/ResNet50_vd/best_model/ppcls"
```

<br/>

#### **说明：**
- -m 将模块当作脚本运行
- -selected_gpus 选择显卡训练，如果使用多卡，可以一起写，用英文逗号隔开
- -c 配置文件路径
- -o 更改配置文件内 配置

<img src="https://ai-studio-static-online.cdn.bcebos.com/29e289b4e6a54af4bbd81775065f022c93a23901d6dd4de1baf414eb37cb0433" width = "800" height = "400" align=center />
<br/>

#### **待评估完成，输出log如下**

<img src="https://ai-studio-static-online.cdn.bcebos.com/8a39032b8f75424ba895bb02f9594cb4de3d628a76d44b2f9298f063fea75dd0" width = "800" height = "400" align=center />
<br/>

####  **训练时只训练一会儿，只用来做演示，还可以通过调学习率，epoch等来增加准确值**

<br/><br/>


## 四、模型推理
### 4.1 分类预测框架简介：
#### **Paddle 的模型保存有多种不同的形式，大体可分为两类：**

#### **1.persistable 模型（fluid.save_persistabels保存的模型）一般做为模型的 checkpoint，也就是我们训练保存的模型，可以加载后重新训练。persistable 模型保存的是零散的权重文件，每个文件代表模型中的一个 Variable，这些零散的文件不包含结构信息，需要结合模型的结构一起使用。**

#### **2.inference 模型（fluid.io.save_inference_model保存的模型） 一般是模型训练完成后保存的固化模型，用于预测部署。与 persistable 模型相比，inference 模型会额外保存模型的结构信息，用于配合权重文件构成完整的模型。**


#### **PaddlePaddle提供三种方式进行预测推理，接下来介绍如何用预测引擎进行推理：**
### 4.2 对训练好的模型进行转换：
#### **参数说明：**
- -m=模型名称，
- -p=之前训练的模型路径
- -o=转换后模型保存的路径
- --class_dim=类别数量
```
python tools/export_model.py \
		--m=ResNet50_vd \
 		--p=”./output/ResNet50_vd/best_model/ppcls” \
 		--o=”./output/ResNet50_vd/best_model/mp” \
		--class_dim=12
```

<img src="https://ai-studio-static-online.cdn.bcebos.com/1fc080c01a57443287b99854f20685eabc29ccec72664beb9ae5d76c06a70f46" width = "800" height = "400" align=center />
<br/>

#### **以上我们将训练好的模型转换完毕，保存在/output/ResNet50_vd/best_model/mp文件及里**

<img src="https://ai-studio-static-online.cdn.bcebos.com/f53bd0211a694473a14ddd15c11808af96364a04d23643e39e1451fc70207844" width = "800" height = "400" align=center />

<br/>

### 4.3 预测引擎 + inference 模型预测
#### **参数说明：**
- -image_file：需要预测的图片路径  
- -model_file为4.2章节转换后保存的模型结构信息model  
- -params_file为权重文件variables
```
python ./tools/infer/predict.py --image_file ./dataset/cat_12/cat_12_test/zyQNTEXeFRu7prgPVaIJOSH5tGfb6n34.jpg \
    --model_file "./output/ResNet50_vd/best_model/mp/inference.pdmodel" \
    --params_file "./output/ResNet50_vd/best_model/mp/inference.pdiparams" \
    --use_gpu=True \
    --use_tensorrt=False
```

<img src="https://ai-studio-static-online.cdn.bcebos.com/10b09b70c58c41e78aab4548a398e5464b2b3673b86b480a828d84228175e11a" width = "800" height = "400" align=center />

<br/>

### 4.4 训练引擎 + persistable 模型预测 （预训练模型）
#### **参数说明：**
- -i：需要预测的图片路径  
- -model 模型名称 
- -pretrained_model  训练后保存的模型
- -use_gpu  是否使用显卡预测
- -class_num  类型数量
```
python tools/infer/infer.py -i ./dataset/cat_12/cat_12_test/0aSixIFj9X73z41LMDUQ6ZykwnBA5YJW.jpg \
                   --model ResNet50_vd \
                   --pretrained_model "./output/ResNet50_vd/best_model/ppcls" \
                   --use_gpu True \
                   --load_static_weights False \
                   --class_num=12
```


<img src="https://ai-studio-static-online.cdn.bcebos.com/ee81f347fad54e3394c742d5ca6d331283878b704f8141ef80fc2e305adaddbf" width = "800" height = "400" align=center />

<br/>

#### 注:
- 运行脚本时，注意当前目录，代码是相对目录写的，不是绝对目录，所以如果当前不是PaddleClas目录，可能会报错
- 使用自定义数据集时，注意标注文件要用空格隔开文件名和标注信息，制表符可能会出错
- 数据集内图片需要使用cv2.imread 读取下，因为有的图片会正常显示，但用来训练时会读取成None（原因就在于读取的图片位深是8。一般三通道的图片位深是24。位深为8的图也可以是彩色图。）
- 此例是笔记本训练的，训练轮数不多，准确率不高，只做演示哈

# 部署
#### **完成以上任务后，接下来会展示如何部署：**
[部署二（基于Hub的serving服务部署）](https://aistudio.baidu.com/aistudio/projectdetail/1585531)
&nbsp; 

[部署三（生成exe文件部署）](https://aistudio.baidu.com/aistudio/projectdetail/1184186)
&nbsp; 

[部署四（Python/C#调用DLL部署，可传参）](https://aistudio.baidu.com/aistudio/projectdetail/1589564)

### **如果在操作过程中有任何问题欢迎在评论区提出，或者在[PaddleClas-Github](https://github.com/PaddlePaddle/PaddleClas)提issue**


