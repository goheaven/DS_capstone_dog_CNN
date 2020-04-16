# 数据科学家 Capstone Project: 为小狗识别应用编写算法

### Table of Contents

1. [项目概述](#ProjectOverview)
2. [项目模块](#ProjectComponents)
3. [文件描述](#FileDescription)
4. [需要的库](#Requirements)
5. [其他指示](#Instructions)

## 1. 项目概述 <a name="ProjectOverview"></a>
一种可用于移动应用或网络应用的算法。
代码将能够接受任何用户提供的图像作为输入。
如果从图像中检测出小狗，该算法将大致识别出小狗品种。
如果检测出人脸，该算法将大致识别出最相似的小狗品种。

### 2. 项目模块 <a name="ProjectComponents"></a>

- 导入数据集
- 检测人脸
- 检测小狗
- 从头开始创建分类小狗品种的 CNN
- 使用迁移学习分类小狗品种的 CNN
- 使用迁移学习创建分类小狗品种的 CNN
- 编写算法
- 测试算法

## 3. 文件描述 <a name="FileDescription"></a>
<pre>
DS_capstone_dog_CNN
|   README.md
|   dog_app-zh.html
|   dog_app-zh.ipynb     # notebook to process data
|   dog_app-zh.py
|   extract_bottleneck_features.py
|   requirements.txt
|   
+---bottleneck_features          //store pre-trained bottleneck feature files
|           
+---data                         //store training image files
|       
+---haarcascades                 //store haar cascade xml files    
|   haarcascade_frontalface_alt.xml  
|  
+---images                       //sample and test images
|  
|  
\---saved_models
        weights.best.from_scratch.hdf5      //CNN trained best weight
        weights.best.VGG16.hdf5
        weights.best.Xception.hdf5
</pre>        


## 4. 需要的库 <a name="Requirements"></a>    

All of the requirements are captured in requirements.txt.  
```python
pip install -r requirements.txt
```

## 5. 其他指示 <a name="Instructions"></a>
1. 关于项目的blog文章地址：

2. 项目通过dog_app-zh.ipynb展示或者dog_app-zh.html展示最终结果

