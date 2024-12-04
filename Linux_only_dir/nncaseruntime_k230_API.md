# nncase K230 运行时库 Python API 文档

## 目录

[toc]

## 1. 概述

`nncaseruntime_k230` 是 nncase 运行时库的 Python 接口，专门为 K230 平台提供 AI功能（KPU）和图像处理功能（AI2D）。

## 2. 基础数据结构

### 2.1 RuntimeTensor

适用于nncaseruntime_k230的数据类型，用于KPU和AI2D的输入、输出。

#### 2.1.1 from_numpy

从numpy对象创建一个RuntimeTensor

```Python
>>> import numpy as np
>>> import nncaseruntime_k230
>>> data = np.zeros([1,3,320,320],dtype=np.uint8).reshape((1,3,320,320))
>>> tensorA = nncaseruntime_k230.RuntimeTensor.from_numpy(data)
>>> tensorA
<nncaseruntime_k230._nncaseruntime_k230.RuntimeTensor object at 0x3fb0acbd70>
```

#### 2.1.2 to_numpy

将RuntimeTensor转换为numpy对象

```Python
>>> dataA = tensorA.to_numpy()
```

#### 2.1.3 dtype

返回RuntimeTensor的数据类型

```py
>>> tensorA.dtype
dtype('uint8')
```

#### 2.1.4 shape

返回RuntimeTensor的数据形状

```py
>>> tensorA
<nncaseruntime_k230._nncaseruntime_k230.RuntimeTensor object at 0x3fb0acbd70>
>>> tensorA.shape 
[1, 3, 320, 320]
```

### 2.2 TensorDesc

用于描述RuntimeTensor的信息。

#### 2.2.1 dtype

RuntimeTensor的类型信息。

```
>>> tensor = kpu.get_input_desc(0) # 需要确保kpu已经调用过set_input_tensor(0)
>>> tensor.dtype
dtype('uint8')
```

#### 2.2.2 start

RuntimeTensor的起始偏移字节数。

```
>>> tensor.start
0
```

#### 2.2.3 size

RuntimeTensor的占用字节数。

```
>>> tensor.size
307200
```

## 3 基础属性

### 3.1 AI2D_FORMAT

图像格式枚举，支持多种图像数据格式

- `YUV420_NV12`：YUV420 NV12 格式
- `YUV420_NV21`：YUV420 NV21 格式
- `YUV420_I420`：YUV420 I420 格式
- `NCHW_FMT`：NCHW 格式
- `RGB_packed`：RGB 格式
- `RAW16`：RAW16 格式

### 3.2 AI2D_PAD_MODE

图像填充模式枚举

- `constant`：常数填充，目前仅支持该方式
- `copy`：复制填充
- `mirror`：镜像填充

### 3.3 AI2D_INTERP_METHOD

图像插值方法枚举

- `tf_nearest`：TensorFlow 最近邻插值
- `tf_bilinear`：TensorFlow 双线性插值

### 3.4 AI2D_INTERP_MODE

图像插值模式枚举

- `none`：无插值
- `align_corner`：对齐角落的插值
- `half_pixel`：半像素插值

## 4. KPU模块

KPU 模块提供了用于调用 KPU 硬件执行神经网络模型推理的基本功能，包括加载模型、设置输入数据、执行推理及获取输出结果等。
使用之前需要初始化KPU对象。

```
import nncaseruntime_k230
kpu = nncaseruntime_k230.KPU()
```

### 4.1 load_model

**描述**

加载编译生成的 kmodel 格式的神经网络模型。

**语法**

```python
# 1.
with open("/path/to/test.kmodel", "r") as f:
    model_data = f.read())
    load_model(model_data)
# 2.
model_path = "/path/to/test.kmodel"
load_model(model_path)
```

**参数**

| 参数名称   | 描述                | 输入/输出 |
| ---------- | ------------------- | --------- |
| model_data | kmodel 的二进制内容 | 输入      |
| model_path | kmodel 的文件路径   | 输入      |

**返回值**

| 返回值 | 描述                        |
| ------ | --------------------------- |
| 无     | 加载成功。                  |
| 其他   | 如果失败，将抛出 C++ 异常。 |

### 4.2 set_input_tensor

**描述**

设置 kmodel 推理时的输入 runtime_tensor。

**语法**

```python
set_input_tensor(index, runtime_tensor)
```

**参数**

| 参数名称      | 描述                      | 输入/输出 |
| ------------- | ------------------------- | --------- |
| index         | kmodel 的输入索引         | 输入      |
| RuntimeTensor | 包含输入数据信息的 tensor | 输入      |

**返回值**

| 返回值 | 描述                        |
| ------ | --------------------------- |
| 无     | 设置成功。                  |
| 其他   | 如果失败，将抛出 C++ 异常。 |

### 4.3 get_input_tensor

**描述**

获取 kmodel 推理时的输入 runtime_tensor。

**语法**

```python
get_input_tensor(index)
```

**参数**

| 参数名称 | 描述              | 输入/输出 |
| -------- | ----------------- | --------- |
| index    | kmodel 的输入索引 | 输入      |

**返回值**

| 返回值        | 描述                        |
| ------------- | --------------------------- |
| RuntimeTensor | 包含输入数据信息的 tensor。 |
| 其他          | 如果失败，将抛出 C++ 异常。 |

### 4.4 set_output_tensor

**描述**

设置 kmodel 推理后的输出结果。

**语法**

```python
set_output_tensor(index, runtime_tensor)
```

**参数**

| 参数名称      | 描述              | 输入/输出 |
| ------------- | ----------------- | --------- |
| index         | kmodel 的输出索引 | 输入      |
| RuntimeTensor | 输出结果的 tensor | 输入      |

**返回值**

| 返回值 | 描述                        |
| ------ | --------------------------- |
| 无     | 设置成功。                  |
| 其他   | 如果失败，将抛出 C++ 异常。 |

### 4.5 get_output_tensor

**描述**

获取 kmodel 推理后的输出结果。

**语法**

```python
get_output_tensor(index)
```

**参数**

| 参数名称 | 描述              | 输入/输出 |
| -------- | ----------------- | --------- |
| index    | kmodel 的输出索引 | 输入      |

**返回值**

| 返回值        | 描述                                   |
| :------------ | -------------------------------------- |
| RuntimeTensor | 获取第 index 个输出的 runtime_tensor。 |
| 其他          | 如果失败，将抛出 C++ 异常。            |

### 4.6 run

**描述**

启动 kmodel 推理过程。

**语法**

```python
run()
```

**返回值**

| 返回值 | 描述                        |
| :----- | --------------------------- |
| 无     | 推理成功                    |
| 其他   | 如果失败，将抛出 C++ 异常。 |

### 4.7 get_input_desc

**描述**

获取指定索引的输入描述信息。

**语法**

```python
get_input_desc(index)
```

**参数**

| 参数名称 | 描述              | 输入/输出 |
| -------- | ----------------- | --------- |
| index    | kmodel 的输入索引 | 输入      |

**返回值**

| 返回值     | 描述                                                         |
| :--------- | ------------------------------------------------------------ |
| TensorDesc | 第 index 个输入的信息，包括 `dtype`, `start`, `size`。 |

### 4.8 get_output_desc

**描述**

获取指定索引的输出描述信息。

**语法**

```python
get_output_desc(index)
```

**参数**

| 参数名称 | 描述              | 输入/输出 |
| -------- | ----------------- | --------- |
| index    | kmodel 的输出索引 | 输入      |

**返回值**

| 返回值     | 描述                                                         |
| :--------- | ------------------------------------------------------------ |
| TensorDesc | 第 index 个输出的信息，包括 `dtype`, `start`, `size`。 |

### 4.9 属性

#### 4.9.1 inputs_size

**描述**

kmodel 的输入个数。

#### 4.9.2 outputs_size

**描述**

kmodel 的输出个数。

## 5. AI2D 模块

AI2D 类提供了丰富的图像处理功能，支持数据类型转换、裁剪、填充、缩放和仿射变换。
使用之前需要初始化AI2D对象。

```
import nncaseruntime_k230
ai2d = nncaseruntime_k230.AI2D()
```

### 5.1 set_dtype

**描述**

设置 AI2D 计算过程中的数据类型。

**语法**

```python
set_dtype(src_format, dst_format, src_type, dst_type)
```

**参数**

| 名称       | 类型        | 描述         |
| ---------- | ----------- | ------------ |
| src_format | AI2D_FORMAT | 输入数据格式 |
| dst_format | AI2D_FORMAT | 输出数据格式 |
| src_type   | np.dtype    | 输入数据类型 |
| dst_type   | np.dtype    | 输出数据类型 |

### 5.2 set_crop_param

**描述**

配置裁剪相关参数。

**语法**

```python
set_crop_param(crop_flag, start_x, start_y, width, height)
```

**参数**

| 名称      |  类型 |      描述           |
| --------- | ---- | ------------------|
| crop_flag | bool | 是否启用裁剪功能     |
| start_x   | int  | 宽度方向的起始像素   |
| start_y   | int  | 高度方向的起始像素   |
| width     | int  | 宽度               |
| height    | int  | 高度               |

### 5.3 set_shift_param

【描述】

用于配置shift相关的参数.

【定义】

```Python
set_shift_param(shift_flag, shift_val)
```

【参数】

| 名称       | 类型 | 描述              |
| ---------- | ---- | ----------------- |
| shift_flag | bool | 是否开启shift功能 |
| shift_val  | int  | 右移的比特数      |

### 5.4 set_pad_param

【描述】

用于配置pad相关的参数.

【定义】

```Python
set_pad_param(pad_flag, paddings, pad_mode, pad_val)
```

【参数】

| 名称     | 类型 | 描述                                                                                          |
| -------- | ---- | --------------------------------------------------------------------------------------------- |
| pad_flag | bool | 是否开启pad功能                                                                               |
| paddings | list | 各个维度的padding, size=8，分别表示dim0到dim4的前后padding的个数，其中dim0/dim1固定配置{0, 0} |
| pad_mode | int  | 只支持pad constant，配置0即可                                                                 |
| pad_val  | list | 每个channel的padding value                                                                    |

### 5.5 set_resize_param

【描述】

用于配置resize相关的参数.

【定义】

```Python
set_resize_param(resize_flag, ai2d_interp_method, ai2d_interp_mode)
```

【参数】

| 名称               | 类型               | 描述               |
| ------------------ | ------------------ | ------------------ |
| resize_flag        | bool               | 是否开启resize功能 |
| ai2d_interp_method | AI2D_INTERP_METHOD | resize插值方法     |
| ai2d_interp_mode   | AI2D_INTERP_MODE   | resize模式         |

### 5.6 set_affine_param

【描述】

用于配置affine相关的参数.

【定义】

```Python
set_affine_param(affine_flag, ai2d_interp_method, cord_round, bound_ind, bound_val, bound_smooth, M)
```

【参数】

| 名称               | 类型               | 描述                                                                                                                     |
| ------------------ | ------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| affine_flag        | bool               | 是否开启affine功能                                                                                                       |
| ai2d_interp_method | AI2D_INTERP_METHOD | Affine采用的插值方法                                                                                                     |
| cord_round         | uint32_t           | 整数边界0或者1                                                                                                           |
| bound_ind          | uint32_t           | 边界像素模式0或者1                                                                                                       |
| bound_val          | uint32_t           | 边界填充值                                                                                                               |
| bound_smooth       | uint32_t           | 边界平滑0或者1                                                                                                           |
| M                  | list               | 仿射变换矩阵对应的vector，仿射变换为Y=\[a_0, a_1; a_2, a_3\] \cdot  X + \[b_0, b_1\] $, 则  M=[a_0,a_1,b_0,a_2,a_3,b_1 ] |

### 5.7 build

**描述**

构建 `AI2D`运行策略

**语法**

```python
build(input_shape, output_shape)
```

**参数**

| 名称         | 描述     |
| ------------ | -------- |
| input_shape  | 输入形状 |
| output_shape | 输出形状 |

**返回值**

| 返回值 | 描述 |
| :----- | ---- |
| 无     | 无   |

### 5.8 run

**描述**

配置寄存器并启动 AI2D 计算。

**语法**

```python
run(input_tensor, output_tensor)
```

**参数**

| 名称          | 描述        |
| ------------- | ----------- |
| input_tensor  | 输入 tensor |
| output_tensor | 输出 tensor |

**返回值**

| 返回值 | 描述                        |
| ------ | --------------------------- |
| 无     | 成功。                      |
| 其他   | 如果失败，将抛出 C++ 异常。 |

## 6. 示例

代码需要封装到函数后再运行，同时尽量少定义全局变量，否则可能导致程序结束后内存没有释放干净。

```py
import nncaseruntime_k230 as nn
import numpy as np
import cv2

def pipeline():
    kmodel_path="/sdcard/examples/kmodel/face_detection_320.kmodel"
    kpu=nn.KPU()
    ai2d=nn.AI2D()
    
    kpu.load_model(kmodel_path)
    
    # prepare kpu input for pipeline
    tmp_tensor = nn.RuntimeTensor.from_numpy(np.ones((1,3,320,320),dtype=np.uint8))
    kpu.set_input_tensor(0, tmp_tensor)
    
    ai2d_output_tensor = kpu_input_tensor = kpu.get_input_tensor(0)
    
    # data prepare
    img = cv2.imread('input.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_nchw = img_rgb.transpose((2, 0, 1))[np.newaxis]
    img_nchw = img_nchw.astype(np.uint8)
    
    # AI2D config, output image is 320*320
    ai2d.set_dtype(nn.AI2D_FORMAT.NCHW_FMT, nn.AI2D_FORMAT.NCHW_FMT, np.uint8, np.uint8)
    ai2d.set_resize_param(True,nn.AI2D_INTERP_METHOD.tf_bilinear, nn.AI2D_INTERP_MODE.half_pixel)
    ai2d.build([1,3,img_chw.shape[1],img_chw.shape[2]], [1,3,320,320])   
    ai2d_input_tensor = nn.RuntimeTensor.from_numpy(img_nchw)
    
    # infer
    ai2d.run(ai2d_input_tensor, ai2d_output_tensor)
    kpu.run()
    
    # get result
    for i in range(kpu.outputs_size):
        output_data = kpu.get_output_tensor(i)
        result = output_data.to_numpy()
        print(result.shape)

if __name__ == '__main__':
    pipeline()

```
