# demo 说明

1. 当前demo没有使用摄像头作为输入，通过cv2读取图片，并转换为AI2D输入使用最多的格式NCHW_FMT

2. ai2d同样支持配置letterbox功能，通过resize+pad实现，这里不需要resize的目标shape，ai2d会通过pad_param和build时传入的第二个shape参数自动计算,只需要计算准确的pad_param即可

3. 重点！！！
很多同学不知道编译kmodel的时候量化校正集该使用什么数据
那么在这里，当你正确配置了ai2d的参数后，ai2d的输出，就是你需要的校正集
以当前demo为例，标准官方yolo11n模型(刚出时候的版本)，输入[1，3，640，640]，输入类型为float32，input_range为[0,1]
前处理基本上就只有letterbox
那么这里按照demo中的流程配置参数，就是letterbox的功能,resize+pad
将你的原始图片作为输入，运行ai2d，收集ai2d的输出(结合demo内代码看)
```
calib_data = kpu_input_tensor.to_numpy()
cv2.imwrite('calib_data_{}.jpg'.format(i), calib_data)
```
在编译kmodel时，将保存下来的图片读取出来，作为校正数据，

校正集格式参考
https://github.com/kendryte/nncase/blob/master/examples/user_guide/k230_simulate-ZH.ipynb
```
calib_data = [[np.random.rand(1, 240, 320, 3).astype(np.float32), np.random.rand(1, 240, 320, 3).astype(np.float32)]]
```
即：校正集的数量为2，校正集的格式为`[[校正图片1，校正图片2]]`
想放多少放多少，适量即可，多了内存会爆哦

同时编译kmodel时的前处理参数只需要配置input_shape为你的模型输入，输入类型为uint8，input_range为[0,1]
这取决于你模型训练时输入的范围,当前demo为yolo11, input_range为[0,1],其他模型按照训练时输入的range配置。

> demo就是demo，别想着demo能跑通你的模型，demo只是给你一个参考，模型输入输出格式，前处理后处理，都是需要你根据自己实际情况配置的。