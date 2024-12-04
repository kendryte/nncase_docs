import numpy as np
import random
import cv2
import nncaseruntime_k230 as nn

confidence_thres, iou_thres = 0.1, 0.2
random.seed(0)

CLASS_COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]
CLASS_NAMES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
def main():
    #########################################
    # 读取图片，并转换为AI2D输入的格式，
    # 当前demo没有使用摄像头作为输入，
    # 因此在这里做一定程度上的模拟
    #########################################
    img = cv2.imread('input.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_nchw = img_rgb.transpose((2, 0, 1))[np.newaxis]
    img_nchw = img_nchw.astype(np.uint8)
    #########################################
    
    
    kpu = nn.KPU()
    ai2d = nn.AI2D()
    
    kpu.load_model("test.kmodel")
    tmp_tensor = nn.RuntimeTensor.from_numpy(np.ones((1,3,640,640),dtype=np.uint8))
    kpu.set_input_tensor(0, tmp_tensor)

    kpu_input_tensor = kpu.get_input_tensor(0)

    #########################################
    # 预处理参数计算
    # img_nchw：[1,3,h,w]
    # 输入模型尺寸：[1,3,640,640]
    #########################################
    input_height, input_width = 640, 640
    img_height, img_width = img_nchw.shape[2], img_nchw.shape[3]
    
    # 计算缩放比例
    r = min(input_width/img_width, input_height / img_height)
    new_unpad = int(round(img_width * r)), int(round(img_height * r))
    
    # 计算填充
    dw, dh = (input_width - new_unpad[0]) / 2, (input_height - new_unpad[1]) / 2
    
    # 计算缩放比例
    ratio = r, r
    pad_h_before = int(round(dh/2))
    pad_h_after = int(dh - pad_h_before)
    pad_w_before = int(round(dw/2))
    pad_w_after = int(dw - pad_w_before)    
    print(pad_h_before,pad_h_after,pad_w_before,pad_w_after)
    
    #########################################
    # ai2d同样支持配置letterbox功能，通过resize+pad实现，
    # 这里不需要resize的目标shape，
    # ai2d会通过pad_param和build时传入的第二个shape参数自动计算
    #########################################

    ai2d.set_datatype(nn.AI2D_FORMAT.NCHW_FMT, nn.AI2D_FORMAT.NCHW_FMT,
                      np.uint8, np.uint8)
    ai2d.set_resize_param(True,
                          nn.AI2D_INTERP_METHOD.tf_bilinear,
                          nn.AI2D_INTERP_MODE.half_pixel)
    ai2d.set_pad_param(True,
                       [0,0,0,0, pad_h_before,pad_h_after, pad_w_before,pad_w_after],
                       0,
                       [114,114,114])
    ai2d.build([1,3,img_nchw.shape[2],img_nchw.shape[3]],
               [1,3,640,640])   
    ai2d_input_tensor = nn.RuntimeTensor.from_numpy(img_nchw)

    #########################################
    # 如果你不知道编译kmodel的时候该使用什么数据进行校正，
    # 将你的原始图片作为输入，运行ai2d，然后保存输出
    # calib_data = kpu_input_tensor.to_numpy()   <kpu_input_tensor 就是ai2d的输出>
    # cv2.imwrite('calib_data_{}.jpg'.format(i), calib_data) <如果看不懂这句代码，后续流程请放弃>
    # 在编译kmodel时，保存下来的图片读取出来，作为校正数据，
    # 校正集格式参考
    # https://github.com/kendryte/nncase/blob/master/examples/user_guide/k230_simulate-ZH.ipynb
    # 校正集的数量为2   
    # calib_data = [[np.random.rand(1, 240, 320, 3).astype(np.float32), np.random.rand(1, 240, 320, 3).astype(np.float32)]]
    # 同时编译kmodel时的前处理参数只需要配置input_shape为你的模型输入，输入类型为uint8，input_range为[0,255]或者[0,1]<这取决于你模型训练时输入的范围,当前demo为yolo11, input_range为[0,1]>,其他参数按照模型中的信息配置即可
    #########################################

    ai2d.run(ai2d_input_tensor, kpu_input_tensor)
    kpu.run()

    #########################################
    # 获取模型输出、进行后处理
    #########################################
    results = []
    for i in range(kpu.outputs_size):
        result = kpu.get_output_tensor(i).to_numpy()
        results.append(result)

    output_image = postprocess(img, results[0], dw, dh,ratio)
    cv2.imwrite('output.jpg', output_image)



def draw_detections(img, box, score, class_id):
        """
        在输入图像上绘制检测到的边界框和标签。
        参数：
            img: 用于绘制检测结果的输入图像。
            box: 检测到的边界框。
            score: 对应的检测分数。
            class_id: 检测到的目标类别 ID。
        
        返回：
            None
        """
        # 提取边界框的坐标
        x1, y1, w, h = box
 
        # 获取类别对应的颜色
        color = CLASS_COLORS[class_id]
 
        # 在图像上绘制边界框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
 
        # 创建包含类别名和分数的标签文本
        label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
 
        # 计算标签文本的尺寸
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
 
        # 计算标签文本的位置
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
 
        # 绘制填充的矩形作为标签文本的背景
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
 
        # 在图像上绘制标签文本
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def nms(boxes, scores, iou_threshold):
    """
    使用NumPy实现非极大值抑制
    参数：
        boxes: 边界框列表 [x, y, w, h]
        scores: 置信度分数列表
        iou_threshold: IoU阈值
    返回：
        保留的边界框索引列表
    """
    scores = np.array(scores)
    # 转换boxes格式为[x1, y1, x2, y2]
    x1 = np.array([box[0] for box in boxes])
    y1 = np.array([box[1] for box in boxes])
    x2 = np.array([box[0] + box[2] for box in boxes])
    y2 = np.array([box[1] + box[3] for box in boxes])
    
    # 计算面积
    areas = (x2 - x1) * (y2 - y1)
    
    # 按分数排序的索引
    order = scores.argsort()[::-1]
    print(order)
    keep = []
    while order.size > 0:
        # 保留分数最高的框
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
            
        # 计算IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # 保留IoU小于阈值的框
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep

def postprocess(input_image, output, dw, dh,ratio):
        """
        对模型输出进行后处理，以提取边界框、分数和类别 ID。
        参数：
            input_image (numpy.ndarray): 输入图像。
            output (numpy.ndarray): 模型的输出。
        返回：
            numpy.ndarray: 包含检测结果的输入图像。
        """
        # 转置并压缩输出，以匹配预期形状
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        print("rows:",rows)
        boxes, scores, class_ids = [], [], []
 
        # 计算缩放比例和填充
        # ratio = img_width / input_width, img_height / input_height
 
        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            # print("max_score:",max_score)
            if max_score >= confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
 
                # 将框调整到原始图像尺寸，考虑缩放和填充
                x -= dw  # 移除填充
                y -= dh    
                x /= ratio[0]  # 缩放回原图
                y /= ratio[1]
                w /= ratio[0]
                h /= ratio[1]
                left = int(x - w / 2)
                top = int(y - h / 2)
                width = int(w)
                height = int(h)
 
                boxes.append([left, top, width, height])
                scores.append(max_score)
                class_ids.append(class_id)
 
        indices = nms(boxes, scores, iou_thres)
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            print(i)
            draw_detections(input_image, box, score, class_id)
        return input_image

if __name__ == '__main__':
    main()