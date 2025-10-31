import numpy as np
import pyrealsense2 as rs
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import (check_img_size, non_max_suppression, scale_coords, set_logging)
from utils.torch_utils import select_device, time_sync
import torch
import torch.backends.cudnn as cudnn
import yaml
import cv2
import random

pipeline = rs.pipeline()  # 定义流程pipeline
config = rs.config()  # 定义配置config
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)  # 848 X 480这里。1280,720   640,480
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)  # 流程开始
align_to = rs.stream.color  # 与color流对齐
align = rs.align(align_to)
######启动相机


######对齐彩色与深度，获取内参，深度彩色图像
def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧
    depth = frames.get_depth_frame()
    aligned_frames = align.process(frames)  # 获取对齐帧
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

    intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile(
    ).intrinsics

    # 获取深度参数（像素坐标系转相机坐标系会用到）
    '''camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
						 'ppx': intr.ppx, 'ppy': intr.ppy,
						 'height': intr.height, 'width': intr.width,
						 'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
						 }'''

    depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
    color_image = np.asanyarray(color_frame.get_data())  # RGB图

    # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame, depth

#######以下都是yolo的部分
class YoloV5:
    def __init__(self, yolov5_yaml_path='models/yolov5s.yaml'):
        '''初始化'''
        # 载入配置文件
        with open(yolov5_yaml_path, 'r', encoding='utf-8') as f:
            self.yolov5 = yaml.load(f.read(), Loader=yaml.SafeLoader)
        # 随机生成每个类别的颜色
        # self.colors = [[np.random.randint(0, 255) for _ in range(
        #     3)] for class_id in range(self.yolov5['class_num'])]
        self.colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        # 模型初始化
        self.init_model()

    @torch.no_grad()
    def init_model(self):
        '''模型初始化'''
        # 设置日志输出
        set_logging()
        # 选择计算设备
        device = select_device(self.yolov5['device'])
        # 如果是GPU则使用半精度浮点数 F16
        is_half = device.type != 'cpu'
        # 载入模型
        model = attempt_load(
            self.yolov5['weight'], map_location=device)  # 载入全精度浮点数的模型
        input_size = check_img_size(
            self.yolov5['input_size'], s=model.stride.max())  # 检查模型的尺寸
        if is_half:
            model.half()  # 将模型转换为半精度
        # 设置BenchMark，加速固定图像的尺寸的推理
        cudnn.benchmark = True
        # 图像缓冲区初始化
        img_torch = torch.zeros(
            (1, 3, self.yolov5['input_size'], self.yolov5['input_size']), device=device)  # init img
        # 创建模型

        _ = model(img_torch.half()
                  if is_half else img_torch) if device.type != 'cpu' else None
        self.is_half = is_half  # 是否开启半精度
        self.device = device  # 计算设备
        self.model = model  # Yolov5模型
        self.img_torch = img_torch  # 图像缓冲区

    def preprocessing(self, img):
        '''图像预处理'''
        # 图像缩放

        img_resize = letterbox(img, new_shape=(
            self.yolov5['input_size'], self.yolov5['input_size']), auto=False)[0]
        # print("img resize shape: {}".format(img_resize.shape))

        img_arr = np.stack([img_resize], 0)

        img_arr = img_arr[:, :, :, ::-1].transpose(0, 3, 1, 2)

        img_arr = np.ascontiguousarray(img_arr)
        return img_arr

    @torch.no_grad()
    def detect(self, img, canvas=None, view_img=True):
        '''模型预测'''
        # 图像预处理
        img_resize = self.preprocessing(img)
        self.img_torch = torch.from_numpy(img_resize).to(self.device)
        self.img_torch = self.img_torch.half(
        ) if self.is_half else self.img_torch.float()
        self.img_torch /= 255.0
        if self.img_torch.ndimension() == 3:
            self.img_torch = self.img_torch.unsqueeze(0)
        # 模型推理
        t1 = time_sync()
        pred = self.model(self.img_torch, augment=False)[0]
        # pred = self.model_trt(self.img_torch, augment=False)[0]
        # NMS 非极大值抑制
        pred = non_max_suppression(pred, self.yolov5['threshold']['confidence'],
                                   self.yolov5['threshold']['iou'], classes=None, agnostic=False)
        t2 = time_sync()
        print("推理时间: inference period = {}".format(t2 - t1))
        # 获取检测结果
        det = pred[0]
        gain_whwh = torch.tensor(img.shape)[[1, 0, 1, 0]]  # [w, h, w, h]

        if view_img and canvas is None:
            canvas = np.copy(img)
        xyxy_list = []
        conf_list = []
        class_id_list = []
        if det is not None and len(det):
            # 画面中存在目标对象
            # 将坐标信息恢复到原始图像的尺寸
            det[:, :4] = scale_coords(
                img_resize.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, class_id in reversed(det):
                class_id = int(class_id)
                xyxy_list.append(xyxy)
                conf_list.append(conf)
                if view_img:
                    # 绘制矩形框与标签
                    label = '%s %.2f' % (
                        self.yolov5['class_name'][class_id], conf)
                    self.plot_one_box(
                        xyxy, canvas, label=label, color=self.colors[class_id], line_thickness=3)
        return canvas, class_id_list, xyxy_list, conf_list

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        ''''绘制矩形框+标签'''
        tl = line_thickness or round(
            0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


if __name__ == '__main__':
############加载yolo模型
    model = YoloV5(yolov5_yaml_path='config/yolov5.yaml')
    print('模型加载成功')
    try:
        while True:
            intr, depth_intrin, color_image, depth_image, aligned_depth_frame, depth = get_aligned_images()  # 获取对齐的图像与相机内参；就是最上面那个函数
            if not depth_image.any() or not color_image.any():
                continue
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
                depth_image, alpha=0.03), cv2.COLORMAP_JET)              #将深度矩阵映射成图的形式，图中颜色梯度变化反应深度变化
            # images = np.hstack((color_image, depth_colormap))
            # cv2.imshow('images', color_image)
            # key = cv2.waitKey(1)
            # if key & 0xFF == ord('q') or key == 27:
            #        cv2.destroyAllWindows()
            # if key == 32:
            canvas, class_id_list, xyxy_list, conf_list = model.detect(color_image)  #########这里是调用目标检测模型，对获取的彩色图像进行预测
            print("leibie", class_id_list)
            camera_xyz_list = []
            if xyxy_list:  ######得到目标框的像素坐标信息
                for i in range(len(xyxy_list)):
                    x1, y1, x2, y2 = xyxy_list[i]             #######目标框左上角右下角像素坐标
                    #########这俩是中心点的像素坐标
                    ux = int((xyxy_list[i][0] + xyxy_list[i][2]) / 2)  # 计算像素坐标系的x
                    uy = int((xyxy_list[i][1] + xyxy_list[i][3]) / 2)  # 计算像素坐标系的y
                    ##############这是根据中心点像素坐标获取该点的深度，get_distance这是py库里的官方
                    dis = aligned_depth_frame.get_distance(ux, uy)

                    dis = round(dis, 3) ###保留三位小数


                    ########这是根据像素坐标，深度，以及相机的内参矩阵得到三位坐标， 内参矩阵depth_intrin
                    camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, [ux, uy], dis)
                    print("内参矩阵", depth_intrin)
                    ##########内参矩阵 [ 848x480  p[425.389 243.427]  f[607.629 606.69]  畸变：Inverse Brown Conrady [0 0 0 0 0] ]
                    ####不同分辨率对应的不同。在最上面改分辨率。
                    camera_xyz = np.round(np.array(camera_xyz), 3)  # 转成3位小数
                    x, y, z = camera_xyz

                    cv2.circle(canvas, (ux, uy), 4, (0, 0, 0), 5)  # 标出中心点
                    text_x = int(x1+10)
                    text_y = int(y1+15)
                    text = f"XYZ:({x:.2f},{y:.2f},{z:.2f});Dist:{dis:.2f}m"
                    # 获取文本的宽度和高度
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    # 设置背景矩形的参数
                    background_color = (255, 255, 255)  # 白色背景
                    padding = 5  # 背景与文本之间的间距
                    a, b, c, d = int(x1), int(y1), int(x2), int(y2)
                    # 绘制背景矩形
                    cv2.rectangle(canvas, (a, b), (c, int(b + 1.45*text_height)), background_color, -1)  # -1 表示填充矩形
                    cv2.putText(canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

            images = np.hstack((canvas, depth_colormap))
            cv2.imshow('RESULT', images)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Stop streaming
        pipeline.stop()
