#!/root/sam2_env/bin/python3.10
# -*- coding: utf-8 -*-
import sys
import os

# 环境配置, 清理可能冲突的 PYTHONPATH，防止引用了 ROS Noetic (Py3.8) 的 C++ 扩展库
if 'PYTHONPATH' in os.environ:
    del os.environ['PYTHONPATH']

# 强制添加 ROS Noetic 的 Python 路径，以便 import rospy, Python 3.10 可以加载纯 Python 的 rospy 库
sys.path.insert(0, '/opt/ros/noetic/lib/python3/dist-packages')
sys.path.insert(0, '/root/catkin_ws/devel/lib/python3/dist-packages')

import rospy
import numpy as np
import torch
import time
import cv2
import message_filters
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator, SAM2ImagePredictor

class SAM2Node:
    def __init__(self):
        rospy.init_node('sam2_node')

        self.bridge = CvBridge()
        # 1. ROS 通信设置, 订阅压缩图像 
        self.sub = rospy.Subscriber("/wide_angle_camera_depth/image_color_rect_resize", Image, self.image_callback, queue_size=1, buff_size=2**24)
        # 发布分割结果 带mask的Image和可视化的Image
        self.pub = rospy.Publisher("/sam2/segmentation", Image, queue_size=1)
        self.pub_overlay = rospy.Publisher("/sam2/segmentation_overlayed", Image, queue_size=1)
        
        # 2. 参数获取，无值时使用默认值
        cfg_path = rospy.get_param("~sam2_model_cfg", "configs/sam2.1/sam2.1_hiera_t.yaml")
        ckpt_path = rospy.get_param("~sam2_checkpoint", "/root/catkin_ws/src/wild_visual_navigation/sam2/checkpoints/sam2.1_hiera_tiny.pt")
        
        # 3. 模型加载
        rospy.loginfo(f"Loading SAM2 from: {cfg_path}")
        rospy.loginfo(f"Checkpoint: {ckpt_path}")
        
        if not os.path.exists(ckpt_path):
            rospy.logerr(f"Checkpoint file not found at {ckpt_path}! Please download it.")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam2_model = build_sam2(cfg_path, ckpt_path, device=device)

        self.predictor = SAM2ImagePredictor(sam2_model)
        
        # 预先计算网格点 (只在初始化时计算一次)
        # 假设图像大小是 224x299 (根据你的日志)，我们将图像划分为 14x15 的网格
        # 你可以根据需要调整密度，越稀疏越快
        self.grid_h, self.grid_w = 14, 15
        # 生成网格坐标
        y_coords = np.linspace(0, 223, self.grid_h) # 0 到 223
        x_coords = np.linspace(0, 298, self.grid_w) # 0 到 298
        xx, yy = np.meshgrid(x_coords, y_coords)
        # 变成 (N, 2) 的数组
        self.grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)

        rospy.loginfo("SAM2 model loaded successfully.")

        rospy.loginfo("SAM2 Node Initialized.")
        

    def image_callback(self, msg):
        start_total = time.time()

        try:
            # 1. 解码图像
            if isinstance(msg, CompressedImage):
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            H, W = rgb_image.shape[:2]

            # 2. 设置图像 
            self.predictor.set_image(rgb_image)

            # 3. 准备稀疏网格点，均匀采样 16 个点进行独立查询。
            # 数量越多越慢，16 个点通常能平衡速度与细节。
            grid_size = 4  # 4x4 网格 = 16 个点
            x_coords = np.linspace(W//grid_size, W - W//grid_size, grid_size)
            y_coords = np.linspace(H//grid_size, H - H//grid_size, grid_size)
            xx, yy = np.meshgrid(x_coords, y_coords)
            grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1).astype(np.float32)

            # 4. 多区域分割逻辑
            infer_start = time.time()
            
            # 初始化最终的面板，用于存储分割结果
            # 0 表示背景，不同的整数代表不同的区域 ID
            final_panel = np.zeros((H, W), dtype=np.uint8)
            
            current_region_id = 1 # 区域计数器，从 1 开始
            
            # 记录已覆盖的区域，用于去重（可选，这里采用简单的叠加策略）
            # 遍历每一个点，独立进行推理
            for i, point in enumerate(grid_points):
                # 构造单个点的输入 [1, 2]
                point_input = point.reshape(1, 2)
                label_input = np.array([1]) # 1: 前景点
                
                # 预测
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_input,
                    point_labels=label_input,
                    multimask_output=True # 返回 3 个候选
                )
                
                # 选择得分最高的 mask
                best_idx = np.argmax(scores)
                best_mask = masks[best_idx] # (H, W), bool or float
                
                # 转换为 bool 类型 (解决之前的报错)
                best_mask_bool = best_mask.astype(bool)
                
                # 质量过滤：如果置信度太低，忽略该区域
                if scores[best_idx] < 0.5:
                    continue

                # --- 区域合并逻辑 ---
                # 策略：如果该 mask 覆盖的区域大部分是空的（未被标记），则将其标记为新区域。
                # 如果该区域已经被标记过了，则保留原有的标记（防止同一个小物体被重复标记不同ID）。
                
                # 计算该 mask 覆盖了多少“空白区域”
                overlap_area = np.sum(final_panel[best_mask_bool] == 0)
                total_area = np.sum(best_mask_bool)
                
                # 只有当该区域有一定比例是新区域时，才进行标记
                # 这里设置阈值为 50%，防止同一个物体被打成两半
                if total_area > 0 and (overlap_area / total_area) > 0.5:
                    # 如果区域 ID 超过 255，归零重置（防止 uint8 溢出，虽然一般不会遇到这么多区域）
                    if current_region_id > 255:
                        current_region_id = 1 
                        
                    final_panel[best_mask_bool] = current_region_id
                    current_region_id += 1
            
            infer_end = time.time()
            rospy.loginfo(f"[Timer] Inference Time: {(infer_end - infer_start)*1000:.2f} ms | Found Regions: {current_region_id - 1}")

            # 5. 后处理与可视化准备
            # final_panel 中：0=背景, 1=区域1, 2=区域2 ...
            # 为了给 DINOv2 使用，我们可以直接发布 final_panel (mono8)
            # 为了可视化，我们需要将其放大以便看清颜色差异
            
            # --- 发布原始 ID 图 (给 DINOv2) ---
            # DINOv2 可以根据 ID 提取对应区域的特征
            mask_msg = self.bridge.cv2_to_imgmsg(final_panel, encoding="mono8")
            mask_msg.header = msg.header
            self.pub.publish(mask_msg)

            # --- 发布彩色可视化 (给 RViz) ---
            # 将 ID 映射为不同的颜色
            # 使用 HSV 颜色空间生成差异明显的颜色
            hsv_panel = np.zeros((H, W, 3), dtype=np.uint8)
            
            # 背景 (ID 0) 设为黑色
            hsv_panel[final_panel == 0] = [0, 0, 0]
            
            # 为每个区域分配颜色
            for i in range(1, current_region_id):
                # 简单的色相偏移，确保颜色不同
                hue = int((i * 137) % 180) # 137 是质数，能让颜色分布比较散
                hsv_panel[final_panel == i] = [hue, 255, 255]
            
            # 转回 BGR
            color_mask = cv2.cvtColor(hsv_panel, cv2.COLOR_HSV2BGR)
            
            # 叠加到原图 (透明度 0.5)
            overlay_image = cv2.addWeighted(cv_image, 0.6, color_mask, 0.4, 0)
            
            # 发布可视化
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay_image, encoding="bgr8")
            overlay_msg.header = msg.header
            self.pub_overlay.publish(overlay_msg)

            end_total = time.time()
            rospy.loginfo(f"===> [Timer] TOTAL CALLBACK TIME: {(end_total - start_total)*1000:.2f} ms <===")

        except Exception as e:
            rospy.logerr(f"Error in callback: {e}")
            import traceback
            traceback.print_exc()




if __name__ == "__main__":
    try:
        SAM2Node()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
