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
import torchvision.transforms as T

class SAM2Node:
    def __init__(self):
        rospy.init_node('sam2_node')

        self.bridge = CvBridge()
        # 1. ROS 通信设置, 订阅图像 
        self.sub = rospy.Subscriber("/wide_angle_camera_depth/image_color_rect_resize", Image, self.image_callback, queue_size=1, buff_size=2**24)
        # 发布分割结果 mask图和可视化的RGB图
        self.pub = rospy.Publisher("/sam2/segmentation", Image, queue_size=1)
        self.pub_overlay = rospy.Publisher("/sam2/segmentation_overlayed", Image, queue_size=1)
        
        # 2. 参数获取，无值时使用默认值
        cfg_path = rospy.get_param("~sam2_model_cfg", "configs/sam2.1/sam2.1_hiera_t.yaml")
        ckpt_path = rospy.get_param("~sam2_checkpoint", "/root/catkin_ws/src/wild_visual_navigation/sam2/checkpoints/sam2.1_hiera_tiny.pt")
        
        # 3. SAM2模型加载
        # rospy.loginfo(f"Loading SAM2 from: {cfg_path}")
        # rospy.loginfo(f"Checkpoint: {ckpt_path}")
        if not os.path.exists(ckpt_path):
            rospy.logerr(f"Checkpoint file not found at {ckpt_path}! Please download it.")
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam2_model = build_sam2(cfg_path, ckpt_path, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)

        # 与特征提取的尺寸对齐
        self._target_h = 224  
        self._target_w = 224  
        # 也可以从参数服务器读取 
        # self._target_h = rospy.get_param("~target_height", 224)
        # self._target_w = rospy.get_param("~target_width", 224)

        # 手动做尺寸压缩 最近邻插值
        self.image_crop = T.Compose([
            T.Resize(self._target_h, T.InterpolationMode.NEAREST), 
            T.CenterCrop(self._target_h)
        ])
        
        # 预先计算网格点, 只在初始化时计算一次
        self.grid_h, self.grid_w = 4, 4
        # 生成网格坐标, 使用动态的极值, 向内缩进半个网格间距,避免点打在图像绝对边缘
        margin_y = self._target_h / (2 * self.grid_h)
        margin_x = self._target_w / (2 * self.grid_w)
        y_coords = np.linspace(margin_y, self._target_h - margin_y, self.grid_h) 
        x_coords = np.linspace(margin_x, self._target_w - margin_x, self.grid_w) 
        xx, yy = np.meshgrid(x_coords, y_coords)
        # 变成 (N, 2) 的数组，注意加上 .astype(np.float32)，SAM2 要求浮点数
        self.grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1).astype(np.float32)

        # rospy.loginfo("SAM2 model loaded successfully.")
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

            # a. Numpy (H,W,C, 0-255) -> Torch Tensor (C,H,W, 0.0-1.0)
            rgb_np = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            torch_image = torch.from_numpy(rgb_np).float() / 255.0
            torch_image = torch_image.permute(2, 0, 1).unsqueeze(0)  # 变成 [1, 3, H, W]
            
            # b. 执行与 WVN ImageProjector 完全相同的 Resize + CenterCrop (NEAREST插值)
            resized_torch_image = self.image_crop(torch_image)
            
            # c. 转回 Numpy 供 SAM2 使用
            resized_np_image = (resized_torch_image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # 压缩后的尺寸
            H, W = resized_np_image.shape[:2]

            # 2. 设置图像 
            self.predictor.set_image(resized_np_image)

            # infer_start = time.time()

            # 4. 多区域分割逻辑
            # 初始化最终的面板，用于存储分割结果，0 表示背景，不同的整数代表不同的区域 ID
            final_panel = np.zeros((H, W), dtype=np.uint8)  # 先全部填充0
            current_region_id = 1 # 区域计数器，从 1 开始
            
            # 遍历每一个点，独立进行推理
            for i, point in enumerate(self.grid_points):
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
                
                # 转换为 bool 类型
                best_mask_bool = best_mask.astype(bool)
                
                # 质量过滤：如果置信度太低，忽略该区域
                # if scores[best_idx] < 0.5:
                #     continue

                # --- 区域合并逻辑 ---
                # 策略：如果该 mask 覆盖的区域大部分是空的（未被标记），则将其标记为新区域。
                # 如果该区域已经被标记过了，则保留原有的标记（防止同一个小物体被重复标记不同ID）。
                # 计算该 mask 覆盖了多少“空白区域”
                overlap_area = np.sum(final_panel[best_mask_bool] == 0)
                total_area = np.sum(best_mask_bool)
                
                # 只有当该区域有一定比例是新区域时，才进行标记
                # 这里设置阈值为 50%，防止同一个物体被打成两半
                if total_area > 0 and (overlap_area / total_area) > 0.5:
                    # 如果区域 ID 超过 255，归零重置（防止 uint8 溢出，一般不会遇到这么多区域）
                    if current_region_id > 255:
                        current_region_id = 1 
                        
                    final_panel[best_mask_bool] = current_region_id
                    current_region_id += 1
            
            # DEBUG：用时检查
            # infer_end = time.time()
            # rospy.loginfo(f"[Timer] Inference Time: {(infer_end - infer_start)*1000:.2f} ms | Found Regions: {current_region_id - 1}")
            
            # ---重映射,将区域ID变成连续的---
            # 1. 获取所有存在的 ID
            unique_ids = np.unique(final_panel)
            
            # 2. 构建映射表 {旧ID: 新ID}
            id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
            
            # 3. 创建一个新的 panel 来存储结果
            continuous_panel = np.zeros_like(final_panel)
            
            # 4. 执行映射, 如果区域非常多，这里可以用向量化操作优化，但通常 ID 数量很少，循环即可
            for old_id, new_id in id_map.items():
                continuous_panel[final_panel == old_id] = new_id
            # 替换原来的 panel
            final_panel = continuous_panel
            new_unique_ids = np.unique(final_panel)
            # rospy.loginfo(f"[SAM2] Remapped IDs. Original: {list(unique_ids)} -> Continuous: {list(new_unique_ids)}")

            
            # --- 发布原始 ID 图 (供特征提取使用) ---
            mask_msg = self.bridge.cv2_to_imgmsg(final_panel, encoding="mono8")
            mask_msg.header = msg.header
            self.pub.publish(mask_msg)

            # --- 发布彩色可视化 (给 RViz) ---
            # 将 ID 映射为不同的颜色, 使用 HSV 颜色空间生成差异明显的颜色
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
            base_bgr_image = cv2.cvtColor(resized_np_image, cv2.COLOR_RGB2BGR)
            overlay_image = cv2.addWeighted(base_bgr_image, 0.6, color_mask, 0.4, 0)
            
            # 叠加到原图 (透明度 0.5)
            overlay_image = cv2.addWeighted(resized_np_image, 0.6, color_mask, 0.4, 0)
            
            # 发布可视化
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay_image, encoding="bgr8")
            overlay_msg.header = msg.header
            self.pub_overlay.publish(overlay_msg)

            # DEBUG：尺寸检查打印
            # rospy.loginfo(f"[SAM2 Publish] segmentation shape (H, W): {final_panel.shape}, RGB shape (H, W): {overlay_image.shape}")
            
            # 用时打印
            # end_total = time.time()
            # rospy.loginfo(f"===> [SAM2] TOTAL CALLBACK TIME: {(end_total - start_total)*1000:.2f} ms <===")

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
