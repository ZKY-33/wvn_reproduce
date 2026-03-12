#!/root/sam2_env/bin/python3.10
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_PATH = os.path.join(CURRENT_DIR, "../image/image2.jpg")   # sam2/sam2
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"   # hydra 指向系统python里sam2 config路径
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, "../checkpoints/sam2.1_hiera_large.pt")
OUTPUT_PATH = os.path.join(CURRENT_DIR, "../image/image2_output.jpg")
IMAGE_MAX_SIZE = 1280  
OBJECT_NUM = 15

def batch_detection(image_path, model_cfg, checkpoint_path, output_path):
    # 1. 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"正在使用设备: {device}")

    # 2. 加载模型
    print("正在加载 SAM2 模型...")
    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    image_array = cv2.imread(image_path)
    if image_array is None:
        print(f"错误：无法读取图片 {image_path}")
        return
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # 3. 读取图片, 自动缩放
    image_array = cv2.imread(image_path)
    if image_array is None:
        print(f"错误：无法读取图片 {image_path}")
        return
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    h, w = image_array.shape[:2]
    scale = 1.0
    
    if max(h, w) > IMAGE_MAX_SIZE:
        scale = IMAGE_MAX_SIZE / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        print(f"原始尺寸: {w}x{h} (过大)，自动缩放至: {new_w}x{new_h}")
        image_array = cv2.resize(image_array, (new_w, new_h))
        h, w = image_array.shape[:2] # 更新尺寸
    else:
        print(f"当前图片尺寸: {w}x{h}，无需缩放。")
        
    # 4. 设置图片
    predictor.set_image(image_array)

    # 5. 生成网格采样点 
    print("正在生成网格搜索点...")
    nx, ny = 12, 12 # 网格密度
    x_steps = np.linspace(50, w - 50, nx)
    y_steps = np.linspace(50, h - 50, ny)
    grid_x, grid_y = np.meshgrid(x_steps, y_steps)
    points_grid = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1).astype(np.float32)
    
    """单点处理"""
    # # 6. 循环单点预测并收集所有结果
    print(f"正在进行 {len(points_grid)} 次单点预测搜索...")
    
    all_masks = []
    all_scores = []
    
    # 遍历每一个点
    for i, point in enumerate(points_grid):
        input_point = np.expand_dims(point, axis=0)
        input_label = np.array([1]) # 前景点
        
        # 预测
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        
        # 排序逻辑：mask of max score (参考官方示例)
        sorted_ind = np.argsort(scores)[::-1]
        best_mask = masks[sorted_ind[0]]
        best_score = scores[sorted_ind[0]]
        
        # 结果
        all_masks.append(best_mask)
        all_scores.append(best_score)

    print(f"预测完成，共生成 {len(all_masks)} 个原始掩码")

    """批量处理"""
    # # 6. 批量预测 
    # # 数据准备：转为 Tensor 并移至 GPU
    # points_tensor = torch.tensor(points_grid, dtype=torch.float, device=predictor.device)
    # labels_tensor = torch.ones(len(points_grid), dtype=torch.int, device=predictor.device)
    
    # # 预处理坐标，调用内部方法进行归一化，对应 predict 源码中的 _prep_prompts
    # _, unnorm_coords, unnorm_labels, _ = predictor._prep_prompts(
    #     point_coords=points_tensor,
    #     point_labels=labels_tensor,
    #     box=None,
    #     mask_input=None,
    #     normalize_coords=True
    # )
    
    # all_masks = []
    # all_scores = []
    # batch_size = 32  # 显存允许的情况下可以调大，如 64
    
    # # 分批推理 
    # with torch.no_grad(): # 节省显存
    #     for i in range(0, len(points_grid), batch_size):
    #         cur_coords = unnorm_coords[i : i + batch_size]
    #         cur_labels = unnorm_labels[i : i + batch_size]
            
    #         # 直接使用 _predict，返回值是原始 Tensor，形状
    #         masks, iou_preds, _ = predictor._predict(
    #             point_coords=cur_coords,
    #             point_labels=cur_labels,
    #             multimask_output=True,
    #             return_logits=False
    #         )
            
    #         # 后处理，转回 CPU Numpy
    #         masks_np = masks.cpu().numpy()      # (N, 3, H, W)
    #         scores_np = iou_preds.cpu().numpy() # (N, 3)
            
    #         # 遍历当前批次，提取最佳结果
    #         for j in range(len(masks_np)):
    #             point_masks = masks_np[j]   # (3, H, W)
    #             point_scores = scores_np[j] # (3,)
                
    #             # 选分数最高的
    #             best_idx = np.argmax(point_scores)
    #             best_mask = point_masks[best_idx]
    #             best_score = point_scores[best_idx]
                
    #             all_masks.append(best_mask)
    #             all_scores.append(best_score)
            
    #         # 进度打印
    #         # if (i + batch_size) % 64 == 0 or (i + batch_size) >= len(points_grid):
    #         #     print(f"已处理 {min(i + batch_size, len(points_grid))}/{len(points_grid)} 个点...")

    # all_masks = np.array(all_masks)
    # all_scores = np.array(all_scores)
    # print(f"预测完成，共生成 {len(all_masks)} 个原始掩码")

    # 7. 阈值过滤与 NMS 去重 (结合面积排序)
    score_threshold = 0.9   # 分数阈值，滤去较小的部分
    iou_threshold = 0.8     # 提高重叠阈值，允许大物体内部有小物体，超过此值为一个整体
    
    # 转换为 numpy 数组
    all_masks = np.array(all_masks)
    all_scores = np.array(all_scores)
    
    # 初步过滤
    valid_indices = all_scores > score_threshold
    all_masks = all_masks[valid_indices]
    all_scores = all_scores[valid_indices]
    
    if len(all_masks) == 0:
        print("未找到符合条件的掩码。")
        return

    # 计算密度并给权重
    densities = []
    for mask in all_masks:
        area = np.sum(mask)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours) + 1e-6
        densities.append(area / perimeter)
    densities = np.array(densities)

    # 综合排序
    density_weight = 2.0
    # 归一化防止溢出
    norm_densities = densities / (densities.max() + 1e-6)
    combined_scores = all_scores + density_weight * norm_densities
    
    sorted_indices = np.argsort(combined_scores)[::-1]
    all_masks = all_masks[sorted_indices]
    all_scores = all_scores[sorted_indices]
    
    # 高性能 NMS 去重，预先将所有 mask 缩小到小尺寸 (例如宽 100 像素)
    small_size = (100, 100) # 缩放后的尺寸
    small_masks = []
    for mask in all_masks:
        # 使用 OpenCV 缩小图片
        small_mask = cv2.resize(mask.astype(np.uint8), small_size, interpolation=cv2.INTER_NEAREST)
        small_masks.append(small_mask)
    small_masks = np.array(small_masks)
    
    keep_masks = []
    keep_scores = []
    
    print("正在进行快速去重计算...")
    for i in range(len(all_masks)):
        # 在大图上保留原始 mask 用于输出
        current_mask = all_masks[i]
        current_score = all_scores[i]
        
        # 在小图上进行重叠计算 (关键加速点)
        current_small = small_masks[i]
        
        should_keep = True
        for j in range(len(keep_masks)):
            pass 

        
    keep_masks = []
    keep_scores = []
    keep_small_masks = [] # 用于快速比较的列表
    
    for i in range(len(all_masks)):
        current_mask = all_masks[i]
        current_score = all_scores[i]
        current_small = small_masks[i]
        
        should_keep = True
        for kept_small in keep_small_masks:
            # 在小尺寸上计算 IoU
            intersection = np.logical_and(current_small, kept_small).sum()
            union = np.logical_or(current_small, kept_small).sum()
            iou = intersection / (union + 1e-6)
            
            if iou > iou_threshold:
                should_keep = False
                break
        
        if should_keep:
            keep_masks.append(current_mask)
            keep_scores.append(current_score)
            keep_small_masks.append(current_small) # 同步添加小图

    print(f"去重完成。最终保留 {len(keep_masks)} 个不同的目标掩码。")


    if len(keep_masks) > OBJECT_NUM:
        keep_masks = keep_masks[:OBJECT_NUM]
        keep_scores = keep_scores[:OBJECT_NUM]
        print(f"保留分数最高的 {OBJECT_NUM} 个掩码。")


    # 8. 可视化结果 OpenCV 合成
    if len(keep_masks) > 0:
        # 复制一份原图，防止修改原图数据
        # 注意：cv2 读取的图是 BGR，绘图通常用 RGB，这里统一处理
        output_image = image_array.copy() 
        
        np.random.seed(42) # 固定随机种子
        
        # 遍历所有掩码进行绘制
        for mask in keep_masks:
            mask = mask.astype(bool) 
            # 随机颜色
            color = np.random.randint(0, 256, (3,)).tolist()
            
            # 全透明彩色掩码层
            colored_mask = np.zeros_like(output_image, dtype=np.uint8)
            # 将掩码区域填充颜色
            colored_mask[mask] = color
            
            # 将彩色掩码混合到原图上 (加权叠加)
            # alpha=0.5 表示掩码 50% 透明，能看到下面的原图
            output_image = cv2.addWeighted(output_image, 1.0, colored_mask, 0.5, 0)
            
            # 绘制边框
            # 寻找轮廓
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 线条宽度为 2
            cv2.drawContours(output_image, contours, -1, color, 2)

        # 保存结果，假设 image_array 是 RGB (plt.imread 或转换过的)，转回 BGR 保存
        output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_image_bgr)
        
        print(f"结果已保存至: {output_path}")
        
        # 窗口查看 
        # cv2.imshow("Result", output_image_bgr)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
    else:
        print("未检测到有效目标。")


if __name__ == '__main__':
    batch_detection(IMAGE_PATH, MODEL_CFG, CHECKPOINT_PATH, OUTPUT_PATH)
