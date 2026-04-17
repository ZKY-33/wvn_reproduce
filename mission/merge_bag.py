#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cd src/wild_visual_navigation/mission/
# python merge_bag.py 
import os
import rosbag
from genpy import Time
from tqdm import tqdm

# path & file
BAG_DIR = "/root/catkin_ws/src/wild_visual_navigation/mission"
BAG_FILES = [
    "wvn_2026-04-16_08-05-12.bag",   
    "wvn_2026-04-16_08-05-36.bag",   
    "wvn_2026-04-16_08-05-58.bag", 
    "wvn_2026-04-16_08-06-19.bag", 
    "wvn_2026-04-16_08-06-45.bag", 
]
OUTPUT_NAME = "wvn_2026-04-16_merged.bag"


def get_bag_paths():
    """path check"""
    bag_paths = []
    for filename in BAG_FILES:
        full_path = os.path.join(BAG_DIR, filename)
        if not os.path.exists(full_path):
            print(f"[错误] 找不到 bag 文件: {full_path}")
            exit(1)
        bag_paths.append(full_path)
    return bag_paths


def print_bag_info(bag_paths):
    """bag info check"""
    # print(f" 共有 {len(bag_paths)} 个 bag 待合并")
    # print(f" 输出目录: {os.path.abspath(BAG_DIR)}")
    # print(f" 输出文件: {OUTPUT_NAME}")
    for i, path in enumerate(bag_paths):
        with rosbag.Bag(path, 'r') as bag:
            start = bag.get_start_time()
            end = bag.get_end_time()
            duration = end - start
            print(f"  [{i+1}] {os.path.basename(path)}")
            print(f"      原始时间范围: {start:.3f} ~ {end:.3f}  (时长 {duration:.2f}s)")

def make_time(sec_float):
    """float secs to genpy.Time data """
    secs = int(sec_float)
    nsecs = int((sec_float - secs) * 1e9)
    if nsecs >= 1000000000:
        secs += 1
        nsecs -= 1000000000
    return Time(secs, nsecs)


def merge_bags(bag_paths, output_path):
    """change timestamp of every bag (except 1st bag)"""

    prev_end_sec = None

    if os.path.exists(output_path):
        os.remove(output_path)

    with rosbag.Bag(output_path, 'w') as outbag:

        for idx, bag_path in enumerate(bag_paths):
    
            with rosbag.Bag(bag_path, 'r') as inbag:
                bag_start_sec = inbag.get_start_time()
                bag_end_sec = inbag.get_end_time()

                # 计算时间偏移
                if idx == 0:
                    time_shift = 0.0
                else:
                    time_shift = prev_end_sec - bag_start_sec

                # 逐条写入
                for topic, msg, t in tqdm(inbag.read_messages(),
                                          total=inbag.get_message_count(),
                                          desc="  写入", unit="msg"):
                    # 外部时间戳
                    new_t = make_time(t.to_sec() + time_shift)

                    # 内部 header.stamp
                    if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                        ss = msg.header.stamp.secs + msg.header.stamp.nsecs / 1e9
                        if ss > 0:
                            nt = make_time(ss + time_shift)
                            msg.header.stamp.secs = nt.secs
                            msg.header.stamp.nsecs = nt.nsecs

                    outbag.write(topic, msg, new_t)

                # 更新时间指针
                prev_end_sec = bag_end_sec + time_shift
                # print(f"prev_end_sec: {prev_end_sec}")



    # 合并后输出打印
    with rosbag.Bag(output_path, 'r') as final_bag:
        dur = final_bag.get_end_time() - final_bag.get_start_time()
        cnt = final_bag.get_message_count()
        print(f"    合并完成")
        print(f"    合并后时长: {dur:.2f}s")
        print("     总消息数: %d" % cnt)


if __name__ == "__main__":
    bag_paths = get_bag_paths()
    print_bag_info(bag_paths)
    output_path = os.path.join(BAG_DIR, OUTPUT_NAME)
    merge_bags(bag_paths, output_path)
