rosbag record -O wvn_$(date +%F_%H-%M-%S).bag \
  /tf /tf_static \
  /motion_reference/command_twist \
  /wild_visual_navigation_node/robot_state \
  /wide_angle_camera_depth/image_color_rect_resize \
  /wide_angle_camera_depth/image_depth_rect_resize \
  /wide_angle_camera_depth_resize/camera_info \
  /firction_predict

# rosbag record -a -O all_$(date +%F_%H-%M-%S).bag
