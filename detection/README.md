# 객체인식 파트
## 카메라를 이용하여 술 분류 객체인식
- 1번 카메라 사용
```
ros2 launch realsense2_camera rs_launch.py \
    camera_name:=camera_1 \
    serial_no:=_311322302867 \
    enable_pointcloud:=true
```
- 1번 카메라 토픽 확인
```
/camera/camera_1/color/image_raw
/camera/camera_1/aligned_depth_to_color/image_raw
```

## 카메라를 이용하여 컵 객체인식
- 2번 카메라 사용
```
ros2 launch realsense2_camera rs_launch.py \
    camera_name:=camera_2 \
    serial_no:=_313522301601 \
    enable_pointcloud:=true
```
- 2번 카메라 토픽 확인
```
/camera/camera_2/color/image_raw
/camera/camera_2/aligned_depth_to_color/image_raw
```
