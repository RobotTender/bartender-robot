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
ros2 topic list

/camera/camera_1/color/image_raw
/camera/camera_1/aligned_depth_to_color/image_raw
```
- 1번 카메라 객체 인식(object detection) 확인
```
# ros2, ultralytics가 설치된 가상환경에서 실행할 것
python3 ./detection/realsense_cam1.py
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
ros2 topic list

/camera/camera_2/color/image_raw
/camera/camera_2/aligned_depth_to_color/image_raw
```
- 2번 카메라 객체 인식(object detection) 확인
```
# ros2, ultralytics가 설치된 가상환경에서 실행할 것
python3 ./detection/realsense_cam2.py
```