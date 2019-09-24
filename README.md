# yolact_ros
This is a ROS package interfacing real-time object detector YOLACT (https://github.com/dbolya/yolact) which provides the masks of the detected objects. A ROS node subscribes bgr image and publishes bgr image, class labels, prediction socres, bounding boxes and masks. This ROS package uses the same trained weights and configurations as in YOLACT.
# Pre-request
- ROS with Python 3.5 
- Pytorch and torch vision
# Compatability
Tested on Ubuntu 16.04 + ROS Kinetic + Python 3.5\
To use cv_bridge with Python 3 in ROS Kinetic, build cv_bridge from source in a catkin workspace, remember to set python library to Python 3.
# Citaiton
If you use yolact_ros in your work, please cite\
```
@misc{leiYolactROS2019,
  title = {{ROS} Interface for Real-Time Object Detector {YOLACT}}, 
  author = {Lei Shi},
  url = {https://github.com/leishi07/yolact_ros},
  year = {2019},
}
```

# Usage
1. **Launching**

  ```roslaunch yolact_ros yolact_ros.launch```

2. **Parameters**

  Parameters can be changed in ```launch/yolcat_ros.launch```

  - ```sub_img_topic_name:``` ROS image topic to subscribe
  - ```model_path:```         path to save model and name of the model, weights can be downloaded at YOLACT site (https://github.com/dbolya/yolact)
  - ```cuda:```               use GPU acceleration or not
  - ```use_fast_nms:```       use fast NMS or not
  - ```threshold:```          threshold for detection
  - ```display_cv:```         Display cv image or not
  - ```top_k:```              get top k objects, you will also only get k objects in published ros topics
  - ```yolact_config:```      set configuration for Yolact

    | "yolact_config" | Model |
    | --- | --- |
    | yolact_base_config | yolact_base_54_800000.pth |
    | yolact_darknet53_config|  yolact_darknet53_54_800000.pth |
    | yolact_resnet50_config | yolact_resnet50_54_800000.pth |
    | yolact_im700_config | yolact_im700_54_800000.pth |


3. **Published Topics**

  - sensor_msgs/Image: ```"/yolact_image"```\
    Publishes ROS image topic.

  - yolact_ros/yolact_objects: ```"/yolact_objects"```\
    Publishes detected objects topic. This is a custom message type.
    
    **yolact_ros/yolact_objects**:
    ```
    Header header
    Object[] object
    ```
    
    **yolact_ros/Object**:
    ```
    string Class
    float64 score
    int64 x1
    int64 y1
    int64 x2
    int64 y2
    sensor_msgs/Image mask
    ```
    ```x1``` and ```y1``` is the top-left point of the bounding box ```x2``` and ```y2``` is the bottom-right one.
