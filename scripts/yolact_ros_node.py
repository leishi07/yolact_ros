#!/usr/bin/env python3
import rospy
import roslib
roslib.load_manifest('yolact_ros')
from std_msgs.msg import Header
from sensor_msgs.msg import Image 
from yolact_ros.msg import yolact_objects, Object
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import torch
from yolact.yolact_ros_class import Yolact_ROS
from cv_bridge import CvBridge, CvBridgeError

class pub_sub:
	def __init__(self, sub_img_topic_name, yolact):
		self.image_pub = rospy.Publisher("/yolact_image", Image, queue_size=5)
		self.yolact_obj_pub = rospy.Publisher("/yolact_objects", yolact_objects, queue_size=5)
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber(sub_img_topic_name, Image, self.img_callback)
		self.yolact = yolact

	def img_callback(self,data):
		#subscribe image
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)
		masks, classes, scores, bboxes = self.yolact.prediction(cv_image)

		img_msg = Image()
		yolact_obj_msg = yolact_objects()
		for i in range(len(classes)):
			object_msg = Object()
			object_msg.Class = classes[i].decode()
			object_msg.score = scores[i]
			object_msg.x1 = bboxes[i][0]
			object_msg.y1 = bboxes[i][1]
			object_msg.x2 = bboxes[i][2]
			object_msg.y2 = bboxes[i][3]
			object_msg.mask = self.bridge.cv2_to_imgmsg(masks[i], "mono8")
			yolact_obj_msg.object.append(object_msg)

		img_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
		yolact_obj_msg.header.stamp = rospy.get_rostime()
		img_msg.header.stamp = rospy.get_rostime()
		yolact_obj_msg.header.frame_id = "yolact_obj"
		img_msg.header.frame_id = "yolact_frame"

		# publish yolact topics
		try:
			self.image_pub.publish(img_msg)
			self.yolact_obj_pub.publish(yolact_obj_msg)
		except CvBridgeError as e:
			print(e)

def main():
	rospy.init_node('pub_sub', anonymous=True)
	#load parameters
	img_topic = rospy.get_param('~sub_img_topic_name')
	model_path = rospy.get_param('~model_path')
	with_cuda = rospy.get_param('~cuda')
	yolact_config = rospy.get_param('~yolact_config')
	use_fast_nms = rospy.get_param('~use_fast_nms')
	threshold = rospy.get_param('~threshold')
	display_cv = rospy.get_param('~display_cv')
	top_k = rospy.get_param('~top_k')

	yolact = Yolact_ROS(model_path, with_cuda, yolact_config, use_fast_nms, threshold, display_cv, top_k)
	ps = pub_sub(img_topic, yolact)

	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
		cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
