import torch
import torch.backends.cudnn as cudnn
import time
from yolact.yolact import Yolact
from yolact.utils.augmentations import FastBaseTransform
from yolact.layers.output_utils import postprocess
from yolact.data.config import cfg, mask_type, set_cfg
from yolact.data import COLORS
from collections import defaultdict
import cv2
import os
import numpy as np

class Yolact_ROS(object):
	def __init__(self, model_path, with_cuda, yolact_config, fast_nms, threshold, display_cv, top_k):
		self.top_k = top_k
		self.threshold = threshold
		self.display_cv = display_cv
		print("loading Yolact ...")

		with torch.no_grad():
			set_cfg(yolact_config)
			print("Configuration: ", yolact_config)

			if with_cuda:
				cudnn.benchmark = True
				cudnn.fastest = True
				torch.set_default_tensor_type('torch.cuda.FloatTensor')
			else:
				torch.set_default_tensor_type('torch.FloatTensor')
			
			print("use cuda: ", with_cuda)

			self.net = Yolact()
			self.net.load_weights(model_path)
			print("Model: ", model_path)
			self.net.eval()

			if with_cuda:
				self.net = self.net.cuda()

			self.net.detect.use_fast_nms = fast_nms
			print("use fast nms: ", fast_nms)
		print("Yolact loaded")

	def prediction(self, img):
		self.net.detect.cross_class_nms = True
		cfg.mask_proto_debug = False

		with torch.no_grad():
			frame = torch.Tensor(img).cuda().float()
			batch = FastBaseTransform()(frame.unsqueeze(0))
			time_start = time.clock()
			preds = self.net(batch)
			h, w, _ = img.shape
			t = postprocess(preds, w, h, visualize_lincomb=False, crop_masks=True, score_threshold=self.threshold) 
			torch.cuda.synchronize()
			masks = t[3][:self.top_k]
			classes, scores, bboxes = [x[:self.top_k].cpu().numpy() for x in t[:3]]
			time_elapsed = (time.clock() - time_start)
			num_dets_to_consider = min(self.top_k, classes.shape[0])

			for i in range(num_dets_to_consider):
				if scores[i] < self.threshold:
					num_dets_to_consider = i
					break

			if num_dets_to_consider >= 1:
				masks = masks[:num_dets_to_consider, :, :, None]
				
			masks_msg = masks.cpu().detach().numpy()
			masks_msg = masks_msg.astype(np.uint8)
			scores_msg = np.zeros(num_dets_to_consider)
			class_label_msg = np.empty(num_dets_to_consider, dtype="S20")
			bboxes_msg = np.zeros([num_dets_to_consider, 4], dtype=int)
			for i in reversed(range(num_dets_to_consider)):
				scores_msg[i] = scores[i]
				class_label_msg[i] = cfg.dataset.class_names[classes[i]]
				bboxes_msg[i] = bboxes[i]
				print(class_label_msg[i].decode(), "%.2f" % (scores_msg[i]))

			os.system('cls' if os.name=='nt' else 'clear')
			print("%.2f" % (1/time_elapsed), "hz") 

			if self.display_cv:
				self.display(frame, masks, classes, scores, bboxes, num_dets_to_consider)

			return masks_msg, class_label_msg, scores_msg, bboxes_msg

	def display(self, img, masks, pred_classes, scores, bboxes, num_dets_to_consider, mask_alpha=0.75):
		img_gpu = img / 255.0
		if num_dets_to_consider == 0:
			return (img_gpu * 255).byte().cpu().numpy()

		use_class_color = True
		colors = torch.cat([self.get_color(i, pred_classes, use_class_color, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for i in range(num_dets_to_consider)], dim=0)
		masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha
		inv_alph_masks = masks * (-mask_alpha) + 1
		masks_color_summand = masks_color[0]

		if num_dets_to_consider > 1:
			inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
			masks_color_cumul = masks_color[1:] * inv_alph_cumul
			masks_color_summand += masks_color_cumul.sum(dim=0)

		img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
		img_numpy = (img_gpu * 255).byte().cpu().numpy()

		for i in reversed(range(num_dets_to_consider)):
			x1, y1, x2, y2 = bboxes[i, :]
			color = self.get_color(i,pred_classes,use_class_color)
			score = scores[i]
			cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)
			_class = cfg.dataset.class_names[pred_classes[i]]
			text_str = '%s: %.2f' % (_class, score) if True else _class
			font_face = cv2.FONT_HERSHEY_DUPLEX
			font_scale = 0.6
			font_thickness = 1
			text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
			text_pt = (x1, y1 - 3)
			text_color = [255, 255, 255]
			cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
			cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

		cv2.imshow("yolact", img_numpy)
		cv2.waitKey(1)
	
	def get_color(self, i, pred_classes, class_color, on_gpu=None ):
		color_cache = defaultdict(lambda: {})
		color_idx = (pred_classes[i] * 5 if class_color else i * 5) % len(COLORS)

		if on_gpu is not None and color_idx in color_cache[on_gpu]:
			return color_cache[on_gpu][color_idx]
		else:
			color = COLORS[color_idx]

			if on_gpu is not None:
				color = torch.Tensor(color).to(on_gpu).float() / 255.
				color_cache[on_gpu][color_idx] = color

			return color
