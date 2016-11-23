import json
import os
import random
import sys

import cv2
import numpy as np
from skimage.feature import hog
from skimage import io, color, transform
from sklearn import svm
from sklearn.externals import joblib

classifier_pickle = 'cache/hog_svm.pkl'
dim = (128, 64)
block = (8, 8)
block_dim = (dim[0]/block[0], dim[1]/block[1])
orient_bins = 8

def get_annotations_dict():
	annotations_filename = 'INRIAPerson/inria_annotations.json'
	annotations_dict = {}
	with open(annotations_filename, 'r') as annotations_file:
		annotations_dict = json.load(annotations_file)
	return annotations_dict
	# training_dict = annotations_dict['training']

def get_rectangle_area(l1, h1):
	return (h1[0]-l1[0])*(h1[1]-l1[1])

def get_rectangle_intersect(l1, h1, l2, h2):
	left = max(l1[0], l2[0])
	right = min(h1[0], h2[0])
	bottom = max(l1[1], l2[1])
	top = min(h1[1], h2[1])
	if(left<right and bottom < top):
		return (right-left)*(top-bottom)
	else:
		return 0


def train_hog():
	pos_train_dir = 'INRIAPerson/96X160H96/Train/pos'
	pos_train = os.listdir(pos_train_dir)
	# print pos_train[0:10]
	print len(pos_train)

	pad = (16, 16)

	use_every = 1
	pos_count = 0
	train_data = []
	train_labels = []
	# get positive samples
	for idx in xrange(0, len(pos_train), use_every):
		im_filename = os.path.join(pos_train_dir, pos_train[idx])
		img_rgb = io.imread(im_filename)
		img = color.rgb2gray(img_rgb)
		# hog = cv2.HOGDescriptor()
		# img = cv2.imread(im_filename)
		crop_img = img[pad[0]:pad[0]+dim[0], pad[1]:pad[1]+dim[1]]
		# io.imsave('cropped.png', crop_img)
		# hog_desc = hog.compute(crop_img)
		hog_desc = hog(crop_img, orientations=orient_bins, pixels_per_cell=block,
	                    cells_per_block=(1, 1), visualise=False, normalise=False)
		train_data.append(hog_desc)
		train_labels.append(1)
		pos_count += 1

	neg_train_dir = 'INRIAPerson/Train/neg'
	neg_train = os.listdir(neg_train_dir)
	print len(neg_train)

	random.seed(0)
	# get negative samples
	for idx in xrange(pos_count):
		# choose a random negative sample and get a random window
		rand_idx = random.randint(0, len(neg_train)-1)
		im_filename = os.path.join(neg_train_dir, neg_train[rand_idx])
		img_rgb = io.imread(im_filename)
		img = color.rgb2gray(img_rgb)
		sy = random.randint(0, img.shape[0]-dim[0]-1)
		sx = random.randint(0, img.shape[1]-dim[1]-1)
		crop_img = img[sy:sy+dim[0], sx:sx+dim[1]]
		hog_desc = hog(crop_img, orientations=orient_bins, pixels_per_cell=block,
	                    cells_per_block=(1, 1), visualise=False, normalise=False)
		train_data.append(hog_desc)
		train_labels.append(0)

	classifier = svm.LinearSVC(C=1.0)
	classifier.fit(train_data, train_labels)

	if os.path.exists(classifier_pickle):
		os.remove(classifier_pickle)
	joblib.dump(classifier, classifier_pickle)

def predict_hog():

	use_every = 1
	start_with = 0
	scales = [0.3, 0.4, 0.6, 0.8]
	write_intermediate = False
	make_heatmap = False
	svm_threshold = 1.5
	nms_iou = 0.2
	# scales = [0.3, 0.6]

	test_dir = 'INRIAPerson/Test/pos'
	test_dict = get_annotations_dict()['test']
	img_filenames = test_dict.keys()
	classifier = joblib.load(classifier_pickle)
	positives = 0
	negatives = 0
	total_windows = 0
	detected_count = 0
	predictions = []
	for idx in xrange(start_with, len(img_filenames), use_every):
		print idx
		img_filename = os.path.join(test_dir, img_filenames[idx])
		img = color.rgb2gray(io.imread(img_filename))
		cv_img = cv2.imread(img_filename)
		img_scale_dict = {}
		img_scale_dict[1.0] = cv_img
		detected = []
		for scale in scales:
			img_scale = transform.resize(img, (int(img.shape[0]*scale), int(img.shape[1]*scale)))
			cv_img_scale = cv2.resize(cv_img, (0,0), fx=scale, fy=scale)
			img_scale_dict[scale] = cv_img_scale
			hog_desc = hog(img_scale, orientations=orient_bins, pixels_per_cell=block,
		                    cells_per_block=(1, 1), visualise=False, normalise=False)
			img_block_dim = (img_scale.shape[0]/block[0], img_scale.shape[1]/block[1])
			# windows = (img_block_dim[0]/block_dim[0] * block_dim[0], img_block_dim[1]/block_dim[1] * block_dim[1])
			windows = (img_block_dim[0], img_block_dim[1])
			# print len(hog_desc)
			# print (img_block_dim[0])*(img_block_dim[1])*orient_bins
			# print block_dim
			# print img_block_dim
			# print windows
			# print img_scale.shape
			# print
			for y in xrange(windows[0] - block_dim[0]):
				for x in xrange(windows[1] - block_dim[1]):
					window_data = []
					start = y*img_block_dim[1]*orient_bins + x*orient_bins
					check = False#y==60 and x==65
					feat_count = 0
					for i in xrange(block_dim[0]):
						start_row = start + i*img_block_dim[1]*orient_bins
						if(check):
							print 'start_row: ' + str(start_row)
						for j in xrange(block_dim[1]):
							hog_idx = start_row + j*orient_bins
							window_data.extend(hog_desc[hog_idx:hog_idx+orient_bins])
							feat_count += orient_bins
							if(check):
								print '  '+str(hog_idx+orient_bins)
					# print "LENGTH: "
					# print len(window_data)
					if(len(window_data) != 1024):
						print len(window_data)
						print feat_count
						print 'y = '+str(y)
						print 'x = '+str(x)
					x_arr = np.array(window_data)
					x_ndarray = x_arr.reshape(1,-1)
					# guess = classifier.predict(x_ndarray)
					guess = classifier.decision_function(x_ndarray)
					if(guess>svm_threshold):
						# probability, scale, top_left_corner, bottom_right_corner, filepath
						detected.append([guess, scale, (x*block[1], y*block[0]), (x*block[1]+dim[1], y*block[0]+dim[0]), False])
						# rr, cc = polygon_perimeter([y*block[0], y*block[0]+dim[0], y*block[0], y*block[0]+dim[0]],
						# 					     		[x*block[1], x*block[1], x*block[1]+dim[1], x*block[1]+dim[1]])
						# set_color(img_scale, (rr, cc), 1)
						# print (x*block[1], y*block[0])
						# print (x*block[1]+dim[1], y*block[0]+dim[0])

						# cv2.rectangle(cv_img_scale, (x*block[1], y*block[0]), (x*block[1]+dim[1], y*block[0]+dim[0]), color=(255,0,0), thickness=2)
			# cv2.imwrite('detected%f.png' % scale, cv_img_scale)

		if(make_heatmap):
			print 'making heatmap . . .'
			scale_final = 1.0
			img_final = img_scale_dict[scale_final]
			heatmap_img = np.zeros((img_final.shape[0], img_final.shape[1], 1), np.uint8)
			quantize = 10
			for det_window in detected:
				to_add = np.zeros((img_final.shape[0], img_final.shape[1], 1), np.uint8)
				l2 = det_window[2]
				h2 = det_window[3]
				if(det_window[1] != scale_final):
					l2 = (int(l2[0]*scale_final / det_window[1]), int(l2[1]*scale_final / det_window[1]))
					h2 = (int(h2[0]*scale_final / det_window[1]), int(h2[1]*scale_final / det_window[1]))
				cv2.rectangle(to_add, l2, h2, color=quantize*det_window[0], thickness=-1)
				heatmap_img = cv2.add(heatmap_img, to_add)
			heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
			cv2.imwrite('heatmap.png', heatmap_color)


		detected.sort()
		detected.reverse()
		maxima = []
		# non maximal suppression
		while len(detected) > 0:
			maxima.append(detected[0])
			l1 = detected[0][2]
			h1 = detected[0][3]
			area0 = get_rectangle_area(l1, h1)
			scale0 = detected[0][1]
			suppressed = []
			for det_idx in xrange(1,len(detected)):
				l2 = detected[det_idx][2]
				h2 = detected[det_idx][3]
				if(detected[det_idx][1] != scale0):
					# print '======================================='
					# print str(detected[det_idx][1]) + ' -> ' + str(scale0)
					# print l2
					l2 = (int(l2[0]*scale0 / detected[det_idx][1]), int(l2[1]*scale0 / detected[det_idx][1]))
					# print l2
					# print h2
					h2 = (int(h2[0]*scale0 / detected[det_idx][1]), int(h2[1]*scale0 / detected[det_idx][1]))
					# print h2
					# print '======================================='
				area1 = get_rectangle_area(l2, h2)
				intersect = get_rectangle_intersect(l1, h1, l2, h2)
				iou = float(intersect) / float(area0+area1-intersect)
				# print iou
				if iou < nms_iou:
					suppressed.append(detected[det_idx])
			# print len(suppressed)
			detected = suppressed

		if write_intermediate:
			print 'output detections . . .'
			scale_final = 1.0#maxima[0][1]
			img_final = img_scale_dict[scale_final]
			for window in maxima:
				l2 = window[2]
				h2 = window[3]
				if(window[1] != scale_final):
					l2 = (int(l2[0]*scale_final / window[1]), int(l2[1]*scale_final / window[1]))
					h2 = (int(h2[0]*scale_final / window[1]), int(h2[1]*scale_final / window[1]))
				cv2.rectangle(img_final, l2, h2, color=(255,0,0), thickness=2)
				# cv2.imwrite('detected.png', img_final)

			for ped_dict in test_dict[img_filenames[idx]]:
				l1 = (ped_dict['left'], ped_dict['top'])
				h1 = (ped_dict['right'], ped_dict['bottom'])
				cv2.rectangle(img_final, l1, h1, color=(0,0,255), thickness=2)
			
			cv2.imwrite('detected.png', img_final)
			print img_filename

		total_windows += len(maxima)
		for ped_dict in test_dict[img_filenames[idx]]:
			l1 = (ped_dict['left'], ped_dict['top'])
			h1 = (ped_dict['right'], ped_dict['bottom'])
			area0 = get_rectangle_area(l1, h1)
			positives += 1
			gt_scale = 1.0
			for window in maxima:
				l2 = window[2]
				h2 = window[3]
				if window[1] != gt_scale:
					l2 = (int(l2[0]*gt_scale / window[1]), int(l2[1]*gt_scale / window[1]))
					h2 = (int(h2[0]*gt_scale / window[1]), int(h2[1]*gt_scale / window[1]))
				area1 = get_rectangle_area(l2, h2)
				intersect = get_rectangle_intersect(l1, h1, l2, h2)
				iou = float(intersect) / float(area0+area1-intersect)
				# print l1
				# print h1
				# print 'compared with'
				# print l2
				# print h2
				# print iou
				# print
				if(iou>0.0):
					window[4] = True
					detected_count += 1
					break

		predictions.extend(maxima)
		# print img_filenames[idx]
		# print test_dict[img_filenames[idx]]
		# print img_filename

	negatives = total_windows - positives
	print 'POSITIVES: %d' % positives
	print 'NEGATIVES: %d' % negatives
	print 'DETECTED: %d' % detected_count
	# result for entire test set 11/22/2016
	# POSITIVES: 589
	# NEGATIVES: 3133
	# DETECTED: 586

	# pr curve
	predictions.sort()
	# predictions.reverse()
	# print predictions[0][0]
	# print predictions[len(predictions)-1][0]
	dimx = 500
	dimy = 500
	padx = 50
	pady = 50
	pr_img = np.zeros((dimx+padx*2, dimy+pady*2, 1), np.uint8)
	ln_color = (255,255,255)
	ln_width = 1
	cv2.line(pr_img, (padx, pady), (padx+dimx, pady), ln_color, 2)
	cv2.line(pr_img, (padx, pady), (padx, pady+dimy), ln_color, 2)
	TP = float(positives)
	FP = float(negatives)
	FN = 0.0
	prev_pt = None
	for prediction in predictions:
		precision = TP / (TP + FP)
		recall = TP / (TP + FN)
		cur_pt = (int(recall*dimx) + padx, int(precision*dimy) + pady)
		if prev_pt:
			cv2.line(pr_img, prev_pt, cur_pt, ln_color, ln_width)

		if prediction[4]:
			TP -= 1
			FN += 1
		else:
			FP -= 1
		prev_pt = cur_pt
	cv2.imwrite('pr_curve.png', pr_img)


if __name__ == '__main__':
	usage = 'USAGE: %s train | predict' % sys.argv[0]
	if len(sys.argv) < 2:
		print usage
	elif sys.argv[1] == 'train':
		train_hog()
	elif sys.argv[1] == 'predict':
		predict_hog()
	else:
		print usage