from pycocotools.coco import COCO
import numpy as np
import cv2
import os

data_dir = 'val2017'
ann_file='instances_val2017.json'

seg_output_path = 'dataset_coco_person_mask/coco/seg'

original_img_path = 'dataset_coco_person_mask/coco/img'

coco = COCO(ann_file)

catIds = coco.getCatIds(catNms=['person']) #Add more categories ['person','dog']
imgIds = coco.getImgIds(catIds=catIds )

for i in range(len(imgIds)):
	img = coco.loadImgs(imgIds[i])[0]
	annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=0)
	anns = coco.loadAnns(annIds)
	mask = coco.annToMask(anns[0])

	for i in range(len(anns)): #get if number annotations contains number of masks
		mask += coco.annToMask(anns[i])

	# print(mask.shape)
	file_name = os.path.join(data_dir,img['file_name'])
	original_img = cv2.imread(file_name)
	cv2.imwrite(os.path.join(original_img_path,img['file_name']),original_img)
	cv2.imwrite(os.path.join(seg_output_path,img['file_name']),mask*255)

	# cv2.imshow(os.path.join(seg_output_path,img['file_name']),mask*255)
	# cv2.waitKey(0)
	
	# print(mask)
	# print(mask*255)
	print("processing...")

print("Done")