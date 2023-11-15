# Custom_X-ray_Datasets
In this repository, I will gradually share labeled datasets for X-ray baggage detection for scientific researches. If you would like to obtain these datasets, please contact me, my email is 2311295@tongji.edu.cn.

GDXray-Expanded dataset is an extension of GDXray dataset, there are 10000 grayscale images which include gun, scissors,knives, and several types of explosives. The image background is complex, and there are varying degrees of occlusion and overlap between items, which is more in line with actual security screening scenarios. The label format is VOC. The classes are grenade-normal, grenade-handle, grenade-rectangle, grenade-tube, scissors, gun and knife. The paper link is https://doi.org/10.3390/app122211848.

Oriented object detection dataset preparation process for this project: https://github.com/tecsai/YOLOv5_DOTA_OBB. Please see the file ‘preparedata.txt’.
