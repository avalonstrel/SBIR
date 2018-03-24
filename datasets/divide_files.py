import os
root = '/home/lhy/Toolkit/holy-edge/hed/imagenet_train'
save_root = '/home/lhy/ILSVRC2012_HED'
for _, _, files in os.walk(root):
	for file in files:
		cls_root_i = file.find('_')
		cls_root = file[:cls_root_i]
		cls_root = os.path.join(save_root, cls_root)
		if not os.path.exists(cls_root): 
			os.system('mkdir '+ cls_root)
		os.system('cp {} {}'.format(os.path.join(root, file),cls_root))


