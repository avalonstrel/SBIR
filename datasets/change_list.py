import pickle

path = "train_imagenet_image_list.pkl"
dic = pickle.load(open(path, 'rb'))

photo_imgs = dic['photo_imgs']

with open('train.lst','w') as writer:
	for photo_img in photo_imgs:
		writer.write(photo_img + '\n')

