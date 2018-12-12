import numpy as np
import sys
import os
#try:
#	caffe_root = os.environ['CAFFE_ROOT'] + '/'
#except KeyError:
#  	raise KeyError("Define CAFFE_ROOT in ~/.bashrc")
#caffe_path = '/home/gaia/Dev/caffe/'
caffe_path = '/home/gaia/caffe/'

if caffe_path not in sys.path:
    sys.path.insert(1, caffe_path+'python/')
import caffe
import cv2
from py_returnCAMmap import py_returnCAMmap
from py_map2jpg import py_map2jpg
import scipy.io

def im2double(im):
	return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

## Be aware that since Matlab is 1-indexed and column-major, 
## the usual 4 blob dimensions in Matlab are [width, height, channels, num]

## In python the dimensions are [num, channels, width, height]
#img_root = '/mnt/2Tdisk/data/Dream/datasets_rnd3/final_dataset/fffdd3aa3f901c55324a771a50dfee02.jpg'
test_txt = "../test_diffuse.txt"
#outside validation set
test_txt = '/mnt/lijian/mount_out/zhongshan_ophthalmology_data20180206/sorted_txt/diffuse_illness_val.txt'

with open(test_txt, 'r') as tf:
    all_images = tf.readlines()

img_root = all_images[1].split(' ')[0]
label = all_images[1].split(' ')[1]

model = 'resnet'
input_size = 224 

if model == 'alexnet':
	net_weights = 'models/alexnetplusCAM_imagenet.caffemodel'
	net_model = 'models/deploy_alexnetplusCAM_imagenet.prototxt'
	out_layer = 'fc9'
	last_conv = 'conv7'
	crop_size = 227
elif model == 'googlenet':
	net_weights = 'models/imagenet_googleletCAM_train_iter_120000.caffemodel'
	net_model = 'models/deploy_googlenetCAM.prototxt'
	out_layer = 'CAM_fc'
	crop_size = 224
	last_conv = 'CAM_conv'
elif model == 'all_fundus':
    net_weights = '/home/gaia/PycharmProjects/diabetic/inception_v3_Dream25_iter_150000.caffemodel'
    net_model = '/home/gaia/PycharmProjects/diabetic/inception_v3_train_val25.prototxt'
    out_layer = 'classifier_ft13'
    crop_size = 961
    last_conv = 'inception_c2_concat'
elif model == 'resnet':
    net_model = "/home/gaia/share/cataractData/codes/resnet-protofiles-master/ResNet50_illness3_deploy_diff.prototxt"
    net_weights = '/home/gaia/share/cataractData/codes/snapshots_dec/resnet_illness3_diffuse_iter_180000.caffemodel'
    out_layer = 'fc3'
    last_conv = 'res5c_branch2c'
    crop_size =224
else:
	raise Exception('This model is not defined')

# categories = scipy.io.loadmat('categories1000.mat')

# load CAM model and extract features
net = caffe.Net(net_model, net_weights, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load('/home/gaia/share/cataractData/codes/lmdb_march/illness3_mean_diffuse.npy').mean(1).mean(1))
#transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

weights_LR = net.params[out_layer][0].data # get the softmax layer of the network
# shape: [1000, N] N-> depends on the network

image = cv2.imread(img_root)
image = cv2.resize(image, (input_size, input_size))

cv2.imshow("src image",image)
# Take center crop.
#center = np.array(image.shape[:2]) / 2.0
#crop = np.tilie(center, (1, 2))[0] + np.concatenate([
#	-np.array([crop_size, crop_size]) / 2.0,
#	np.array([crop_size, crop_size]) / 2.0
#])
#crop = crop.astype(int)
#input_ = image[crop[0]:crop[2], crop[1]:crop[3], :]
input_ = image

# extract conv features
net.blobs['data'].reshape(*np.asarray([1,3,crop_size,crop_size])) # run only one image
net.blobs['data'].data[...][0,:,:,:] = transformer.preprocess('data', input_)
out = net.forward()
scores = out['prob']
activation_lastconv = net.blobs[last_conv].data
print "label: ", label, "  predict: ", scores.argmax()

## Class Activation Mapping

topNum = 3 # generate heatmap for top X prediction results
scoresMean = np.mean(scores, axis=0)
ascending_order = np.argsort(scoresMean)
IDX_category = ascending_order[::-1] # [::-1] to sort in descending order

curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[IDX_category[:topNum],:])

curResult = im2double(image)

for j in range(topNum):
	# for one image
	curCAMmap_crops = curCAMmapAll[:,:,j]
	curCAMmapLarge_crops = cv2.resize(curCAMmap_crops, (input_size,input_size))
	curHeatMap = cv2.resize(im2double(curCAMmapLarge_crops),(input_size,input_size)) # this line is not doing much
	curHeatMap = im2double(curHeatMap)

	curHeatMap = py_map2jpg(curHeatMap, None, 'jet')
	curHeatMap = im2double(image)*0.2+im2double(curHeatMap)*0.4

	cv2.imshow(img_root.split()[0].split('/')[-1]+'_'+str(IDX_category[j]),curHeatMap)
	#cv2.imshow(categories['categories'][IDX_category[j]][0][0], curHeatMap)
	cv2.waitKey(0)
