import numpy as np
import sys
import cv2, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--caffe_folder", type=str, help="Caffe folder to run")
parser.add_argument("--img_root", type=str, help="root for image to analyse")
parser.add_argument("--net_model", type=str, help="prototxt of network architecture")
parser.add_argument("--out_layer", type=str, help="last fully connect layer")
parser.add_argument("--crop_size", type=int, help="input image size")
parser.add_argument("--last_conv", type=str, help="last convolution layer")
parser.add_argument("--net_weights", type=str, help="network weights")

args = parser.parse_args()

img_root = args.img_root
net_model = args.net_model
out_layer = args.out_layer
crop_size = args.crop_size
last_conv = args.last_conv
caffe_path = args.caffe_folder
net_weights = args.net_weights

if caffe_path not in sys.path:
    sys.path.insert(1, caffe_path+'python/')
import caffe


def im2double(im):
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

def py_map2jpg(imgmap, rang, colorMap):
    if rang is None:
        rang = [np.min(imgmap), np.max(imgmap)]

    heatmap_x = np.round(imgmap*255).astype(np.uint8)
    return cv2.applyColorMap(heatmap_x, cv2.COLORMAP_JET)

def py_returnCAMmap(activation, weights_LR):
    print(activation.shape)
    if activation.shape[0] == 1: # only one image
        n_feat, w, h = activation[0].shape
        act_vec = np.reshape(activation[0], [n_feat, w*h])
        n_top = weights_LR.shape[0]
        out = np.zeros([w, h, n_top])

        for t in range(n_top):
            weights_vec = np.reshape(weights_LR[t], [1, weights_LR[t].shape[0]])
            heatmap_vec = np.dot(weights_vec,act_vec)
            heatmap = np.reshape( np.squeeze(heatmap_vec) , [w, h])
            out[:,:,t] = heatmap
    else: # 10 images (over-sampling)
        raise Exception('Not implemented')
    return out

net = caffe.Net(net_model, net_weights, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))

weights_LR = net.params[out_layer][0].data
image = cv2.imread(img_root)
print image
image = cv2.resize(image, (crop_size, crop_size))

# extract conv features
net.blobs['data'].reshape(*np.asarray([1,3,crop_size,crop_size])) # run only one image
net.blobs['data'].data[...][0,:,:,:] = transformer.preprocess('data', image)
out = net.forward()
scores = out['prob']
activation_lastconv = net.blobs[last_conv].data

topNum = 5 # generate heatmap for top X prediction results
scoresMean = np.mean(scores, axis=0)
ascending_order = np.argsort(scoresMean)
IDX_category = ascending_order[::-1] # [::-1] to sort in descending order

curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[IDX_category[:topNum],:])

curResult = im2double(image)

for j in range(topNum):
    # for one image
    curCAMmap_crops = curCAMmapAll[:,:,j]
    curCAMmapLarge_crops = cv2.resize(curCAMmap_crops, (crop_size,crop_size))
    curHeatMap = cv2.resize(im2double(curCAMmapLarge_crops),(crop_size,crop_size)) # this line is not doing much
    curHeatMap = im2double(curHeatMap)

    curHeatMap = py_map2jpg(curHeatMap, None, 'jet')
    curHeatMap = im2double(image)*0.2+im2double(curHeatMap)*0.4


    print "output results: "+str(IDX_category[j])+'_' + img_root.split('/')[-1]
    cv2.imwrite(str(IDX_category[j])+'_' + img_root.split('/')[-1],curHeatMap)
    # cv2.waitKey(0)
cv2.imwrite(img_root.split('/')[-1], image)