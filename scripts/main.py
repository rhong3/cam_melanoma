"""
Outputing CAM of tiles

Created on 04/21/2020

@author: RH

"""
import os
import numpy as np
import cv2
import tensorflow as tf
import data_input
from slim import slim
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


# format activation and weight to get heatmap
def py_returnCAMmap(activation, weights_LR):
    n_feat, w, h, n = activation.shape
    act_vec = np.reshape(activation, [n_feat, w*h])
    n_top = weights_LR.shape[0]
    out = np.zeros([w, h, n_top])

    for t in range(n_top):
        weights_vec = np.reshape(weights_LR[t], [1, weights_LR[t].shape[0]])
        heatmap_vec = np.dot(weights_vec,act_vec)
        heatmap = np.reshape(np.squeeze(heatmap_vec), [w, h])
        out[:, :, t] = heatmap
    return out


# image to double
def im2double(im):
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


# image to jpg
def py_map2jpg(imgmap):
    heatmap_x = np.round(imgmap*255).astype(np.uint8)
    return cv2.applyColorMap(heatmap_x, cv2.COLORMAP_JET)


# CAM for real test; no need to determine correct or wrong
def CAM(net, w, pred, x, path, name, bs, rd=0):
    DIRR = "../Results/{}/out/{}_img".format(path, name)
    rd = rd * bs

    try:
        os.mkdir(DIRR)
    except(FileExistsError):
        pass

    pdx = np.asmatrix(pred)

    prl = pdx.argmax(axis=1).astype('uint8')

    for ij in range(len(prl)):
        id = str(ij + rd)
        weights_LR = w
        activation_lastconv = np.array([net[ij]])
        weights_LR = weights_LR.T
        activation_lastconv = activation_lastconv.T

        topNum = 1  # generate heatmap for top X prediction results
        curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[[1], :])
        for kk in range(topNum):
            curCAMmap_crops = curCAMmapAll[:, :, kk]
            curCAMmapLarge_crops = cv2.resize(curCAMmap_crops, (299, 299))
            curHeatMap = cv2.resize(im2double(curCAMmapLarge_crops), (299, 299))  # this line is not doing much
            curHeatMap = im2double(curHeatMap)
            curHeatMap = py_map2jpg(curHeatMap)
            xim = x[ij].reshape(-1, 3)
            xim1 = xim[:, 0].reshape(-1, 299)
            xim2 = xim[:, 1].reshape(-1, 299)
            xim3 = xim[:, 2].reshape(-1, 299)
            image = np.empty([299, 299, 3])
            image[:, :, 0] = xim1
            image[:, :, 1] = xim2
            image[:, :, 2] = xim3
            a = im2double(image) * 255
            b = im2double(curHeatMap) * 255
            curHeatMap = a * 0.6 + b * 0.4
            ab = np.hstack((a, b))
            full = np.hstack((curHeatMap, ab))
            # imname = DIRR + '/' + id + '.png'
            # imname1 = DIRR + '/' + id + '_img.png'
            # imname2 = DIRR + '/' + id +'_hm.png'
            imname3 = DIRR + '/' + id + '_full.png'
            # cv2.imwrite(imname, curHeatMap)
            # cv2.imwrite(imname1, a)
            # cv2.imwrite(imname2, b)
            cv2.imwrite(imname3, full)


if __name__ == "__main__":
    filename = '../Results/test.tfrecords'
    datasets = data_input.DataSet(100, filename=filename)
    itr, file, ph = datasets.data()
    next_element = itr.get_next()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(itr.initializer, feed_dict={ph: file})
        saver = tf.train.import_meta_graph('../model/model.ckpt-31500.meta')
        saver.restore(sess, '../model/model.ckpt-31500')
        # print(sess.run('logits/logits/weights:0'))
        print_tensors_in_checkpoint_file(file_name='../model/model.ckpt-31500', tensor_name='',
                                         all_tensors=False, all_tensor_names=False)
        # graph = tf.get_default_graph()
        # x = sess.run(next_element)
        # feed_dict = {images: x, num_classes: 2}

        # with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
        #     with slim.arg_scope([slim.ops.conv2d],
        #                         stddev=0.1,
        #                         activation=tf.nn.relu,
        #                         batch_norm_params={'decay': 0.9997, 'epsilon': 0.001}):
        #         logits, endpoints, net2048, sel_endpoints = slim.inception.inception_v3(
        #               x,
        #               dropout_keep_prob=0.8,
        #               num_classes=2,
        #               is_training=False,
        #               restore_logits=True,
        #               scope=None)

