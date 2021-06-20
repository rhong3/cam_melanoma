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
import saliency.tf1 as saliency
from matplotlib import pylab as P
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


# # CAM for real test; no need to determine correct or wrong
# def CAM(net, w, pred, x, path, name, bs, rd=0):
#     DIRR = "../Results/{}/out/{}_img".format(path, name)
#     rd = rd * bs
#
#     try:
#         os.mkdir(DIRR)
#     except(FileExistsError):
#         pass
#
#     pdx = np.asmatrix(pred)
#
#     prl = pdx.argmax(axis=1).astype('uint8')
#
#     for ij in range(len(prl)):
#         id = str(ij + rd)
#         weights_LR = w
#         activation_lastconv = np.array([net[ij]])
#         weights_LR = weights_LR.T
#         activation_lastconv = activation_lastconv.T
#
#         topNum = 1  # generate heatmap for top X prediction results
#         curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[[1], :])
#         for kk in range(topNum):
#             curCAMmap_crops = curCAMmapAll[:, :, kk]
#             curCAMmapLarge_crops = cv2.resize(curCAMmap_crops, (299, 299))
#             curHeatMap = cv2.resize(im2double(curCAMmapLarge_crops), (299, 299))  # this line is not doing much
#             curHeatMap = im2double(curHeatMap)
#             curHeatMap = py_map2jpg(curHeatMap)
#             xim = x[ij].reshape(-1, 3)
#             xim1 = xim[:, 0].reshape(-1, 299)
#             xim2 = xim[:, 1].reshape(-1, 299)
#             xim3 = xim[:, 2].reshape(-1, 299)
#             image = np.empty([299, 299, 3])
#             image[:, :, 0] = xim1
#             image[:, :, 1] = xim2
#             image[:, :, 2] = xim3
#             a = im2double(image) * 255
#             b = im2double(curHeatMap) * 255
#             curHeatMap = a * 0.6 + b * 0.4
#             ab = np.hstack((a, b))
#             full = np.hstack((curHeatMap, ab))
#             # imname = DIRR + '/' + id + '.png'
#             # imname1 = DIRR + '/' + id + '_img.png'
#             # imname2 = DIRR + '/' + id +'_hm.png'
#             imname3 = DIRR + '/' + id + '_full.png'
#             # cv2.imwrite(imname, curHeatMap)
#             # cv2.imwrite(imname1, a)
#             # cv2.imwrite(imname2, b)
#             cv2.imwrite(imname3, full)


def inference(images, num_classes, for_training=False, restore_logits=True,
              scope=None):
  """Build Inception v3 model architecture.

  See here for reference: http://arxiv.org/abs/1512.00567

  Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.

  Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
  """

  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params={'decay': 0.9997, 'epsilon': 0.001}):
      logits, endpoints, net2048, sel_endpoints, netts = slim.inception.inception_v3(
          images,
          dropout_keep_prob=0.8,
          num_classes=num_classes,
          is_training=for_training,
          restore_logits=restore_logits,
          scope=scope)

  # Grab the logits associated with the side head. Employed during training.
  auxiliary_logits = endpoints['aux_logits']

  #return logits, auxiliary_logits
  return logits, auxiliary_logits, endpoints, net2048, sel_endpoints, netts


if __name__ == "__main__":
    # saver = tf.train.import_meta_graph('../model/model.ckpt-31500.meta')

    # print(sess.run('logits/logits/weights:0'))
    # print_tensors_in_checkpoint_file(file_name='../model/model.ckpt-31500', tensor_name='',
    #                                  all_tensors=False, all_tensor_names=False)
    graph = tf.Graph()
    with graph.as_default():
        # image input
        x_in = tf.placeholder(tf.float32, name="x")
        x_in_reshape = tf.reshape(x_in, [-1, 299, 299, 3])
        logits, _, _, _, _, nett = inference(x_in_reshape, 2)
        pred = tf.nn.softmax(logits, name="prediction")
        # prediction = tf.argmax(logits, 1)
        neuron_selector = tf.placeholder(tf.int32)
        y = logits[0][neuron_selector]

        with tf.Session(graph=graph,
                        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.import_meta_graph('../tiles_for_saliency/model.ckpt-99000.meta')
            saver.restore(sess, '../tiles_for_saliency/model.ckpt-99000')
            for dirr in ['top_200_NYU', 'top_200_TCGA',
                         'bottom_200_NYU', 'bottom_200_TCGA']:
                dirpath = str("../tiles_for_saliency/Results/"+dirr)
                try:
                    os.mkdir(dirpath)
                except FileExistsError:
                    pass
                for aa in os.listdir(str("../tiles_for_saliency/"+dirr)):
                    if 'jpeg' in aa and aa not in os.listdir(str("../tiles_for_saliency/Results/"+dirr)):
                        img = cv2.imread(str("../tiles_for_saliency/"+dirr + '/' + aa))
                        img = img.astype(np.float32)
                        # prediction_class = sess.run(
                        #     [prediction], {x_in: img})[0]
                        # weight = sess.run('logits/logits/weights:0')
                        grad = saliency.IntegratedGradients(graph, sess, y, x_in_reshape)
                        # Baseline is a white image.
                        baseline = np.zeros(img.shape)
                        baseline.fill(255)

                        # vanilla_mask_3d = grad.GetMask(img, feed_dict={neuron_selector: 1},
                        #                                x_stepIntegratedGradientss=25, x_baseline=baseline)
                        smoothgrad_mask_3d = grad.GetSmoothedMask(img, feed_dict={
                            neuron_selector: 1}, x_steps=5, x_baseline=baseline, batch_size=1)

                        # Call the visualization methods to convert the 3D tensors to 2D grayscale.
                        # vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
                        smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)

                        print(aa)
                        # vanilla_mask_grayscale = im2double(vanilla_mask_grayscale)
                        # vanilla_mask_grayscale = py_map2jpg(vanilla_mask_grayscale)
                        # a = im2double(img) * 255
                        # b = im2double(vanilla_mask_grayscale) * 255
                        # curHeatMap = a * 0.5 + b * 0.5
                        # ab = np.hstack((a, b))
                        # full = np.hstack((curHeatMap, ab))
                        # cv2.imwrite(str(dirpath + '/' + aa), full)

                        smoothgrad_mask_grayscale = im2double(smoothgrad_mask_grayscale)
                        smoothgrad_mask_grayscale = py_map2jpg(smoothgrad_mask_grayscale)
                        sa = im2double(img) * 255
                        sb = im2double(smoothgrad_mask_grayscale) * 255
                        scurHeatMap = sa * 0.5 + sb * 0.5
                        sab = np.hstack((sa, sb))
                        sfull = np.hstack((scurHeatMap, sab))
                        cv2.imwrite(str(dirpath + '/' + aa), sfull)

    # print(np.shape(x_))
    # print(np.shape(nett_))
    # print(np.shape(pred_))
    # print(np.shape(weight))

    # CAM(nett_, weight, pred_, x_, 'CAM', 'test', bs=100)


