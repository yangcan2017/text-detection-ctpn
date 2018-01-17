#coding:utf-8
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os, sys, cv2
import glob
import shutil
import pytesseract
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg,cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from PIL import Image, ImageFilter
import ImageEnhance
import operator as op

sys.path.append(os.getcwd())


def binarizing(img, threshold):  # input: gray image
    pixdata = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            if pixdata[x, y] < threshold:
                pixdata[x, y] = 0
            else:
                pixdata[x, y] = 255
    return img


def tesser_ocr(inputfile):
    im = Image.open(inputfile)
    if im.size[0] >= 500 or im.size[1] >= 100:
        scale = 1.0
    else:
        scale = max(float(500) / float(im.size[0]), float(100) / float(im.size[1]))
    im_resized = im.resize((int(scale * im.size[0]), int(scale * im.size[1])), Image.ANTIALIAS)

    #将PIL图像转换为opencv图像
    cv2_img = cv2.cvtColor(np.asarray(im_resized), cv2.COLOR_RGB2BGR)
    #求图片清晰度
    imageVar = cv2.Laplacian(cv2_img, cv2.CV_64F).var()
    if imageVar <= 2000:
        im_resized = ImageEnhance.Sharpness(im_resized).enhance(3.0)

    im_gray = im_resized.convert("L")
    im_binary = binarizing(im_gray, 127)

    text = pytesseract.image_to_string(im_binary, lang="chi_sim+eng", config="-psm 7")
    return text


def resize_im(im, scale, max_scale=None):
    f = float(scale)/min(im.shape[0], im.shape[1])
    if max_scale != None and f*max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img, image_name, boxes, scale):
    base_name = image_name.split('/')[-1]
    orgi_img = cv2.resize(img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
    #sort by y1,x1 asc
    boxes = sorted(boxes, key=op.itemgetter(1, 0))
    info = []
    if len(boxes) == 6:
        info = ['name', 'gender&nation', 'birthday', 'address1', 'address2', 'id_no']
    elif len(boxes) == 7:
        info = ['name', 'gender', 'nation', 'birthday', 'address1', 'address2', 'id_no']
        if boxes[1][0] > boxes[2][0]:
            temp = boxes[1]
            boxes[1] = boxes[2]
            boxes[2] = temp
    with open('data/results/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        index = 0
        for box in boxes:
            # if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            #     continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            min_y = min(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
            max_x = max(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            max_y = max(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))

            roi = orgi_img[min_y:max_y, min_x:max_x]
            roi_name = base_name.split('.')[0] + "_" + str(index) + "." + base_name.split('.')[1]
            print(roi_name)
            cv2.imwrite(os.path.join("data/results", roi_name), roi)
            text = tesser_ocr(os.path.join("data/results", roi_name))

            line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y), str(info[index])])+'\r\n'
            f.write(line)
            f.write(text.encode('utf8') + '\r\n')
            index += 1

    img = cv2.resize(img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("data/results", base_name), img)


def ctpn(sess, net, image_name):
    timer = Timer()
    timer.tic()

    img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)

    #将OPENCV图像转换为PIL图像，
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #求图片清晰度
    imageVar = cv2.Laplacian(img, cv2.CV_64F).var()
    if imageVar <= 5000:
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(3.0)
    #将PIL图像转换为opencv图像
    img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

    scores, boxes = test_ctpn(sess, net, img)

    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    draw_boxes(img, image_name, boxes, scale)
    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))


if __name__ == '__main__':
    if os.path.exists("data/results/"):
        shutil.rmtree("data/results/")
    os.makedirs("data/results/")

    cfg_from_file('ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        ctpn(sess, net, im_name)

