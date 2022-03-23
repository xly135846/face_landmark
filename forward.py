import tensorflow as tf
import numpy as np
import cv2
from glob import glob
import datetime
import time
from numpy import *
import os
from scipy import ndimage

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

def opencv_det(classfier, gray):
    h, w = np.shape(gray)
    s = 1
    gray_1 = cv2.resize(gray, (int(w / s), int(h / s)))
    rects = classfier.detectMultiScale(gray_1, scaleFactor=1.2, minNeighbors=3, minSize=(60, 60))
    if len(rects) > 0:
        x, y, w, h = rects[0]
        det = [x, y, x + w, y + h]
        det = [s * i for i in det]
        return det
    else:
        return None

def test_single_pic():
    graph = load_graph(r"G:\yfgu\test_single_pic\dde_graph_48_48_96_192_hyper100_2224653.pb")
    config = tf.ConfigProto(log_device_placement=False)
    sess = tf.Session(graph=graph, config=config)

    img = cv2.imread(r"G:\yfgu\test_single_pic\test3.jpg")
    gray = cv2.imread(r"G:\yfgu\test_single_pic\test3.jpg", 0)
    det = [140, 1, 400, 261]
    crop_frame = gray[det[1]:det[3], det[0]:det[2]]
    frame_resize = cv2.resize(crop_frame, (int(112), int(112)), interpolation=cv2.INTER_LINEAR)

    factor_s = [float((det[3] - det[1]) / 112.), float((det[2] - det[0]) / 112.)]
    I = np.expand_dims(frame_resize, axis=2)
    I = ((I - 127.5) / 127.5).reshape((-1, int(112), int(112), 1))

    Landmark, vis = sess.run(['prefix/add:0', 'prefix/vis:0'], {'prefix/input:0': I})
    Landmark = np.reshape(Landmark, [75, 2])

    Landmark[:, 0] = Landmark[:, 0] * factor_s[1] + det[0]
    Landmark[:, 1] = Landmark[:, 1] * factor_s[0] + det[1]

    # f = open("./test_landmark.pts", 'w')
    # f.write(str(Landmark))
    # f.close()

    Frame = img / 255.
    for i in range(75):
        if vis[0, i] < 0.5:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        center = (int(round(Landmark[i, 0] * 16)), int(round(Landmark[i, 1] * 16)))
        cv2.circle(Frame, center, 10, color, 1, shift=4)
    cv2.imshow('src', Frame)
    cv2.imshow("sad", crop_frame / 255.)
    cv2.waitKey(0)

def run_pb_stage2_image():
    def softmax(X):
        return np.exp(X) / np.sum(np.exp(X))
    def cornerHarris(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)  # 图像膨胀
        # img[dst>0.00000001*dst.max()]=[0,0,255] #可以试试这个参数，角点被标记的多余了一些
        img[dst > 0.01 * dst.max()] = [0, 255, 0]  # 角点位置用红色标记
        return img
    def sobel_demo(image):
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)  # 对x求一阶导
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)  # 对y求一阶导
        gradx = cv2.convertScaleAbs(grad_x)  # 用convertScaleAbs()函数将其转回原来的uint8形式
        grady = cv2.convertScaleAbs(grad_y)
        gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)  # 图片融合
        return gradxy
    def bestFitRect(box, meanS):
        boxCenter = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

        boxWidth = box[2] - box[0]
        boxHeight = box[3] - box[1]

        meanShapeWidth = meanS[:, 0].max() - meanS[:, 0].min()
        meanShapeHeight = meanS[:, 1].max() - meanS[:, 1].min()

        scaleWidth = boxWidth / meanShapeWidth
        scaleHeight = boxHeight / meanShapeHeight
        scale = (scaleWidth + scaleHeight) / 2

        S0 = meanS * scale

        S0Center = [(S0[:, 0].min() + S0[:, 0].max()) / 2, (S0[:, 1].min() + S0[:, 1].max()) / 2]
        S0 += boxCenter - S0Center

        return S0
    def transform(form, to):
        destMean = np.mean(to, axis=0)
        srcMean = np.mean(form, axis=0)

        srcVec = (form - srcMean).flatten()
        destVec = (to - destMean).flatten()

        a = np.dot(srcVec, destVec) / np.linalg.norm(srcVec) ** 2
        b = 0
        for i in range(form.shape[0]):
            b += srcVec[2 * i] * destVec[2 * i + 1] - srcVec[2 * i + 1] * destVec[2 * i]
        b = b / np.linalg.norm(srcVec) ** 2

        T = np.array([[a, b], [-b, a]])
        srcMean = np.dot(srcMean, T)

        return T, destMean - srcMean
    def minbox(Landmark):
        Landmark = Landmark.astype(int)
        return [min(Landmark[:, 0]), min(Landmark[:, 1]), max(Landmark[:, 0]), max(Landmark[:, 1])]
    def det_face_api(sess, image, size, rect):
        IMG_SIZE = 112
        ##crop###########
        if 1:
            top_left = False
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if (rect[0] < 0 or rect[1] < 0 or rect[2] > size[0] or rect[3] > size[1]):
                if rect[0] < 0 or rect[1] < 0:
                    boader = [max(0, -rect[0]), max(0, -rect[1]), max(rect[2] - size[0], 0), max(rect[3] - size[1], 0)]
                    rect += boader
                    top_left = True
                else:
                    boader = [max(0, -rect[0]), max(0, -rect[1]), max(rect[2] - size[0], 0), max(rect[3] - size[1], 0)]

                gray = cv2.copyMakeBorder(gray, boader[1], boader[3], boader[0], boader[2], cv2.BORDER_CONSTANT,
                                          value=(0))

            crop_frame = gray[rect[1]:rect[3], rect[0]:rect[2]]
            frame_resize = cv2.resize(crop_frame, (int(IMG_SIZE), int(IMG_SIZE)), interpolation=cv2.INTER_LINEAR)
            factor_s = [float((det[3] - det[1]) / IMG_SIZE), float((det[2] - det[0]) / IMG_SIZE)]

            img = np.expand_dims(frame_resize, axis=2)
            I = ((img - 127.5) / 127.5).reshape((-1, int(IMG_SIZE), int(IMG_SIZE), 1))

            fstart = datetime.datetime.now()
            Landmark, vis, detect = sess.run(['prefix/add:0', 'prefix/vis:0', 'prefix/detect:0'], {'prefix/input:0': I})
            fend = datetime.datetime.now()
            duration = fend - fstart

            p = softmax(detect)[0][0]
            Landmark = np.reshape(Landmark, [75, 2])

            Landmark[:, 0] = Landmark[:, 0] * factor_s[1] + det[0]
            Landmark[:, 1] = Landmark[:, 1] * factor_s[0] + det[1]
            if top_left is True:
                Landmark -= np.array([boader[0], boader[1]])
        ##crop###########
        if 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            FitMeanshape = bestFitRect(rect, MeanShape)
            R, T = transform(FitMeanshape, MeanShape)
            R = np.linalg.inv(R)
            T = np.dot(-T, R)
            crop_frame = ndimage.interpolation.affine_transform(gray, R, T[[1, 0]], output_shape=(112, 112))

            img = np.expand_dims(crop_frame, axis=2)
            I = ((img - 127.5) / 127.5).reshape((-1, int(IMG_SIZE), int(IMG_SIZE), 1))

            fstart = datetime.datetime.now()
            Landmark, vis, detect = sess.run(['prefix/add:0', 'prefix/vis:0', 'prefix/detect:0'], {'prefix/input:0': I})
            fend = datetime.datetime.now()
            p = softmax(detect)[0][0]
            duration = fend - fstart
            Landmark = np.reshape(Landmark, [-1, 2])
            Landmark = np.dot(Landmark, R) + T

        return Landmark, p, vis, duration, crop_frame
    def det_mouth_api(sess_m, image, landmark):
        mouth = landmark[46:64]
        rect = minbox(mouth)
        maxsize = int(max((rect[2] - rect[0]), (rect[3] - rect[1])) * 5 / 8)
        centor = [int((rect[3] + rect[1]) / 2), int((rect[2] + rect[0]) / 2)]

        det_mouth = [centor[1] - maxsize, centor[0] - maxsize]
        factor_s_mouth = [(maxsize + maxsize) / 40., (maxsize + maxsize) / 40.]

        crop = image[centor[0] - maxsize:centor[0] + maxsize, centor[1] - maxsize:centor[1] + maxsize]
        resize = cv2.resize(crop, (int(40), int(40)), interpolation=cv2.INTER_LINEAR)
        I = (resize - 127.5) / 127.5
        start = time.time()
        lm_mouth36 = sess_m.run(['prefix/add_mouth:0'], {'prefix/input_mouth:0': [I]})
        end = time.time()
        lm_mouth = np.reshape(lm_mouth36, [64, 2])
        lm_mouth[:, 0] = lm_mouth[:, 0] * factor_s_mouth[1] + det_mouth[0]
        lm_mouth[:, 1] = lm_mouth[:, 1] * factor_s_mouth[0] + det_mouth[1]
        return lm_mouth, end - start
    def det_eye_api(sess_e, image, landmark):
        def crop_resize(patch_landmark):
            rect = minbox(patch_landmark)
            maxsize = int(max((rect[2] - rect[0]), (rect[3] - rect[1])) * 5 / 8)
            centor = [int((rect[3] + rect[1]) / 2), int((rect[2] + rect[0]) / 2)]
            det = [centor[1] - maxsize, centor[0] - maxsize]
            factor = [(maxsize + maxsize) / 40., (maxsize + maxsize) / 40.]
            crop = image[centor[0] - maxsize:centor[0] + maxsize, centor[1] - maxsize:centor[1] + maxsize]
            resize = cv2.resize(crop, (int(40), int(40)), interpolation=cv2.INTER_LINEAR)
            I = (resize - 127.5) / 127.5
            return I, factor, det

        left_eye = landmark[[31, 32, 33, 34, 69, 70, 71, 72, 74]]
        right_eye = landmark[[27, 28, 29, 30, 65, 66, 67, 68]]
        left_I, left_factor, left_det = crop_resize(left_eye)
        right_I, right_factor, right_det = crop_resize(right_eye)
        h, w, c = np.shape(right_I)
        right_I_mirror = cv2.flip(right_I, 1)

        start = time.time()
        output = sess_e.run(['prefix/add_eye:0'], {'prefix/input_eye:0': [left_I, right_I_mirror]})
        left_lm_eye22 = output[0][0]
        right_lm_eye22 = output[0][1]
        # right_lm_eye22 = sess_e.run(['prefix/add_eye:0'], {'prefix/input_eye:0': [right_I_mirror]})
        end = time.time()

        left_lm_eye = np.reshape(left_lm_eye22, [22, 2])
        left_lm_eye[:, 0] = left_lm_eye[:, 0] * left_factor[1] + left_det[0]
        left_lm_eye[:, 1] = left_lm_eye[:, 1] * left_factor[0] + left_det[1]

        right_lm_eye = np.reshape(right_lm_eye22, [22, 2])
        right_lm_eye[:, 0] = w - right_lm_eye[:, 0]
        right_lm_eye[:, 0] = right_lm_eye[:, 0] * right_factor[1] + right_det[0]
        right_lm_eye[:, 1] = right_lm_eye[:, 1] * right_factor[0] + right_det[1]

        return left_lm_eye, right_lm_eye, end - start
    def det_eyebrow_api(sess_eb, image, landmark):
        def crop_resize(patch_landmark):
            rect = minbox(patch_landmark)
            maxsize = int(max((rect[2] - rect[0]), (rect[3] - rect[1])) * 5 / 8)
            centor = [int((rect[3] + rect[1]) / 2), int((rect[2] + rect[0]) / 2)]
            det = [centor[1] - maxsize, centor[0] - maxsize]
            factor = [(maxsize + maxsize) / 40., (maxsize + maxsize) / 40.]
            crop = image[centor[0] - maxsize:centor[0] + maxsize, centor[1] - maxsize:centor[1] + maxsize]
            resize = cv2.resize(crop, (int(40), int(40)), interpolation=cv2.INTER_LINEAR)
            I = (resize - 127.5) / 127.5
            return I, factor, det

        left_eyebrow = landmark[15:21]
        right_eyebrow = landmark[21:27]
        left_I, left_factor, left_det = crop_resize(left_eyebrow)
        right_I, right_factor, right_det = crop_resize(right_eyebrow)
        h, w, c = np.shape(right_I)
        right_I_mirror = cv2.flip(right_I, 1)

        start = time.time()
        output = sess_eb.run(['prefix/add_eyebrow:0'], {'prefix/input_eyebrow:0': [left_I, right_I_mirror]})
        left_lm_eyebrow13 = output[0][0]
        right_lm_eyebrow13 = output[0][1]
        end = time.time()

        left_lm_eyebrow = np.reshape(left_lm_eyebrow13, [13, 2])
        left_lm_eyebrow[:, 0] = left_lm_eyebrow[:, 0] * left_factor[1] + left_det[0]
        left_lm_eyebrow[:, 1] = left_lm_eyebrow[:, 1] * left_factor[0] + left_det[1]

        right_lm_eyebrow = np.reshape(right_lm_eyebrow13, [13, 2])
        right_lm_eyebrow[:, 0] = h - right_lm_eyebrow[:, 0]
        right_lm_eyebrow[:, 0] = right_lm_eyebrow[:, 0] * right_factor[1] + right_det[0]
        right_lm_eyebrow[:, 1] = right_lm_eyebrow[:, 1] * right_factor[0] + right_det[1]
        return left_lm_eyebrow, right_lm_eyebrow, end - start, left_I, right_I_mirror

    ######################config#######################
    det_mouth = 0
    det_eye = 0
    det_eyebrow = 0
    save = 0
    ######################init####################
    config = tf.ConfigProto(log_device_placement=False)
    # graph = load_graph(r".\model\dde_graph_48_48_96_192_sd_d_v_ec_1253893.pb")  # detect
    graph = load_graph(r".\model\dde_graph_48_48_96_192_detect_finall1967750.pb")  # detect
    sess = tf.Session(graph=graph, config=config)
    if det_mouth:
        graph_mouth = load_graph(r".\model\dde_graph_48_48_mouth_1938533.pb")  # mouth
        sess_m = tf.Session(graph=graph_mouth, config=config)
    if det_eye:
        graph_eye = load_graph(r".\model\dde_graph_48_48_eye_951968.pb")  # eye
        sess_e = tf.Session(graph=graph_eye, config=config)
    if det_eyebrow:
        graph_eyebrow = load_graph(r".\model\dde_graph_48_48_eyebrow_949308.pb")  # eyebrow
        sess_eb = tf.Session(graph=graph_eyebrow, config=config)

    # file = np.load(r".\model\dde_meanshape112.npz")
    # file = np.load(r".\model\dde_meanshape112_sd.npz")
    # MeanShape = np.array(file["Meanshape"])

    classfier = cv2.CascadeClassifier("./model/haarcascade_frontalface_alt2.xml")
    imgs = glob(r"./imgs/*.jpg")
    # imgs = [r"E:\yfgu\test_single_pic\czy03418_80.jpg"]
    key = 1

    for path in imgs:
        print("process ", path)
        frame = cv2.imread(path)
        size = np.shape(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det = opencv_det(classfier, gray)
        if det is not None:
            max_side = max(det[2] - det[0], det[3] - det[1]) * 5 / 8
            rect_centor = [(det[0] + det[2]) / 2, (det[1] + det[3]) / 2]
            rect_centor[1] += (det[3] - det[1]) / 10
            det = np.array([rect_centor[0] - max_side, rect_centor[1] - max_side, rect_centor[0] + max_side,
                            rect_centor[1] + max_side], dtype=int)

            # max_side = max(det[2] - det[0], det[3] - det[1])*4/10
            # rect_centor = [(det[0] + det[2]) / 2, (det[1] + det[3]) / 2]
            # rect_centor[1] += (det[3] - det[1]) / 10
            # det = np.array([rect_centor[0] - max_side, rect_centor[1] - max_side, rect_centor[0] + max_side,
            #                 rect_centor[1] + max_side], dtype=int)
        else:
            print(path, "no detect")
            continue

        Landmark, p, vis, face_time, crop_frame = det_face_api(sess, frame, size, det)

        if det_mouth:
            lm_mouth, mouth_time = det_mouth_api(sess_m, frame, Landmark)
        if det_eye:
            left_lm_eye, right_lm_eye, eye_time = det_eye_api(sess_e, frame, Landmark)
        if det_eyebrow:
            left_lm_eyebrow, right_lm_eyebrow, eyebrow_time, left_eye_crop, right_eye_crop = det_eyebrow_api(sess_eb,
                                                                                                             frame,
                                                                                                             Landmark)

        ##################draw face points######################
        contour = arange(0, 15)
        nose = np.append(arange(35, 46), 64)
        array = np.concatenate((nose, contour))
        for i in range(75):
            # if vis[0, i] < 0.5:
            #     color = (0, 0, 255)
            # else:
            #     color = (0, 255, 0)
            color = (0, 0, 255)
            center = (int(round(Landmark[i, 0] * 16)), int(round(Landmark[i, 1] * 16)))
            cv2.circle(frame, center, 10, color, 1, shift=4)
            cv2.rectangle(frame, (det[0], det[1]), (det[2], det[3]), (0, 0, 255))
        #################draw mouth #############################
        if det_mouth:
            for i in range(len(lm_mouth)):
                color = (0, 0, 255)
                center = (int(round(lm_mouth[i, 0] * 16)), int(round(lm_mouth[i, 1] * 16)))
                cv2.circle(frame, center, 10, color, 1, shift=4)
        #################draw eye #############################
        if det_eye:
            for i in range(len(left_lm_eye)):
                color = (0, 0, 255)
                center = (int(round(left_lm_eye[i, 0] * 16)), int(round(left_lm_eye[i, 1] * 16)))
                cv2.circle(frame, center, 10, color, 1, shift=4)
                center = (int(round(right_lm_eye[i, 0] * 16)), int(round(right_lm_eye[i, 1] * 16)))
                cv2.circle(frame, center, 10, color, 1, shift=4)
        #################draw eyebrow #############################
        if det_eyebrow:
            for i in range(len(left_lm_eyebrow)):
                color = (0, 0, 255)
                center = (int(round(left_lm_eyebrow[i, 0] * 16)), int(round(left_lm_eyebrow[i, 1] * 16)))
                cv2.circle(frame, center, 10, color, 1, shift=4)
                center = (int(round(right_lm_eyebrow[i, 0] * 16)), int(round(right_lm_eyebrow[i, 1] * 16)))
                cv2.circle(frame, center, 10, color, 1, shift=4)

        cv2.imshow('src', frame)
        cv2.imshow('crop', crop_frame)
        cv2.waitKey(0)
        if save:
            cv2.imwrite(path.replace(path[-4:], '_draw.jpg'), frame)
        key += 1

def run_eyebrow():
    def minbox(Landmark):
        Landmark = Landmark.astype(int)
        return [min(Landmark[:, 0]), min(Landmark[:, 1]), max(Landmark[:, 0]), max(Landmark[:, 1])]
    graph_eyebrow = load_graph(r".\model\dde_graph_48_48_eyebrow_949308.pb")  # eyebrow
    sess_eb = tf.Session(graph=graph_eyebrow)
    frame = cv2.imread(r"G:\data\1_right_eyebrows.png")

    def crop_resize(image, patch_landmark):
        rect = minbox(patch_landmark)
        maxsize = int(max((rect[2] - rect[0]), (rect[3] - rect[1])) * 5 / 8)
        centor = [int((rect[3] + rect[1]) / 2), int((rect[2] + rect[0]) / 2)]
        det = [centor[1] - maxsize, centor[0] - maxsize]
        factor = [(maxsize + maxsize) / 40., (maxsize + maxsize) / 40.]
        crop = image[centor[0] - maxsize:centor[0] + maxsize, centor[1] - maxsize:centor[1] + maxsize]
        resize = cv2.resize(crop, (int(40), int(40)), interpolation=cv2.INTER_LINEAR)
        I = (resize - 127.5) / 127.5
        return I, factor, det

    right_eyebrow = np.array([[255, 388], [389, 301],[544,277],[725,350],[573,346],[400,360]])
    # right_eyebrow[:, [1, 0]] = right_eyebrow[:,[0, 1]]

    right_I, right_factor, right_det = crop_resize(frame, right_eyebrow)
    h, w, c = np.shape(right_I)
    right_I_mirror = cv2.flip(right_I, 1)


    output = sess_eb.run(['prefix/add_eyebrow:0'], {'prefix/input_eyebrow:0': [right_I_mirror]})
    right_lm_eyebrow13 = output[0][0]

    right_lm_eyebrow13 = np.reshape(right_lm_eyebrow13, [13, 2])

    right_lm_eyebrow = np.copy(right_lm_eyebrow13)
    right_lm_eyebrow[:, 0] = h - right_lm_eyebrow[:, 0]
    right_lm_eyebrow[:, 0] = right_lm_eyebrow[:, 0] * right_factor[1] + right_det[0]
    right_lm_eyebrow[:, 1] = right_lm_eyebrow[:, 1] * right_factor[0] + right_det[1]

    frame = (right_I_mirror+1)/2.
    # frame = cv2.resize(frame, (int(400), int(400)), interpolation=cv2.INTER_LINEAR)
    for j in range(13):
        cv2.circle(frame, (int(right_lm_eyebrow13[j, 0]), int(right_lm_eyebrow13[j, 1])), 1, (0, 0, 255), -1)
    cv2.imshow('src', frame)
    cv2.waitKey(0)

def run_pb_image():
    def softmax(X):
        return np.exp(X) / np.sum(np.exp(X))
    def bestFitRect(box, meanS):
        boxCenter = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

        boxWidth = box[2] - box[0]
        boxHeight = box[3] - box[1]

        meanShapeWidth = meanS[:, 0].max() - meanS[:, 0].min()
        meanShapeHeight = meanS[:, 1].max() - meanS[:, 1].min()

        scaleWidth = boxWidth / meanShapeWidth
        scaleHeight = boxHeight / meanShapeHeight
        scale = (scaleWidth + scaleHeight) / 2

        S0 = meanS * scale

        S0Center = [(S0[:, 0].min() + S0[:, 0].max()) / 2, (S0[:, 1].min() + S0[:, 1].max()) / 2]
        S0 += boxCenter - S0Center

        return S0
    def transform(form, to):
        destMean = np.mean(to, axis=0)
        srcMean = np.mean(form, axis=0)

        srcVec = (form - srcMean).flatten()
        destVec = (to - destMean).flatten()

        a = np.dot(srcVec, destVec) / np.linalg.norm(srcVec) ** 2
        b = 0
        for i in range(form.shape[0]):
            b += srcVec[2 * i] * destVec[2 * i + 1] - srcVec[2 * i + 1] * destVec[2 * i]
        b = b / np.linalg.norm(srcVec) ** 2

        T = np.array([[a, b], [-b, a]])
        srcMean = np.dot(srcMean, T)

        return T, destMean - srcMean
    def minbox(Landmark):
        Landmark = Landmark.astype(int)
        return [min(Landmark[:, 0]), min(Landmark[:, 1]), max(Landmark[:, 0]), max(Landmark[:, 1])]
    def mtcnn_dector_init():
        import sys
        sys.path.append('../')
        from MTCNN.Detection.MtcnnDetector import MtcnnDetector
        from MTCNN.Detection.detector import Detector
        from MTCNN.Detection.fcn_detector import FcnDetector
        from MTCNN.train_models.mtcnn_model import P_Net, R_Net, O_Net

        thresh = [0.9, 0.6, 0.7]
        min_face_size = 200
        stride = 5
        slide_window = False
        detectors = [None, None, None]
        prefix = ['MTCNN/data/MTCNN_model/PNet_landmark/PNet', 'MTCNN/data/MTCNN_model/RNet_landmark/RNet',
                  'MTCNN/data/MTCNN_model/ONet_landmark/ONet']
        epoch = [18, 14, 16]
        model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
        PNet = FcnDetector(P_Net, model_path[0])
        detectors[0] = PNet
        RNet = Detector(R_Net, 24, 1, model_path[1])
        detectors[1] = RNet
        ONet = Detector(O_Net, 48, 1, model_path[2])
        detectors[2] = ONet
        mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                       stride=stride, threshold=thresh, slide_window=slide_window)
        return mtcnn_detector
    def det_face_api(sess, image, size, rect):
        ##s1###########
        if 1:
            img_size = 40
            top_left = False
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if (rect[0] < 0 or rect[1] < 0 or rect[2] > size[0] or rect[3] > size[1]):
                if rect[0] < 0 or rect[1] < 0:
                    boader = [max(0, -rect[0]), max(0, -rect[1]), max(rect[2] - size[0], 0), max(rect[3] - size[1], 0)]
                    rect += boader
                    top_left = True
                else:
                    boader = [max(0, -rect[0]), max(0, -rect[1]), max(rect[2] - size[0], 0), max(rect[3] - size[1], 0)]

                gray = cv2.copyMakeBorder(gray, boader[1], boader[3], boader[0], boader[2], cv2.BORDER_CONSTANT, value=(0))

            crop_frame = gray[rect[1]:rect[3], rect[0]:rect[2]]
            cv2.imshow("crop_frame",crop_frame)

            frame_resize = cv2.resize(crop_frame, (int(img_size), int(img_size)), interpolation=cv2.INTER_LINEAR)
            factor_s = [float((det[3] - det[1]) / img_size), float((det[2] - det[0]) / img_size)]
            img = np.expand_dims(frame_resize, axis=2)
            # I = ((img - 127.5) / 127.5).reshape((-1, int(img_size), int(img_size), 1))

            Landmark = sess_light.run(['prefix/landmark:0'], {'prefix/input:0': [img]})
            Landmark = np.reshape(Landmark, [77, 2])

            Landmark[:, 0] = Landmark[:, 0] * factor_s[1] + det[0]
            Landmark[:, 1] = Landmark[:, 1] * factor_s[0] + det[1]
            if top_left is True:
                Landmark -= np.array([boader[0], boader[1]])
        ##s2###########
        IMG_SIZE = 224
        if 1:
            R, T = transform(Landmark, MeanShape)
            frontal = np.dot(Landmark, R) + T
            MeanShapeWidth = MeanShape[:, 1].max() - MeanShape[:, 1].min()
            frontalWidth = (frontal[:, 1].max() - frontal[:, 1].min())
            scale = MeanShapeWidth / frontalWidth
            frontalCenter = [(frontal[:, 0].max() + frontal[:, 0].min()) / 2,
                             (frontal[:, 1].max() + frontal[:, 1].min()) / 2]
            frontal = (frontal - frontalCenter) * scale + frontalCenter
            R, T = transform(Landmark, frontal)
            R = np.linalg.inv(R)
            T = np.dot(-T, R)
            crop_frame = ndimage.interpolation.affine_transform(gray, R, T[[1, 0]], output_shape=(IMG_SIZE, IMG_SIZE))
            img = np.expand_dims(crop_frame, axis=2)
            cv2.imshow("crop_frame",crop_frame)

            I = ((img - 127.5) / 127.5).reshape((-1, int(IMG_SIZE), int(IMG_SIZE), 1))
            Landmark = sess.run(['prefix/add:0'], {'prefix/input:0': I})
            Landmark = np.reshape(Landmark, [-1, 2])
            Landmark = np.dot(Landmark, R) + T
        return Landmark

    save = 0
    mtcnn_detector = mtcnn_dector_init()
    config = tf.ConfigProto(log_device_placement=False)

    graph = load_graph(r".\model\landmark_light77.pb")  # detect
    sess_light = tf.Session(graph=graph, config=config)

    graph = load_graph(r".\model\res224_20210705.pb")  # detect
    sess = tf.Session(graph=graph, config=config)
    file = np.load(r".\model\dde_meanshape64_sin_contour77.npz")
    MeanShape = np.array(file["Meanshape"])
    MeanShape = np.reshape(MeanShape, [-1, 2]) / 64 * 224
    imgs = glob(r"./imgs/*.jpg")
    key = 1

    for path in imgs:
        print("process ", path)
        frame = cv2.imread(path)
        size = np.shape(frame)
        det_boxes, det_landmarks = mtcnn_detector.detect(np.array(frame))
        for i in range(det_boxes.shape[0]):
            det = det_boxes[i, :4]
            if det is not None:
                max_side = max(det[2] - det[0], det[3] - det[1]) * 3 / 8
                rect_centor = [(det[0] + det[2]) / 2, (det[1] + det[3]) / 2]
                rect_centor[1] += (det[3] - det[1]) / 10
                det = np.array([rect_centor[0] - max_side, rect_centor[1] - max_side, rect_centor[0] + max_side,
                                rect_centor[1] + max_side], dtype=int)
            else:
                print(path, "no detect")
                continue
            Landmark = det_face_api(sess, frame, size, det)

            ##################draw face points######################
            contour = arange(0, 15)
            nose = np.append(arange(35, 46), 64)
            array = np.concatenate((nose, contour))
            for i in range(77):
                color = (0, 0, 255)
                center = (int(round(Landmark[i, 0] * 16)), int(round(Landmark[i, 1] * 16)))
                cv2.circle(frame, center, 10, color, 1, shift=4)

        cv2.imshow('src', frame)
        cv2.waitKey(0)
        if save:
            cv2.imwrite(path.replace(path[-4:], '_draw.jpg'), frame)
        key += 1

def run_pb_video():
    def softmax(X):
        return np.exp(X) / np.sum(np.exp(X))
    def bestFitRect(box, meanS):
        boxCenter = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

        boxWidth = box[2] - box[0]
        boxHeight = box[3] - box[1]

        meanShapeWidth = meanS[:, 0].max() - meanS[:, 0].min()
        meanShapeHeight = meanS[:, 1].max() - meanS[:, 1].min()

        scaleWidth = boxWidth / meanShapeWidth
        scaleHeight = boxHeight / meanShapeHeight
        scale = (scaleWidth + scaleHeight) / 2

        S0 = meanS * scale

        S0Center = [(S0[:, 0].min() + S0[:, 0].max()) / 2, (S0[:, 1].min() + S0[:, 1].max()) / 2]
        S0 += boxCenter - S0Center

        return S0
    def transform(form, to):
        destMean = np.mean(to, axis=0)
        srcMean = np.mean(form, axis=0)

        srcVec = (form - srcMean).flatten()
        destVec = (to - destMean).flatten()

        a = np.dot(srcVec, destVec) / np.linalg.norm(srcVec) ** 2
        b = 0
        for i in range(form.shape[0]):
            b += srcVec[2 * i] * destVec[2 * i + 1] - srcVec[2 * i + 1] * destVec[2 * i]
        b = b / np.linalg.norm(srcVec) ** 2

        T = np.array([[a, b], [-b, a]])
        srcMean = np.dot(srcMean, T)

        return T, destMean - srcMean
    def minbox(Landmark):
        Landmark = Landmark.astype(int)
        return [min(Landmark[:, 0]), min(Landmark[:, 1]), max(Landmark[:, 0]), max(Landmark[:, 1])]
    def mtcnn_dector_init():
        import sys
        sys.path.append('../')
        from MTCNN.Detection.MtcnnDetector import MtcnnDetector
        from MTCNN.Detection.detector import Detector
        from MTCNN.Detection.fcn_detector import FcnDetector
        from MTCNN.train_models.mtcnn_model import P_Net, R_Net, O_Net

        thresh = [0.9, 0.6, 0.7]
        min_face_size = 200
        stride = 5
        slide_window = False
        detectors = [None, None, None]
        prefix = ['MTCNN/data/MTCNN_model/PNet_landmark/PNet', 'MTCNN/data/MTCNN_model/RNet_landmark/RNet',
                  'MTCNN/data/MTCNN_model/ONet_landmark/ONet']
        epoch = [18, 14, 16]
        model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
        PNet = FcnDetector(P_Net, model_path[0])
        detectors[0] = PNet
        RNet = Detector(R_Net, 24, 1, model_path[1])
        detectors[1] = RNet
        ONet = Detector(O_Net, 48, 1, model_path[2])
        detectors[2] = ONet
        mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                       stride=stride, threshold=thresh, slide_window=slide_window)
        return mtcnn_detector
    def lm91_lm77(lm):
            lm = np.reshape(np.array(lm),[-1,2])
            return np.concatenate((lm[[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28]],lm[29:]),axis=0)
    def det_face_api(sess, image, rect, Landmark, MeanShape):
        IMG_SIZE = 224
        flag = 0
        # MeanShape = lm91_lm77(MeanShape)
        # if len(Landmark) != 0:
        #     Landmark = lm91_lm77(Landmark)
        if 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(Landmark) == 0:
                FitMeanshape = bestFitRect(rect, MeanShape)
                R, T = transform(FitMeanshape, MeanShape)
            else:
                flag = 1

                R, T = transform(Landmark, MeanShape)
                frontal = np.dot(Landmark, R) + T
                MeanShapeWidth = MeanShape[:, 1].max() - MeanShape[:, 1].min()
                frontalWidth = (frontal[:, 1].max() - frontal[:, 1].min())
                scale = MeanShapeWidth / frontalWidth

                frontalCenter = [(frontal[:, 0].max() + frontal[:, 0].min()) / 2,
                                 (frontal[:, 1].max() + frontal[:, 1].min()) / 2]
                MeanShapeCenter = [(MeanShape[:, 0].max() + MeanShape[:, 0].min()) / 2,
                                   (MeanShape[:, 1].max() + MeanShape[:, 1].min()) / 2]

                frontal = (frontal - frontalCenter) * scale + frontalCenter
                R, T = transform(Landmark, frontal)

            R = np.linalg.inv(R)
            T = np.dot(-T, R)
            crop_frame = ndimage.interpolation.affine_transform(gray, R, T[[1, 0]], output_shape=(IMG_SIZE, IMG_SIZE))

            img = np.expand_dims(crop_frame, axis=2)
            I = ((img - 127.5) / 127.5).reshape((-1, int(IMG_SIZE), int(IMG_SIZE), 1))

            Landmark = sess.run(['prefix/add:0'], {'prefix/input:0': I})
            Landmark = np.reshape(Landmark, [-1, 2])
            Landmark = np.dot(Landmark, R) + T

            if flag:
                Landmark = Landmark

        return Landmark, crop_frame
    def det_face_check(sess, frame, landmark, MeanShape_check):
        # MeanShape_check = lm91_lm77(MeanShape_check)
        # landmark = lm91_lm77(landmark)
        landmark5 = landmark[[74,73,64,52,46]]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        R, T = transform(landmark5, MeanShape_check)
        R = np.linalg.inv(R)
        T = np.dot(-T, R)
        crop_frame = ndimage.interpolation.affine_transform(gray, R, T[[1, 0]], output_shape=(40, 40))
        # crop_frame = cv2.flip(crop_frame, 0)

        cv2.imshow("face_check", crop_frame)
        img = np.expand_dims(crop_frame, axis=2)
        # I = ((img - 127.5) / 127.5).reshape((-1, int(40), int(40), 1))

        output = sess.run(['prefix/detect:0'], {'prefix/input:0': [img]})
        # p = softmax(output[0][0])[0]
        p = output[0][0][0]
        return p

    mtcnn_detector = mtcnn_dector_init()
    config = tf.ConfigProto(log_device_placement=False)
    graph = load_graph(r".\model\res224_20210705.pb")  # detect
    sess = tf.Session(graph=graph, config=config)
    file = np.load(r".\model\dde_meanshape64_sin_contour77.npz")
    MeanShape = np.array(file["Meanshape"])
    MeanShape = np.reshape(MeanShape, [-1, 2]) / 64 * 224

    graph_face_check = load_graph(r".\model\score_model_20210705.pb")
    sess_check = tf.Session(graph=graph_face_check, config=config)
    file = np.load(r".\model\dde_meanshape5.npz")
    MeanShape_check = np.array(file["Meanshape"])

    videos = glob(r".\videos\*\*\*.mp4")

    for n_v,video_path in enumerate(videos):
        if n_v % 1 == 0:
            print("proccess {}/{}".format(n_v,len(videos)))

        if not os.path.exists(video_path):
            print("video path not exist!, continue", video_path)
            continue
        # save_lm = []
        video = cv2.VideoCapture(0)
        success, frame = video.read()
        tracking = False
        key = 0

        while(success):
            if tracking is False:
                Landmark = []
                det_boxes, det_landmarks = mtcnn_detector.detect(np.array(frame))
                if len(det_boxes) > 0:
                    det = det_boxes[0, :4]
                    max_side = max(det[2] - det[0], det[3] - det[1]) * 4 / 8
                    rect_centor = [(det[0] + det[2]) / 2, (det[1] + det[3]) / 2]
                    # rect_centor[1] += (det[3] - det[1]) / 10
                    det = np.array([rect_centor[0] - max_side, rect_centor[1] - max_side, rect_centor[0] + max_side,
                                    rect_centor[1] + max_side], dtype=int)
                else:
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, 'no detect', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    cv2.imshow('src', frame)
                    cv2.waitKey(1)
                    # save_lm.append(np.zeros(77,2))
                    key += 1
                    success, frame = video.read()
                    continue

            Landmark, crop_frame = det_face_api(sess, np.copy(frame), det, Landmark, MeanShape)
            p = det_face_check(sess_check, frame, Landmark, MeanShape_check)
            if p < 0.5:
                tracking = False
            else:
                tracking = True

            ##################draw face points######################
            for i in range(Landmark.shape[0]):
                color = (0, 0, 255)
                center = (int(round(Landmark[i, 0] * 16)), int(round(Landmark[i, 1] * 16)))
                cv2.circle(frame, center, 10, color, 1, shift=4)
            frame = cv2.flip(frame, 1)
            cv2.imshow('src', frame)
            cv2.waitKey(1)

            success, frame = video.read()
            # save_lm.append(Landmark)
            key += 1

        # np.savez(video_path.replace("mp4","npz"),**{"landmark": save_lm})
        # np.save(video_path.replace("mp4","npy"), np.array(save_lm))
        # with open(video_path.replace("mp4","json"), "w") as f:
        #     content = json.dumps(save_lm)
        #     f.write(content)

def interpolate_contour():
    from scipy import interpolate
    def b_inter(mouth, n):
        x = mouth[:, 0]
        y = mouth[:, 1]
        tck, u = interpolate.splprep([x, y], s=5)
        interval = 1. / (6 * n)
        unew = np.arange(0, 1. + interval, interval)
        out = interpolate.splev(unew, tck)
        out = np.transpose(np.array(out), [1, 0])
        return out
    def cal_dis_curve(lms, n):
        interval = int((lms.shape[0] - 1) / n)
        sel_lms = []
        for i in range(lms.shape[0]):
            if i % interval == 0:
                sel_lms.append(lms[i])
        sel_lms[0] = lms[0]
        sel_lms[-1] = lms[-1]
        return np.array(sel_lms)
    def contour_interpolate(lm):
        dense_lm = b_inter(lm, 5000)
        n_lms = cal_dis_curve(dense_lm, 5000)
        return n_lms

    files = glob(r"G:\yfgu\test_single_pic\Img_lmforward_py\videos\*\*\*.npy")
    for n_v, file in enumerate(files):
        if n_v % 10 ==0:
            print("process {}/{}".format(n_v,len(files)))

        landmarks = np.load(file)
        landmarks77 = []
        for key in range(len(landmarks)):
            lm = np.array(landmarks[key])
            if np.sum(lm) == 0:
                landmarks77.append(np.zeros((77,2)))
                continue
            n_lms = contour_interpolate(lm[0:32])
            index = np.argmin(np.linalg.norm(n_lms - lm[16:17], axis=1))
            contour1 = n_lms[:index + 1]
            contour2 = n_lms[index:]
            contour1 = cal_dis_curve(contour1, 7)
            contour2 = cal_dis_curve(contour2, 7)
            Landmark = np.concatenate((contour1, contour2[1:], lm[33:]))
            landmarks77.append(Landmark)
        np.save(file,landmarks77)

def show():
    videos = glob( r"G:\yfgu\test_single_pic\Img_lmforward_py\videos\*\*\*.mp4")
    for n_v, video_path in enumerate(videos):
        if not os.path.exists(video_path):
            print("video path not exist!, continue", video_path)
            continue

        video = cv2.VideoCapture(video_path)
        landmarks = np.load(video_path.replace("mp4","npy"))
        # landmarks = json.loads(video_path.replace("mp4","json"))["landmark"]
        success, frame = video.read()
        key = 0
        while success:
            Landmark = np.array(landmarks[key])

            for i in range(Landmark.shape[0]):
                color = (0, 0, 255)
                center = (int(round(Landmark[i, 0] * 16)), int(round(Landmark[i, 1] * 16)))
                cv2.circle(frame, center, 10, color, 1, shift=4)
            cv2.imshow('src', frame)
            cv2.waitKey(30)
            success, frame = video.read()
            key+=1

if __name__ == "__main__":
    run_pb_video()
    # run_pb_image()