import os
import sys
import subprocess
import math
from math import exp

import cv2
import yaml
import numpy as np
from functools import reduce

try:
    from openvino.inference_engine import IECore
except ModuleNotFoundError:
    raise ModuleNotFoundError('Failed to import \'OpenVINO\'. Install OpenVINO and set environment variables. https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html')


class ov_model:
    ie = IECore()
    available_devices  = [ 'CPU', 'GPU', 'MYRIAD', 'HDDL', 'FPGA', 'GNA' ]
    available_vdevices = [ 'HETERO', 'MULTI' ]
    model_categories   = [ 'public', 'intel' ]
    model_precision    = 'FP16'

    def __init__(self, device='CPU', model=None, **kwargs):
        self.ie     = ov_model.ie        # not deep copy, use given object
        self.net    = None
        self.exenet = None
        self.iblob = []
        self.oblob = []
        self.modelDir = '.'
        self.setDevice(device)
        self.postprocess_params = kwargs['kwargs']
        self.labels = None
        if not model is None:
            self.loadModel(model)

    def __del__(self):
        del self.exenet
        del self.net

    def checkDevice(self, device):
        """
        Check if the device descriptor is acceptable by IE.  
        Args:  
          device (string) : Inference device descriptor for IE
        Return:
          True or False : True = descriptor is acceptable
        """
        if ':' in device:
            vdevice, devices = device.split(':')
            if not vdevice in ov_model.available_vdevices:
                return False
            devices = devices.split(',')
            for device in devices:
                if self.checkDevice(device) == False:
                    return False
        elif not device in ov_model.available_devices:
            return False
        return True

    def setDevice(self, device='CPU'):
        """
        Set inference device.  
        Args:  
          device (string) : OpenVINO Inference engine acceptable device descriptor. ('CPU', 'GPU', 'MYRIAD', 'MULTI:CPU,GPU', ...)
        Return:
          None
        """
        if self.checkDevice(device) == False:
            raise Exception('Not supported device ({})'.format(device))
        self.device = device
        if not self.net is None:
            del self.exenet
            self.exenet = self.ie.load_network(self.net, self.device, num_requests=4)

    def getInterfaceInfo(self):
        """
        Obtain information of input and output blob of the model
        Args:
         None
        Returns:
         iblob/oblob : [ [ 'name':name0, 'shape':shape0, 'precision':precision0 ], ...]
        """
        if self.net is None or self.exenet is None:
            return
        self.iblob = [ {'name':bname, 'shape':self.net.input_info[bname].tensor_desc.dims, 'precision':self.net.input_info[bname].precision } for bname in self.net.input_info ]
        self.oblob = [ {'name':bname, 'shape':self.net.outputs[bname].shape, 'precision':self.net.outputs[bname].precision } for bname in self.net.outputs ]

    def loadLabel(self, labelFile):
        """
        Load label file.  
        Args:  
          labelFile (string) : Label file name
        Return:
          None
        """
        self.labels = None
        if os.path.isfile(labelFile):
            with open(labelFile, 'rt') as f:
                self.labels = [ line.rstrip('\n') for line in f ]

    def loadModel(self, modelFile):
        """
        Read IR model and load the model to IE.  
        This function will search the model location under `./public` and `./intel`.  
        Args:  
          model : IR model file name without path ('mmmmm.xml')
        Return:  
          None
        """
        self.net    = None
        self.exenet = None
        base, ext = os.path.splitext(modelFile)
        for modelcat in ov_model.model_categories:
            model_dir = os.path.join(self.modelDir, modelcat, base)
            if os.path.isdir(model_dir):
                modelfile = os.path.join(model_dir, ov_model.model_precision, base)
                self.net    = self.ie.read_network(modelfile+'.xml', modelfile+'.bin')
                self.exenet = self.ie.load_network(self.net, self.device, num_requests=4)
                self.getInterfaceInfo()

    def inference(self, ocvimg):
        """
        Do inference.  
        Args:  
          ocvimg : OpenCV input image for inference.
        Return:  
          res : Inference result returned by OpenVINO IE
        """
        iblobName = self.iblob[0]['name']
        N,C,H,W = self.iblob[0]['shape']
        img = cv2.resize(ocvimg, (W, H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose( (2,0,1) )
        img = img.reshape( (1,C,H,W) )
        res = self.exenet.infer( { iblobName : img })
        return res


    # --------------------------- Parsing Algorithms

    def bbox_IOU(self, bbox1, bbox2):
        """
        Calculate IOU of 2 bboxes. bboxes are in SSD format (7 elements)  
        bbox = [id, cls, prob, x1, y1, x2, y2]  
        Args:
            bbox1 (bbox)
            bbox2 (bbox)
        Returns:
            IOU value
        """
        _xmin, _ymin, _xmax, _ymax = [ 3, 4, 5, 6 ]
        width_of_overlap_area  = min(bbox1[_xmax], bbox2[_xmax]) - max(bbox1[_xmin], bbox2[_xmin])
        height_of_overlap_area = min(bbox1[_ymax], bbox2[_ymax]) - max(bbox1[_ymin], bbox2[_ymin])
        if width_of_overlap_area < 0 or height_of_overlap_area < 0:
            area_of_overlap = 0
        else:
            area_of_overlap = width_of_overlap_area * height_of_overlap_area
        bbox1_area = (bbox1[_ymax] - bbox1[_ymin]) * (bbox1[_xmax] - bbox1[_xmin])
        bbox2_area = (bbox2[_ymax] - bbox2[_ymin]) * (bbox2[_xmax] - bbox2[_xmin])
        area_of_union = bbox1_area + bbox2_area - area_of_overlap
        if area_of_union == 0:
            return 0
        return area_of_overlap / area_of_union

    def bbox_NMS(self, bboxes, iou_threshold=0.7):
        """
        Perform non maximum suppression for bboxes to reject redundunt detections.  
        bbox = [id, cls, prob, x1, y1, x2, y2]  
        Args:
            bboxes ([bbox,...]):
            iou_threshold (float): Threshold value of rejection
        Returns:
            NMS applied bboxes
        """
        _clsid, _prob = [ 1, 2 ]
        bboxes = sorted(bboxes, key=lambda x: x[_prob], reverse=True)
        for i in range(len(bboxes)):
            if bboxes[i][_prob] == -1:
                continue
            for j in range(i + 1, len(bboxes)):
                iou = self.bbox_IOU(bboxes[i], bboxes[j])
                if iou > iou_threshold:
                    bboxes[j][_prob] = -1
        res = [ bbox for bbox in bboxes if bbox[_prob]!=-1 ]
        return res

    def parse_yolo_region_v3(self, blob, resized_image_shape, params, threshold):
        """
        Parse YOLO region. This function is intented to be called from decode_yolo_result().
        Args:
            blob               : An output blob of YOLO model inference result (one blob only).
            resized_image_shape: Shape information of the resized input image.
            params (dict)      : YOLO parameters to decode the result
            threshold (float)  : Threshold value for object rejection
        Returns:
            objs ([bbox]): bbox = [id, clsId, prob, x1, y1, x2, y2]
        """
        def entry_index(side, coord, classes, location, entry):
            side_power_2 = side ** 2
            n = location // side_power_2
            loc = location % side_power_2
            return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)

        def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
            xmin = int((x - w / 2) * w_scale)
            ymin = int((y - h / 2) * h_scale)
            xmax = int(xmin + w * w_scale)
            ymax = int(ymin + h * h_scale)
            return [class_id, confidence, xmin, ymin, xmax, ymax]

        param_num     = 3  if 'num'     not in params else int(params['num'])
        param_coords  = 4  if 'coords'  not in params else int(params['coords'])
        param_classes = 80 if 'classes' not in params else int(params['classes'])
        param_side    = int(params['side'])
        if 'anchors' not in params:
            anchors = [ 10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0 ]
        else:
            anchors = [ float(anchor) for anchor in params['anchors'].split(',') ]

        if 'mask' not in params:
            param_anchors  = anchors
            param_isYoloV3 = False
        else:
            if params['mask'] == '':
                param_anchors  = anchors
                param_isYoloV3 = False
            else:
                masks          = [ int(m) for m in params['mask'].split(',')]
                param_num      = len(masks)
                param_anchors  = [ [anchors[mask*2], anchors[mask*2+1]] for mask in masks ]
                param_isYoloV3 = True

        out_blob_h, out_blob_w = blob.shape[-2:]

        resized_image_h, resized_image_w = resized_image_shape
        objects = list()
        predictions = blob.flatten()
        side_square = param_side * param_side

        for i in range(side_square):
            row = i // param_side
            col = i % param_side
            for n in range(param_num):
                obj_index = entry_index(param_side, param_coords, param_classes, n * side_square + i, param_coords)
                scale = predictions[obj_index]
                if scale < threshold:
                    continue
                box_index = entry_index(param_side, param_coords, param_classes, n * side_square + i, 0)

                x = (col + predictions[box_index + 0 * side_square]) / param_side
                y = (row + predictions[box_index + 1 * side_square]) / param_side
                try:
                    w_exp = exp(predictions[box_index + 2 * side_square])
                    h_exp = exp(predictions[box_index + 3 * side_square])
                except OverflowError:
                    continue
                w = w_exp * param_anchors[n][0] / (resized_image_w if param_isYoloV3 else param_side)
                h = h_exp * param_anchors[n][1] / (resized_image_h if param_isYoloV3 else param_side)
                for j in range(param_classes):
                    class_index = entry_index(param_side, param_coords, param_classes, n * side_square + i,
                                            param_coords + 1 + j)
                    confidence = scale * predictions[class_index]
                    if confidence < threshold:
                        continue
                    objects.append([0., j, confidence, x-w/2, y-h/2, x+w/2, y+h/2])
        return objects


    def softmax_channel(self, data):
        for i in range(0, len(data), 2):
            m=max(data[i], data[i+1])
            data[i  ] = math.exp(data[i  ]-m)
            data[i+1] = math.exp(data[i+1]-m)
            s=data[i  ]+data[i+1]
            data[i  ]/=s
            data[i+1]/=s
        return data

    def findRoot(self, point, group_mask):
        root = point
        update_parent = False
        while group_mask[root] != -1:
            root = group_mask[root]
            update_parent = True
        if update_parent:
            group_mask[point] = root
        return root

    def join(self, p1, p2, group_mask):
        root1 = self.findRoot(p1, group_mask)
        root2 = self.findRoot(p2, group_mask)
        if root1 != root2:
            group_mask[root1] = root2

    def get_all(self, points, w, h, group_mask):
        root_map = {}
        mask = np.zeros((h, w), np.int32)
        for px, py in points:
            point_root = self.findRoot(px+py*w, group_mask)
            if not point_root in root_map:
                root_map[point_root] = len(root_map)+1
            mask[py, px] = root_map[point_root]
        return mask

    def decodeImageByJoin(self, segm_data, segm_data_shape, link_data, link_data_shape, segm_conf_thresh, link_conf_thresh):
        h = segm_data_shape[1]
        w = segm_data_shape[2]
        pixel_mask = np.full((h*w,), False, dtype=np.bool)
        group_mask = {}
        points     = []
        for i, segm in enumerate(segm_data):
            if segm>segm_conf_thresh:
                pixel_mask[i] = True
                points.append((i%w, i//w))
                group_mask[i] = -1
            else:
                pixel_mask[i] = False
        
        link_mask = np.array([ ld>=link_conf_thresh for ld in link_data ])

        neighbours = int(link_data_shape[3])
        for px, py in points:
            neighbor = 0
            for ny in range(py-1, py+1+1):
                for nx in range(px-1, px+1+1):
                    if nx==px and ny==py:
                        continue
                    if nx<0 or nx>=w or ny<0 or ny>=h:
                        continue
                    pixel_value = pixel_mask[ny*w + nx]
                    link_value  = link_mask [py*w + px*neighbours + neighbor ]
                    if pixel_value and link_value:
                        self.join(px+py*w, nx+ny*w, group_mask)
                    neighbor+=1
        return self.get_all(points, w, h, group_mask)

    def maskToBoxes(self, mask, min_area, min_height, image_size):
        _X, _Y = 0, 1 
        bboxes = []
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(mask)
        max_bbox_idx = int(max_val)
        resized_mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)

        for i in range(1, max_bbox_idx+1):
            bbox_mask = np.where(resized_mask==i, 255, 0).astype(np.uint8)
            contours, hierarchy = cv2.findContours(bbox_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours)==0:
                continue
            center, size, angle = cv2.minAreaRect(contours[0])
            if min(size[_X], size[_Y]) < min_height:
                continue
            if size[_X]*size[_Y] < min_area:
                continue
            bboxes.append((center, size, angle))
        return bboxes

    def text_detection_postprocess(self, link, segm, image_size, segm_conf_thresh, link_conf_thresh):
        _N, _C, _H, _W = 0, 1, 2, 3
        kMinArea   = 300
        kMinHeight = 10

        link_shape = link.shape
        link_data_size = reduce(lambda a, b: a*b, link_shape)
        link_data = link.transpose((_N, _H, _W, _C))
        link_data = link_data.flatten()
        link_data = self.softmax_channel(link_data)
        link_data = link_data.reshape((-1,2))[:,1]
        new_link_data_shape = [ link_shape[0], link_shape[2], link_shape[3], link_shape[1]/2 ]

        segm_shape = segm.shape
        segm_data_size = reduce(lambda a, b: a*b, segm_shape)
        segm_data = segm.transpose((_N, _H, _W, _C))
        segm_data = segm_data.flatten()
        segm_data = self.softmax_channel(segm_data)
        segm_data = segm_data.reshape((-1,2))[:,1]
        new_segm_data_shape = [ segm_shape[0], segm_shape[2], segm_shape[3], segm_shape[1]/2 ]

        mask = self.decodeImageByJoin(segm_data, new_segm_data_shape, link_data, new_link_data_shape, 
                                segm_conf_thresh, link_conf_thresh)
        rects = self.maskToBoxes(mask, kMinArea, kMinHeight, image_size)
        return rects

    # Crop image by rotated rectangle from the input image
    def cropRotatedImage(self, image, rect):
        def topLeftPoint(points):
            big_number = 1e10
            _X, _Y = 0, 1
            most_left        = [big_number, big_number]
            almost_most_left = [big_number, big_number]
            most_left_idx        = -1
            almost_most_left_idx = -1
            for i, point in enumerate(points):
                px, py = point
                if most_left[_X]>px:
                    if most_left[_X]<big_number:
                        almost_most_left     = most_left
                        almost_most_left_idx = most_left_idx
                    most_left = [px, py]
                    most_left_idx = i
                if almost_most_left[_X] > px and [px,py]!=most_left:
                    almost_most_left = [px,py]
                    almost_most_left_idx = i
            if almost_most_left[_Y]<most_left[_Y]:
                most_left     = almost_most_left
                most_left_idx = almost_most_left_idx
            return most_left_idx, most_left

        _X, _Y, _C = 1, 0, 2
        points = cv2.boxPoints(rect).astype(np.int32)
        most_left_idx, most_left = topLeftPoint(points)
        point0 = points[ most_left_idx       ]
        point1 = points[(most_left_idx+1) % 4]
        point2 = points[(most_left_idx+2) % 4]
        target_size = (int(np.linalg.norm(point2-point1, ord=2)), int(np.linalg.norm(point1-point0, ord=2)), 3)
        crop  = np.full(target_size, 255, np.uint8)
        _from = np.array([ point0, point1, point2 ], dtype=np.float32)
        _to   = np.array([ [0,0], [target_size[_X]-1, 0], [target_size[_X]-1, target_size[_Y]-1] ], dtype=np.float32)
        M     = cv2.getAffineTransform(_from, _to)
        crop  = cv2.warpAffine(image, M, (target_size[_X], target_size[_Y]))
        return crop




class omz_image_classification(ov_model):
    def run(self, ocvimg):
        num_results = 5
        if 'num_results' in self.postprocess_params:
            num_results = self.postprocess_params['num_results'] 
        infres = self.inference(ocvimg)
        name = self.oblob[0]['name']
        infres = infres[name].ravel()
        idx = infres.argsort()[::-1]
        if self.labels is None:
            res = [ [idx[i], '',                  infres[idx[i]]] for i in range(num_results) ]
        else:
            res = [ [idx[i], self.labels[idx[i]], infres[idx[i]]] for i in range(num_results) ]
        return res

class omz_object_detection_ssd(ov_model):
    def run(self, ocvimg):
        infres = self.inference(ocvimg)
        name = self.oblob[0]['name']
        infres = infres[name].reshape(-1, 7)      # reshape to (x, 7)
        objs = []
        threshold = 0.7
        if 'threshold' in self.postprocess_params:
            threshold = self.postprocess_params['threshold']
        for obj in infres:
            imgid, clsid, confidence, x0, y0, x1, y1 = obj
            H, W, C = ocvimg.shape
            if confidence>threshold:              # Draw a bounding box and label when confidence>threshold
                clsid = int(clsid)
                pt0 = ( int(x0 * W), int(y0 * H) )
                pt1 = ( int(x1 * W), int(y1 * H) )
                if self.labels is None:
                    objs.append([ clsid, '',                 confidence, pt0, pt1 ])
                else:
                    objs.append([ clsid, self.labels[clsid], confidence, pt0, pt1 ])
        return objs

class omz_object_detection_yolo_v3(ov_model):
    def run(self, ocvimg):
        infres = self.inference(ocvimg)
        objs = []
        threshold = 0.7
        if 'threshold' in self.postprocess_params:
            threshold = self.postprocess_params['threshold']
        iou_threshold = 0.7
        if 'iou' in self.postprocess_params:
            iou_threshold = self.postprocess_params['iou']
        for outblob in self.oblob:
            bname = outblob['name']
            blob_res = infres[bname]
            params = self.net.layers[bname].params
            params['side'] = outblob['shape'][-1]
            objs += self.parse_yolo_region_v3(
                        blob = blob_res, 
                        resized_image_shape = self.iblob[0]['shape'][-2:], 
                        params = params, 
                        threshold = threshold )
        objs = self.bbox_NMS(objs, iou_threshold = iou_threshold)
        print(objs)
        return objs
        # params = {'anchors': '10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326', 'axis': '1', 'classes': '80', 'coords': '4', 'do_softmax': '0', 'end_axis': '3', 'mask': '6,7,8', 'num': '9', 'originalLayersNames': 'conv2d_58/Conv2D/YoloRegion'}

class omz_age_gender_estimation(ov_model):
    def run(self, ocvimg):
        infres = self.inference(ocvimg)
        age = int(infres['age_conv3'].ravel()[0] * 100)
        gender_res = infres['prob'].ravel()
        if gender_res[0]>gender_res[1]:
            return (age, 'female', gender_res[0])
        else:
            return (age, 'male', gender_res[1])

class omz_head_pose_estimation(ov_model):
    def run(self, ocvimg):
        infres = self.inference(ocvimg)
        yaw   = infres['angle_y_fc'].ravel()[0]
        pitch = infres['angle_p_fc'].ravel()[0]
        roll  = infres['angle_r_fc'].ravel()[0]
        return (yaw, pitch, roll)

class omz_emotion_estimation(ov_model):
    def run(self, ocvimg):
        infres = self.inference(ocvimg)
        infres = infres[self.oblob[0]['name']].ravel()
        idx = infres.argsort()[::-1]
        emotion = [ 'neutral', 'happy', 'sad', 'surprise', 'anger' ][idx[0]]
        return emotion

class omz_face_landmarks_regression(ov_model):
    def run(self, ocvimg):
        infres = self.inference(ocvimg)
        infres = infres[self.oblob[0]['name']].ravel()
        H, W, C = ocvimg.shape
        points = [ (int(infres[i]*W), int(infres[i+1]*H)) for i in range(0, infres.size ,2) ]
        return points

class omz_text_detection(ov_model):
    def run(self, ocvimg):
        infres = self.inference(ocvimg)
        link = infres['model/link_logits_/add']
        segm = infres['model/segm_logits/add']
        H, W, C = ocvimg.shape
        rects = self.text_detection_postprocess(link, segm, (W,H), 0.7, 0.7)
        imgs = []
        for rect in rects:
            imgs.append(self.cropRotatedImage(ocvimg, rect))  # Cut out the text region
        return rects, imgs

try:
    # C++ module for human pose estimation to extract the human pose from PAFs and heatmaps.
    # This module comes with OpenVINO human_pose_estimation_3d Python demo project.
    # You need to build the module to enable this feature.
    human_pose_available = True
    from pose_extractor import extract_poses
except ModuleNotFoundError:
    human_pose_available = False

class omz_human_pose_estimation(ov_model):
    def run(self, ocvimg):
        global human_pose_available
        if human_pose_available == False:
            return None
        infres = self.inference(ocvimg)
        PAFs = infres['Mconv7_stage2_L1'][0]
        HMs  = infres['Mconv7_stage2_L2'][0]
        people = extract_poses(HMs[:-1], PAFs, 4)                      # Construct poses from HMs and PAFs
        return people




class openvino_omz:
    model_categories = [ 'public', 'intel' ]

    def __init__(self):
        self.openvino_dir = os.environ['INTEL_OPENVINO_DIR']
        if self.openvino_dir is None:
            raise Exception('OpenVINO environment variables are not set.')
        self.omz_dir      = os.path.join(self.openvino_dir, 'deployment_tools', 'open_model_zoo')
        with open('model_def.yml') as f:
            self.model_def = yaml.safe_load(f)
        self.ie = IECore()
        with open('default_models.yml') as f:
            self.default_models = yaml.safe_load(f)
        self.ie = IECore()

    def __del__(self):
        del self.ie

    def checkModelCategory(self, omzmodel):
        """
        Search OMZ model and determine whether the model belongs to eitgher 'public' or 'intel'.  
        Args:
          omzmodel: OMZ model name
        Return:
          model category ('public', 'intel', or None)
        """
        for modelcat in openvino_omz.model_categories:
            if os.path.isfile(os.path.join(self.omz_dir, 'models', modelcat, omzmodel, 'model.yml')):
                return modelcat
        return None

    def downloadModel(self, omzmodel):
        """
        Check whether specified OMZ model is existing or not, and download it if it's not existing.  
        This function will call `model downloader` and `model converter` of OpenVINO to obtain specified OMZ IR model.  
        Args:  
          omzmodel: OMZ model name
        Return:
          None
        """
        # omzmodel : omz model name (e.g. googlenet-v1)
        model_path = os.path.join(omzmodel, 'FP16')

        # Check if IR model is existing
        exist = False
        for modelcat in openvino_omz.model_categories:
            if os.path.isfile(os.path.join(modelcat, model_path, omzmodel+'.xml')) == True:
                exist = True
 
        if exist == False:     # IR model is not existing. Let's download it.
            if os.name == 'nt':
                python = 'python'
            elif os.name == 'posix':
                python = 'python3'
            else:
                raise Exception('Unknown OS type ({})'.format(os.name))
            downloader_path = os.path.join(self.omz_dir, 'tools', 'downloader')
            # Download a OMZ model
            cmd = [ python , os.path.join(downloader_path, 'downloader.py'), '--name', omzmodel, '--precisions', 'FP16' ]
            subprocess.call(cmd)
            # Convert the model into IR if the model is a 'public' model
            category = self.checkModelCategory(omzmodel)
            if category == 'public':
                cmd = [ python , os.path.join(downloader_path, 'converter.py'), '--name', omzmodel , '--precisions', 'FP16']
                subprocess.call(cmd)
    
    def findModelDef(self, omzmodel):
        """
        Search a model definition and return the record in a dictionary.  
        Args:
          omzmodel: OMZ model name
        Return:
          `model_def` record for the specified omzmodel
        """
        for modelinfo in self.model_def:
            if modelinfo['name'] == omzmodel:
                return modelinfo
        return None

    def loadModel(self, omzmodel, download=True, **kwargs):
        """
        Download the OMZ IR model if the specified model has not been downloaded yet.  
        Create an `ov_model` object and load the OMZ IR model. The `ov_model` is created based on the description in `model_def.yml`.  
        Return the created `ov_model` object.  
        Args:
          omzmodel : OMZ model name
          download (bool): default=True
          kwargs (dict): parameters for postprocessing
        Returns:
          ov_model object
        """
        if download:
            self.downloadModel(omzmodel)
        modelinfo = self.findModelDef(omzmodel)
        objname = modelinfo['object']
        obj = globals()[objname](model=omzmodel+'.xml', kwargs=kwargs['kwargs'])
        return obj

    def getDefaultmodel(self, name):
        """
        Get default OMZ model name for a specified task from `default_models.yml`.  
        Args:  
          name: Task name to find the default model (image_classification, object_detection, face_detection)   
        Return:
          Default OMZ model name. Return 'None' if the default model is not found.  
        """
        for model in self.default_models:
            if model['task'] == name:
                return model['model']
        return None

    def modelObjectFactory(self, taskName, omzModel=None):
        """
        Create `ov_model` for a specified IR model by `omzModel`.  
        The default IR model will be searched in `default_models.yml` based on `taskName` if `omzModel` is not specified.
        Args:  
          taskName: Name of the NN-task ('image_classification', 'object_detection', ...) used for finding the default IR model from `default_model.yml` 
          omzModel: This IR model will be used to create the ov_model object if specified. Otherwise, default model will be searched based on taskName.
        Return:  
          ov_model object
        """
        if omzModel is None:
            omzModel = self.getDefaultmodel(taskName)
        modelInfo = self.findModelDef(omzModel)
        params = modelInfo['postprocess']
        obj = self.loadModel(omzModel, kwargs=params)
        if 'label' in modelInfo:
            obj.loadLabel(modelInfo['label'])
        return obj

    # ------------------------------------------------------

    def imageClassifier(self, omzmodel=None):
        return self.modelObjectFactory('image_classification', omzmodel)

    def objectDetector(self, omzmodel=None):
        return self.modelObjectFactory('object_detection', omzmodel)

    def faceDetector(self, omzmodel=None):
        return self.modelObjectFactory('face_detection', omzmodel)

    def ageGenderEstimator(self, omzmodel=None):
        return self.modelObjectFactory('age_gender', omzmodel)

    def headPoseEstimator(self, omzmodel=None):
        return self.modelObjectFactory('head_pose', omzmodel)

    def emotionEstimator(self, omzmodel=None):
        return self.modelObjectFactory('emotion', omzmodel)
    
    def faceLandmarksEstimator(self, omzmodel=None):
        return self.modelObjectFactory('face_landmarks', omzmodel)

    def textDetector(self, omzmodel=None):
        return self.modelObjectFactory('text_detect', omzmodel)

    def humanPoseEstimator(self, omzmodel=None):
        return self.modelObjectFactory('human_pose', omzmodel)



def ocv_crop(ocvimg, top_left, bottom_right, scale=1.0):
    """
    Crop OpenCV image  
    Args:  
      ocvimg : OpenCV input image
      top_left (tuple) : top-left point of cropping region
      bottom_right (tuple) : bottom-right point of cropping region
      scale (float) : scale factor for the cropping region
    Return:
      cropped OpenCV image
    """
    if top_left[0]>bottom_right[0] or top_left[1]>bottom_right[1]:
        top_left, bottom_right = bottom_right, top_left
    w, h   = (bottom_right[0]-top_left[0])*scale, (bottom_right[1]-top_left[1])*scale
    cx, cy = (bottom_right[0]+top_left[0])//2   , (bottom_right[1]+top_left[1])//2
    x1 = max(int(cx - w//2), 0)
    y1 = max(int(cy - h//2), 0)
    x2 = min(int(cx + w//2), ocvimg.shape[1]-1)
    y2 = min(int(cy + h//2), ocvimg.shape[0]-1)
    img = ocvimg[y1:y2,x1:x2].copy()
    return img

def ocv_rotate(ocvimg, angle_deg):
    """
    Rotate OpenCV image. Rotation center is the center of the input image.  
    Args:  
      ocvimg : OpenCV input image
      angle_deg (float) : Angle to rotate in degree
    Return:  
      Rotated OpenCV image
    """
    h, w = ocvimg.shape[:2]
    rotmat = cv2.getRotationMatrix2D((w//2, h//2), angle_deg, 1.0)
    img = cv2.warpAffine(ocvimg, rotmat, (ocvimg.shape[1], ocvimg.shape[0]))
    return img

def renderPeople(img, people, scaleFactor=4, threshold=0.5):
    """
    Render people's bones estimated by human_pose_estimator.  
    Args:  
      img : OpenCV image
      people : Estimated human pose [ [person0], [person1], ...]
      scaleFactor (float) : Scale factor of heatmaps and PAFs (default=4)
      threshold (float) : Thresold value to determine whether draw or not
    Return:  
      None
    """
    limbIds = [ [ 1,  2], [ 1,  5], [ 2,  3], [ 3,  4], [ 5,  6], 
                [ 6,  7], [ 1,  8], [ 8,  9], [ 9, 10], [ 1, 11],
                [11, 12], [12, 13], [ 1,  0], [ 0, 14], [14, 16], 
                [ 0, 15], [15, 17] ]

    limbColors = [
        (255,  0,  0), (255, 85,  0), (255,170,  0),
        (255,255,  0), (170,255,  0), ( 85,255,  0),
        (  0,255,  0), (  0,255, 85), (  0,255,170),
        (  0,255,255), (  0,170,255), (  0, 85,255),
        (  0,  0,255), ( 85,  0,255), (170,  0,255),
        (255,  0,255), (255,  0,170), (255,  0, 85)
    ]
    # 57x32 = resolution of HM and PAF
    scalex = img.shape[1]/(57 * scaleFactor)
    scaley = img.shape[0]/(32 * scaleFactor)
    for person in people:
        for i, limbId in enumerate(limbIds):
            x1, y1, conf1 = person[ limbId[0]*3 : limbId[0]*3+2 +1 ]
            x2, y2, conf2 = person[ limbId[1]*3 : limbId[1]*3+2 +1 ]
            if conf1>threshold and conf2>threshold:
                cv2.line(img, (int(x1*scalex),int(y1*scaley)), (int(x2*scalex),int(y2*scaley)), limbColors[i], 2)



if __name__ == "__main__":
    pass
    # test code
