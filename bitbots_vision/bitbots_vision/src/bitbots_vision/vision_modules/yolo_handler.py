import cv2
import os
import abc
import rospy
import numpy as np
import PIL.Image as pl
from math import exp
from collections import defaultdict
from .candidate import CandidateFinder, Candidate

from .models111 import load_model
from .mydetect import detect_image

try:
    from pydarknet import Detector, Image
except ImportError:
    rospy.logerr("Not able to run Darknet YOLO! Its only executable under python3 with yolo34py or yolo34py-gpu installed.", logger_name="vision_yolo")
try:
    from openvino.inference_engine import IENetwork, IECore
except ImportError:
    rospy.logerr("Not able to run YOLO on the Intel NCS2 TPU! The OpenVINO SDK should be installed if you intend to run YOLO on the TPU", logger_name="vision_yolo")
try:
    ie = IECore()
except NameError:
    rospy.logerr("Please install/source OpenVino environment to use the NCS2 YOLO Handler.", logger_name="vision_yolo")


class YoloHandler:
    """
    Defines an abstract YoloHandler, which runs/manages the YOLO inference.

    Our YOLO is currently able to detect goalpost and ball candidates.
    """
    def __init__(self, config, model_path):
        """
        Initialization of the abstract YoloHandler.
        """
        self._candidates = None
        self._image = None

        # Load possible class names
        self.namepath = os.path.join(model_path, "obj.names")

        with open(self.namepath, "r") as fp:
            self._class_names = fp.read().splitlines()

        #print(f"{self._class_names}11")

        # Set config
        self.set_config(config)

    def set_config(self, config):
        """
        Set a new config dict, for parameter adjestments

        :param dict: dict with config values
        """
        # Set if values should be cached
        self._caching = config['caching']
        self._nms_threshold = config['yolo_nms_threshold']
        self._confidence_threshold = config['yolo_confidence_threshold']
        self._config = config

    def set_image(self, img):
        """
        Set a image for yolo. This also resets the caches.

        :param image: current vision image
        """
        # Set image
        self._image = img
        # Reset cached stuff
        self._candidates = None

    @abc.abstractmethod
    def predict(self,classname):
        """
        Implemented version should run the neural metwork on the latest image. (Cached)
        """
        raise NotImplementedError

    def get_candidates(self, class_name):
        """
        Runs neural network and returns results for all classes. (Cached)

        :param class_name: The name of the class you want to query
        """
        assert class_name in self._class_names, f"Class '{class_name}' is not available for the current yolo model!"

        self.predict(class_name)
        # this point is correct
        #print(f"classname is {class_name}")
        #print(f"self.class_name is{self._class_names}")
        #print(self.namepath)
        #print(f"the candidate{self._candidates[class_name]}")

        return self._candidates

    def get_classes(self):
        return self._class_names


class YoloHandlerDarknet(YoloHandler):
    """
    Yolo34py library implementation of our yolo model.
    """
    def __init__(self, config, model_path):
        """
        Initialization of the YoloHandlerDarknet

        :param config: vision config dict
        :param model_path: path to the yolo model
        """
        # Define more paths
        weightpath = os.path.join(model_path, "yolo_weights.weights")
        configpath = os.path.join(model_path, "config.cfg")
        datapath = os.path.join("/tmp/obj.data")
        namepath = os.path.join(model_path, "obj.names")
        # Generates a dummy file for the library
        self._generate_dummy_obj_data_file(namepath)

        self._config = config

        # Setup detector
        self.model = load_model(configpath, weightpath)
        super().__init__(config, model_path)

    def _generate_dummy_obj_data_file(self, obj_name_path):
        """
        Generates a dummy object data file.
        In which some meta information for the library is stored.

        :param obj_name_path: path to the class name file
        """
        # Generate file content
        obj_data = "classes = 2\nnames = " + obj_name_path
        # Write file
        with open('/tmp/obj.data', 'w') as f:
            f.write(obj_data)

    def predict(self, classname):
        """
        Runs the neural network
        """
        # Check if cached
        '''if self._candidates is None or not self._caching:'''
        if True:
            # Run neural network
            results = detect_image(self.model, self._image, img_size=416, conf_thres=0.5, nms_thres=0.5)
            self._candidates=[]
            '''
            print("model is runned")
            print(self._class_names)
            print(classname)
            '''
            # Go through results
            for out in results:
                # Get class id
                class_id = int(out[5])
                if self._class_names[class_id]!=classname:
                    continue
                # Get confidence
                confidence = out[4]
                if confidence > self._confidence_threshold:
                    # Get candidate position and size
                    x1 = out[0]
                    y1 = out[1]
                    x2 = out[2]
                    y2 = out[3]
                    w = x2-x1
                    h = y2-y1
                    # Create candidate
                    c = Candidate(int(x1), int(y1), int(w), int(h), confidence)
                    # Append candidate to the right list depending on the class
                    self._candidates.append(c)
                    # print(f"{self._class_names[class_id]} is {classname}")

class YoloHandlerOpenCV(YoloHandler):
    """
    Opencv library implementation of our yolo model.
    """
    def __init__(self, config, model_path):
        """
        Initialization of the YoloHandlerOpenCV

        :param config:
        :param model_path:
        """
        # Build paths
        weightpath = os.path.join(model_path, "yolo_weights.weights")
        configpath = os.path.join(model_path, "config.cfg")
        # Setup neural network
        self._net = cv2.dnn.readNet(weightpath, configpath)
        # Set default state to all cached values
        self._image = None
        super().__init__(config, model_path)

    def _get_output_layers(self):
        """
        Library stuff
        """
        layer_names = self._net.getLayerNames()

        output_layers = [layer_names[i[0] - 1] for i in self._net.getUnconnectedOutLayers()]

        return output_layers

    def predict(self,classname):
        """
        Runs the neural network
        """
        # Check if cached
        if self._candidates is None or not self._caching:
            # Set image
            blob = cv2.dnn.blobFromImage(self._image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self._net.setInput(blob)
            self._width = self._image.shape[1]
            self._height = self._image.shape[0]
            # Run net
            self._outs = self._net.forward(self._get_output_layers())
            # Create lists
            class_ids = []
            confidences = []
            boxes = []
            self._candidates = defaultdict(list)
            # Iterate over output/detections
            for out in self._outs:
                for detection in out:
                    # Get score
                    scores = detection[5:]
                    # Ger class
                    class_id = np.argmax(scores)
                    # Get confidence from score
                    confidence = scores[class_id]
                    # First threshold to decrease candidate count and inscrease performance
                    if confidence > self._confidence_threshold:
                        # Get center point of the candidate
                        center_x = int(detection[0] * self._width)
                        center_y = int(detection[1] * self._height)
                        # Get the heigh/width
                        w = int(detection[2] * self._width)
                        h = int(detection[3] * self._height)
                        # Calc the upper left point
                        x = center_x - w / 2
                        y = center_y - h / 2
                        # Append result
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            # Merge boxes
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self._confidence_threshold, self._nms_threshold)

            # Iterate over filtered boxes
            for i in indices:
                # Get id
                i = i[0]
                # Get box
                box = boxes[i]
                # Convert the box position/size to int
                box = list(map(int, box))
                # Create the candidate
                c = Candidate(*box, confidences[i])
                # Append candidate to the right list depending on the class
                class_id = class_ids[i]
                class_name = self._class_names[class_id]
                self._candidates[class_name].append(c)

class YoloHandlerNCS2(YoloHandler):
    """
    The following code is based on a code example from the Intel documentation under following licensing:

    Copyright (C) 2018-2019 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Following changes were made:
        - Different class handling
        - Modifications for our framework
        - Different NMS approach

    Used parts of the original code:
        - Parts of the comunication with the NCS stick
        - Output extraction for the Yolo network output
    """
    class _YoloParams:
        """
        Class to store params of yolo layers
        """
        def __init__(self, param, side):
            self.num = 3 if 'num' not in param else int(param['num'])
            self.coords = 4 if 'coords' not in param else int(param['coords'])
            self.classes = 2 if 'classes' not in param else int(param['classes'])
            self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                            198.0,
                            373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

            if 'mask' in param:
                mask = [int(idx) for idx in param['mask'].split(',')]
                self.num = len(mask)

                maskedAnchors = []
                for idx in mask:
                    maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
                self.anchors = maskedAnchors

            self.side = side
            self.isYoloV3 = 'mask' in param  # Weak way to determine but the only one.


    def __init__(self, config, model_path):
        # Init parent constructor
        super().__init__(config, model_path)

        # Create model file paths
        model_xml = os.path.join(model_path, "yolo.xml")
        model_bin = os.path.join(model_path, "yolo.bin")

        # Plugin initialization
        rospy.logdebug("Creating Inference Engine...", logger_name="vision_yolo")

        # Reading the IR generated by the Model Optimizer (.xml and .bin files)
        rospy.logdebug(f"Loading network files:\n\t{model_xml}\n\t{model_bin}")
        self._net = IENetwork(model=model_xml, weights=model_bin)

        assert len(self._net.inputs.keys()) == 1, "Sample supports only YOLO V3 based single input topologies"

        # Preparing network inputs
        rospy.logdebug("Preparing inputs")
        self._input_blob = next(iter(self._net.inputs))

        #  Defaulf batch_size is 1
        self._net.batch_size = 1

        # Read and pre-process input images
        self._n, self._c, self._h, self._w = self._net.inputs[self._input_blob].shape

        # Device type
        device = "MYRIAD"

        # Loading model to the plugin
        rospy.logdebug("Loading model to the plugin", logger_name="vision_yolo")
        self._exec_net = ie.load_network(network=self._net, num_requests=2, device_name=device)

    def _entry_index(self, side, coord, classes, location, entry):
        """
        Calculates the index of a yolo object.
        """
        side_power_2 = side ** 2
        n = location // side_power_2
        loc = location % side_power_2
        return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)

    def _parse_yolo_region(self, blob, resized_image_shape, original_im_shape, params, threshold):
        """
        Parses bounding boxes out of an yolo output layer.

        :param blob: Yolo layer output blob
        :param resized_image_shape: Yolo input image shape
        :param original_im_shape: Vision image shape
        :param params: Layer parameters
        :param threshold: Yolo bounding box threshold
        :return: List of bounding boxes
        """
        # Validating output parameters
        _, _, out_blob_h, out_blob_w = blob.shape
        assert out_blob_w == out_blob_h, \
            f"Invalid size of output blob. It should be in NCHW layout and height should be equal to width. Current height: '{out_blob_h}', current width = '{out_blob_w}'"

        # Extracting layer parameters
        original_image_height, original_image_width = original_im_shape
        resized_image_h, resized_image_w = resized_image_shape
        objects = list()
        predictions = blob.flatten()
        side_square = params.side ** 2

        # Parsing YOLO Region output
        for i in range(side_square):
            row = i // params.side
            col = i % params.side
            for n in range(params.num):
                obj_index = self._entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
                scale = predictions[obj_index]
                # Skip unrealistic boxes
                if scale < threshold:
                    continue
                box_index = self._entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
                # Network produces location predictions in absolute coordinates of feature maps.
                # Scale it to relative coordinates.
                x = (col + predictions[box_index + 0 * side_square]) / params.side
                y = (row + predictions[box_index + 1 * side_square]) / params.side
                # Value for exp might be a very large number, so the following construction is used here
                try:
                    w_exp = exp(predictions[box_index + 2 * side_square])
                    h_exp = exp(predictions[box_index + 3 * side_square])
                except OverflowError:
                    continue
                # Depending on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
                w = w_exp * params.anchors[2 * n] / (resized_image_w if params.isYoloV3 else params.side)
                h = h_exp * params.anchors[2 * n + 1] / (resized_image_h if params.isYoloV3 else params.side)
                # Iterate over classes
                for j in range(params.classes):
                    class_index = self._entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                            params.coords + 1 + j)
                    confidence = scale * predictions[class_index]
                    # Skip box if confidence in class is too low
                    if confidence < threshold:
                        continue
                    h = int(h * original_image_height)
                    w = int(w * original_image_width)
                    x = x * original_image_width - w / 2
                    y = y * original_image_height - h / 2
                    list_of_coordinates = [int(x), int(y), int(w), int(h)]
                    # Convert to int
                    objects.append([list_of_coordinates, float(confidence), j])
        return objects

    def predict(self,classname):
        if self._candidates is None or not self._caching:
            # Set up variables
            self._candidates = defaultdict(list)

            rospy.logdebug("Starting inference...", logger_name="vision_yolo")

            # Set request id for the stick. Since we only make one call at a time, we use a static parameter.
            request_id = 1
            # Resize image to yolo input size
            in_frame = cv2.resize(self._image, (self._w, self._h))

            # resize input_frame to network size
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((self._n, self._c, self._h, self._w))

            # Start inference
            self._exec_net.start_async(request_id=request_id, inputs={self._input_blob: in_frame})

            # Collecting object detection results
            detections = list()
            # Create barrier. This lets all following processing steps wait until the prediction is calculated.
            if self._exec_net.requests[request_id].wait(-1) == 0:
                # Get output
                output = self._exec_net.requests[request_id].output_blobs
                # Iterate over output layers
                for layer_name, out_blob in output.items():
                    buff = out_blob.buffer
                    # Reshape output layer
                    out_blob = buff.reshape(self._net.layers[self._net.layers[layer_name].parents[0]].out_data[0].shape)
                    # Create layer params object
                    layer_params = self._YoloParams(self._net.layers[layer_name].params, out_blob.shape[2])
                    # Parse yolo bounding boxes out of output blob
                    detections.extend(
                        self._parse_yolo_region(
                            out_blob,
                            in_frame.shape[2:],
                            self._image.shape[:-1],
                            layer_params,
                            self._confidence_threshold))

            if detections:
                # Transpose detections
                boxes, confidences, class_ids = list(map(list, zip(*detections)))
                # Non-maximum Suppression. This effectively chooses one bounding box if multiple are laying over each other
                box_indices = cv2.dnn.NMSBoxes(boxes, confidences, self._confidence_threshold, self._nms_threshold)
                # Iterate over filtered boxes
                for index in box_indices:
                    # Get id
                    index = index[0]
                    # Get box
                    box = boxes[index]
                    # Convert the box position/size to int
                    box = list(map(int, box))
                    # Create the candidate
                    c = Candidate(*box, confidences[index])
                    # Append candidate to the right list depending on the class
                    class_id = class_ids[index]
                    class_name = self._class_names[class_id]
                    self._candidates[class_name].append(c)


class YoloDetector(CandidateFinder):
    """
    An abstract object detector using the yolo neural network.
    This layer connects a single YOLO network with multiple candidate finders for the different classes,
    """
    def __init__(self, config, yolo):
        """
        Constructor for the YoloDetector.

        :param config: The vision config
        :param yolo: An YoloHandler implementation that runs the yolo network
        """
        self._config = config
        self._yolo = yolo

    def set_image(self, image):
        """
        Set a image for yolo. This is cached.

        :param image: current vision image
        """
        self._yolo.set_image(image)

    @abc.abstractmethod
    def get_candidates(self):
        """
        :return: all found candidates
        """
        raise NotImplementedError

    def compute(self):
        """
        Runs the yolo network
        """
        self._yolo.predict()

class YoloBallDetector(YoloDetector):
    """
    A ball detector using the yolo neural network.
    This layer connects a single YOLO network with multiple candidate finders for the different classes,
    in this case the ball class.
    """
    def __init__(self, config, yolo):
        super().__init__(config, yolo)

    def get_candidates(self):
        """
        :return: all found ball candidates
        """
        return self._yolo.get_candidates("ball")

class YoloGoalpostDetector(YoloDetector):
    """
    A goalpost detector using the yolo neural network.
    This layer connects a single YOLO network with multiple candidate finders for the different classes,
    in this case the goalpost class.
    """
    def __init__(self, config, yolo):
        super().__init__(config, yolo)

    def get_candidates(self):
        """
        :return: all found goalpost candidates
        """
        return self._yolo.get_candidates("goalpost")


class YoloRobotDetector(YoloDetector):
    """
    A robot detector using the yolo neural network.
    This layer connects a single YOLO network with multiple candidate finders for the different classes,
    in this case the robot class.
    """
    def __init__(self, config, yolo):
        super().__init__(config, yolo)

    def get_candidates(self):
        """
        :return: all found robot candidates
        """
        return self._yolo.get_candidates("robot")

class YoloXIntersectionDetector(YoloDetector):
    """
    A X-Intersection detector using the yolo neural network.
    This layer connects a single YOLO network with multiple candidate finders for the different classes,
    in this case the X-Intersection class.
    """
    def __init__(self, config, yolo):
        super().__init__(config, yolo)

    def get_candidates(self):
        """
        :return: all found X-Intersection candidates
        """
        return self._yolo.get_candidates("X-Intersection")


class YoloLIntersectionDetector(YoloDetector):
    """
    A L-Intersection detector using the yolo neural network.
    This layer connects a single YOLO network with multiple candidate finders for the different classes,
    in this case the L-Intersection class.
    """
    def __init__(self, config, yolo):
        super().__init__(config, yolo)

    def get_candidates(self):
        """
        :return: all found L-Intersection candidates
        """
        return self._yolo.get_candidates("L-Intersection")


class YoloTIntersectionDetector(YoloDetector):
    """
    A T-Intersection detector using the yolo neural network.
    This layer connects a single YOLO network with multiple candidate finders for the different classes,
    in this case the T-Intersection class.
    """
    def __init__(self, config, yolo):
        super().__init__(config, yolo)

    def get_candidates(self):
        """
        :return: all found T-Intersection candidates
        """
        return self._yolo.get_candidates("T-Intersection")
