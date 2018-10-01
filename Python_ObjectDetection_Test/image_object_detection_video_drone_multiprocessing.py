import multiprocessing

import pygame
import pygame.locals

import cv2.cv2 as cv2  # for avoidance of pylint error
import numpy

#Drone Imports
import tellopy
import av


import time

from distutils.version import StrictVersion
import numpy as np
import os
import sys
import math
import tensorflow as tf

from collections import defaultdict

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


from utils import label_map_util
from utils import visualization_utils as vis_util


class JoystickPS3:
    # d-pad
    UP = 4  # UP
    DOWN = 6  # DOWN
    ROTATE_LEFT = 7  # LEFT
    ROTATE_RIGHT = 5  # RIGHT

    # bumper triggers
    TAKEOFF = 11  # R1
    LAND = 10  # L1
    # UNUSED = 9 #R2
    # UNUSED = 8 #L2

    # buttons
    FORWARD = 12  # TRIANGLE
    BACKWARD = 14  # CROSS
    LEFT = 15  # SQUARE
    RIGHT = 13  # CIRCLE

    # axis
    LEFT_X = 0
    LEFT_Y = 1
    RIGHT_X = 2
    RIGHT_Y = 3
    LEFT_X_REVERSE = 1.0
    LEFT_Y_REVERSE = -1.0
    RIGHT_X_REVERSE = 1.0
    RIGHT_Y_REVERSE = -1.0
    DEADZONE = 0.1


class JoystickPS4:
    # d-pad
    UP = -1  # UP
    DOWN = -1  # DOWN
    ROTATE_LEFT = -1  # LEFT
    ROTATE_RIGHT = -1  # RIGHT

    # bumper triggers
    TAKEOFF = 5  # R1
    LAND = 4  # L1
    # UNUSED = 7 #R2
    # UNUSED = 6 #L2

    # buttons
    FORWARD = 3  # TRIANGLE
    BACKWARD = 1  # CROSS
    LEFT = 0  # SQUARE
    RIGHT = 2  # CIRCLE

    # axis
    LEFT_X = 0
    LEFT_Y = 1
    RIGHT_X = 2
    RIGHT_Y = 3
    LEFT_X_REVERSE = 1.0
    LEFT_Y_REVERSE = -1.0
    RIGHT_X_REVERSE = 1.0
    RIGHT_Y_REVERSE = -1.0
    DEADZONE = 0.08


class JoystickXONE:
    # d-pad
    UP = 0  # UP
    DOWN = 1  # DOWN
    ROTATE_LEFT = 2  # LEFT
    ROTATE_RIGHT = 3  # RIGHT

    # bumper triggers
    TAKEOFF = 9  # RB
    LAND = 8  # LB
    # UNUSED = 7 #RT
    # UNUSED = 6 #LT

    # buttons
    FORWARD = 14  # Y
    BACKWARD = 11  # A
    LEFT = 13  # X
    RIGHT = 12  # B

    # axis
    LEFT_X = 0
    LEFT_Y = 1
    RIGHT_X = 2
    RIGHT_Y = 3
    LEFT_X_REVERSE = 1.0
    LEFT_Y_REVERSE = -1.0
    RIGHT_X_REVERSE = 1.0
    RIGHT_Y_REVERSE = -1.0
    DEADZONE = 0.09






# What model to download.
MODEL_NAME = 'drone_inference_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'object-detection.pbtxt')

NUM_CLASSES = 3


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:


def run_inference_for_single_image(image, graph):

  # Get handles to input and output tensors
  ops = tf.get_default_graph().get_operations()
  all_tensor_names = {output.name for op in ops for output in op.outputs}
  tensor_dict = {}
  for key in [
      'num_detections', 'detection_boxes', 'detection_scores',
      'detection_classes', 'detection_masks'
  ]:
    tensor_name = key + ':0'
    if tensor_name in all_tensor_names:
      tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
          tensor_name)
  if 'detection_masks' in tensor_dict:
    # The following processing is only for single image
    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        detection_masks, detection_boxes, image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(
        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    # Follow the convention by adding back the batch dimension
    tensor_dict['detection_masks'] = tf.expand_dims(
        detection_masks_reframed, 0)
  image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

  # Run inference
  output_dict = sess.run(tensor_dict,
                          feed_dict={image_tensor: np.expand_dims(image, 0)})

  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.uint8)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  if 'detection_masks' in output_dict:
    output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


class ObjectDetection:  
  def __init__(self, box, score, cat):
    self.box = box
    self.score = int(score * 100)
    self.cat = int(cat)
    self.tl = (int(self.box[1] * width), int(self.box[0] * height))
    self.br = (int(self.box[3] * width), int(self.box[2] * height))
    self.w = self.br[0] -  self.tl[0]
    self.h = self.br[1] -  self.tl[1]
    self.ctr = (int(self.w / 2) + self.tl[0], int(self.h / 2) + self.tl[1])
    self.pyt = int(math.sqrt(math.pow(self.w, 2) + math.pow(self.h, 2)))
  def getObj(self):
    res = {}
    res['tl'] = self.tl#Box Top Left Touple (x, y)
    res['br'] = self.br#Box Bottom Rigth Touple (x, y)
    res['w'] = self.w#Box Width Int
    res['h'] = self.h#Box Height Int
    res['cat'] = self.cat#Category Int
    res['score'] = self.score#Score Int
    res['ctr'] = self.ctr#Box Center Touple (x, y)
    res['pyt'] = self.pyt#Box Diagonal Length
    return res
  def getTxt(self):
    res = {}
    res['txt'] = "C{}-S{}-P{}".format(str(self.cat), str(self.score), str(self.pyt))
    res['pos'] = (self.tl[0], self.tl[1] - 10)
    return res
  def obsBox(self):
    factor = 2
    res = {}
    res['obs_tl'] = (int(self.tl[0] / factor), int(self.tl[1] / factor))
    res['obs_br'] = (int(self.br[0] * 1.3), int(self.br[1] * 1.3))
    return res



height = 720
width = 960






min_score = 0.7


font = cv2.FONT_HERSHEY_PLAIN
cat_colors = [(0,0,0), (0,255,0), (0,0,255), (255,0,0)]



def mergeList(lista , listb):
  c = 0
  for la in lista:
    cb = 0
    for l_a in la:
      listb[c][cb] = lista[c][cb]
      cb += 1
    c += 1
  return listb

#Create 2D List
class CatList:  
  def __init__(self, x, y, mag, pos):
    self.main_list = self.fillList(x, y)
    self.mag = mag
    self.pos = pos
    self.x = x
    self.y = y 
  def fillList(self, x, y):
    res = []
    lx = 0
    while lx < x:
      ly = 0
      temp = []
      while ly < y:
        temp.append(0)
        ly += 1
      res.append(temp)
      lx += 1
    return res
  def getList2(self):
    return self.main_list
  def fillCat(self):
    min_x = int(self.pos[0] / int(width / self.x ))
    min_y = int(self.pos[1] /  int(height / self.y))
    self.main_list[min_y][min_x] = 1
    if self.mag > 0:
      m = min_y - self.mag
      while m < (min_y + self.mag) + 1:
        n = min_x - self.mag
        while n < (min_x + self.mag) + 1:
          #print("{}-{}".format(m, n))
          if m < self.y and n < self.x:
            self.main_list[m][n] = 1
          n += 1
        m += 1
    else:
        self.main_list[min_y][min_x] = 1
    return self.main_list
#Create 2D List


def producer(ns, event):
    count = 0
    while True:
        time.sleep(3)
        ns.value = 'Count: {}'.format(count)
        count += 1
        event.set()

def joystick(ns, event):
    pygame.init()
    pygame.joystick.init()
    
    try:
        js = pygame.joystick.Joystick(0)
        js.init()
        js_name = js.get_name()
        print('Joystick name: ' + js_name)
        if js_name in ('Wireless Controller', 'Sony Computer Entertainment Wireless Controller'):
            buttons = JoystickPS4
        elif js_name in ('PLAYSTATION(R)3 Controller', 'Sony PLAYSTATION(R)3 Controller'):
            buttons = JoystickPS3
        elif js_name == 'Xbox One Wired Controller':
            buttons = JoystickXONE
    except pygame.error:
        pass

    if buttons is None:
        print('no supported joystick found')
        return
    try:
      while True:
        time.sleep(0.01)
        for e in pygame.event.get():
          #if e.type == pygame.locals.JOYAXISMOTION:
          if e.type == pygame.locals.JOYHATMOTION:
            ns.type = e.type
            ns.value = e.value
            ns.button = None
          elif e.type == pygame.locals.JOYBUTTONDOWN:
            ns.type = e.type
            ns.value = None
            ns.button = e.button
          elif e.type == pygame.locals.JOYBUTTONUP:
            ns.type = e.type
            ns.value = None
            ns.button = e.button
          #print(ns.type, " - ", ns.value, " - ", ns.button)
          event.set()
    except Exception as e:
      print(e)

def tf_main(ns, event):
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:

      frame_skip = 300

      speed = 100

      ##START DRONE CONNECTION
      drone = tellopy.Tello()
      drone.connect()
      #drone.start_video()
      drone.wait_for_connection(60.0)
      #drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
      #container = av.open(ns.drone.get_video_stream())
      container = av.open(drone.get_video_stream())

      while True:       
        for frame in container.decode(video=0):
##Control Drone With Controller
          if ns.type == 10:
            if ns.button == 4:
              drone.takeoff()
            elif ns.button == 5:
              drone.land()
            elif ns.button == 3:
              drone.forward(speed)
            elif ns.button == 1:
              drone.forward(speed)
            elif ns.button == 0:
              drone.left(speed)
            elif ns.button == 2:
              drone.right(speed)
          elif ns.type == 11:
            if ns.button == 3:
              drone.forward(0)
            elif ns.button == 1:
              drone.forward(0)
            elif ns.button == 0:
              drone.left(0)
            elif ns.button == 2:
              drone.right(0)
          elif ns.type == 9:
            if ns.value[0] < 0:
                drone.counter_clockwise(speed)
            if ns.value[0] == 0:
                drone.clockwise(0)
            if ns.value[0] > 0:
                drone.clockwise(speed)
            if ns.value[1] < 0:
                drone.down(speed)
            if ns.value[1] == 0:
                drone.up(0)
            if ns.value[1] > 0:
                drone.up(speed)
##Control Drone With Controller


          if 0 < frame_skip:
              frame_skip = frame_skip - 1
              continue
          start_time = time.time()

          image_np = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)


          height = image_np.shape[0]
          width = image_np.shape[1]
          obstacle_list = []
          stop_list = []
          wall_list = []
          counter = 0

          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          
          # Actual detection.
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})

          #DRAW BOXES
          for box in boxes[0]:
            if scores[0][counter] > min_score:
              bx = ObjectDetection(box, scores[0][counter], classes[0][counter])
              bx_obj = bx.getObj()
              if bx_obj['cat'] == 3:#WALL 2D LIST
                cat_list = CatList(16, 16, 0, bx_obj['ctr'])
                wall_list = mergeList(wall_list, cat_list.fillCat())
              elif bx_obj['cat'] == 2 and bx_obj['pyt']:#OBSTACLE 2D LIST
                #Ternary
                cat_list = CatList(16, 16, 3 if bx_obj['pyt'] > 200 else 2 if bx_obj['pyt'] > 100 else 1, bx_obj['ctr'])
                obstacle_list = mergeList(obstacle_list, cat_list.fillCat())
                #obstacle_list = cat_list.fillCat()
              elif bx_obj['cat'] == 1:#STOP 2D LIST
                cat_list = CatList(16, 16, 0, bx_obj['ctr'])
                stop_list = mergeList(stop_list, cat_list.fillCat())

              #Draw Box
              cv2.rectangle(image_np, bx_obj['tl'], bx_obj['br'], cat_colors[bx_obj['cat']], 2)
              #Draw Box Center Point
              cv2.circle(image_np, bx_obj['ctr'], 2, (89, 255, 249), -1)
              #Draw Box Text Detail
              bx_txt = bx.getTxt()
              cv2.putText(image_np, bx_txt['txt'], bx_txt['pos'], font, 1, (255,255,255))

            counter += 1



          if False:
            #DRAW LINE GUIDES
            x_lines = 16
            y_lines = 16
            lines_color = (112,255,183)

            def retLineList(value, lines):
              min_cal = int(value / lines)
              res = []
              x = 0
              while x < lines:
                res.append(min_cal * (x + 1))
                x += 1
              return res

            for x_line in retLineList(height, x_lines):
              cv2.line(image_np, (0, x_line), (width, x_line), lines_color, 1)
            for y_line in retLineList(width, y_lines):
              cv2.line(image_np, (y_line, 0), (y_line, height), lines_color, 1)
            #DRAW LINE GUIDES

          #VIDEO
          cv2.namedWindow('image', cv2.WINDOW_NORMAL)
          cv2.imshow('image', image_np)
          frame_skip = int((time.time() - start_time)/frame.time_base)
          if cv2.waitKey(1) & 0xFF == ord('q'):
            drone.quit()
            cv2.destroyAllWindows()
            break



if __name__ == '__main__':
    mgr = multiprocessing.Manager()
    namespace = mgr.Namespace()
    namespace.value = None
    namespace.type = None
    namespace.button = None
    event = multiprocessing.Event()
    p = multiprocessing.Process(
        target=joystick,
        args=(namespace, event),
    )
    c = multiprocessing.Process(
        target=tf_main,
        args=(namespace, event),
    )
    c.start()
    p.start()

    c.join()
    p.join()
    #main()