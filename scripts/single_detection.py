#!/usr/bin/env python
# license removed for brevity
from __future__ import absolute_import
from __future__ import division
from torch._C import dtype
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from pyquaternion import Quaternion
import math
import torch
import time
from typing_extensions import OrderedDict
import kornia

from Conet.detector import MultiAgentDetector
from Conet.lib.opts import opts
from Conet.lib.transformation import get_2d_polygon
class ROS_MultiAgentDetector:

    def __init__(self,opt):
        rospy.init_node('ImageProcess', anonymous=True)
        self.opt = opt
        self.opt=opts().update_dataset_info_and_set_heads(opt)
        self.detector = MultiAgentDetector(self.opt)
        self.height_list = [-1., -0.5, 0., 0.5, 0.75, 1., 1.5, 2., 8.]
        self.camera_intrinsic = np.array([[486.023, 0, 359.066],
                                [0, 486.023, 240.959],
                                [0, 0, 1]])
        self.img_index = 0
        self._valid_ids = [1]
        self.vis_score_thre = 0.4
        scale_h = 500/500 
        scale_w = 500/500 
        map_scale_h = 1 / scale_h
        map_scale_w = 1 / scale_w
        self.image_size = (int(96/map_scale_h), int(192/map_scale_w))
        self.world_X_left = 200
        self.world_Y_left = 200
        self.worldgrid2worldcoord_mat = np.array([[1, 0, -self.world_X_left], [0, 1, -self.world_Y_left], [0, 0, 1]])
        self.image_sub = message_filters.Subscriber("iris/usb_cam/image_raw", Image)
        self.location_sub = message_filters.Subscriber("mavros/global_position/local", Odometry)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.location_sub], 10, 0.5, allow_headerless=False)
        self.ts.registerCallback(self.DetectionCallback)
        rospy.spin()

    def DetectionCallback(self,Image,Odometry):
        print('enter_call_back:',time.time())
        try:
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(Image, "bgr8")
            cv_image = cv2.resize(cv_image, (720, 480))
            img_tensor = torch.tensor(cv_image, dtype=torch.float32)
            images=[]
            images.append(img_tensor)
            scaled_images, meta = {}, {}
            for scale in opt.test_scales:
                cur_images = []
                cur_image, cur_meta=self.detector.pre_process(cv_image, scale)
                cur_images.append(cur_image)
                scaled_images[scale] = np.array([np.concatenate(cur_images, axis=0)])
                scaled_images[scale] = torch.from_numpy(scaled_images[scale]).to(torch.float32)
                meta[scale] = cur_meta
            shift_mats = OrderedDict()
            trans_mats = OrderedDict()
            shift_mats_np = OrderedDict()
            trans_mats_np = OrderedDict()

            orientation = Odometry.pose.pose.orientation
            position = Odometry.pose.pose.position
            roll,pitch,yaw =self.euler_from_quaternion(orientation.x,orientation.y,orientation.z,orientation.w)
            rotation_1 = Quaternion(self.euler2quaternion(- roll, - yaw, 180)) 
            rotation_2 = Quaternion(self.euler2quaternion(0, 0,   - pitch))
            self.rotation = rotation_2.rotation_matrix @ rotation_1.rotation_matrix
            self.im_position = [position.x, position.y, position.z]

            print('im_position: ',self.im_position)
            for scale in [1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64]:
                cur_shift_mat, _ = self.get_crop_shift_mat(map_scale_w=scale, map_scale_h=scale, \
                                                            world_X_left=self.world_X_left, world_Y_left=self.world_Y_left) 
                shift_mats_np[scale] = cur_shift_mat
                shift_mats[scale] = torch.from_numpy(cur_shift_mat).to(torch.float32)
            for height in self.height_list :
                cur_trans_mat = self.get_trans_mat(Odometry = Odometry, z = height)
                trans_mats_np[height] = cur_trans_mat
                trans_mats[height] = torch.from_numpy(cur_trans_mat).to(torch.float32)
            preprocessed_Data = {'images': scaled_images, 'image': images, 'meta': meta, \
                            'trans_mats': trans_mats[0.0], 'trans_mats_n010': trans_mats[-1.0], 'trans_mats_n005': trans_mats[-0.5], 'trans_mats_p005': trans_mats[0.5],\
                            'trans_mats_p007': trans_mats[0.75], 'trans_mats_p010': trans_mats[1.0], 'trans_mats_p015': trans_mats[1.5], 'trans_mats_p020': trans_mats[2.0],\
                            'trans_mats_p080': trans_mats[8.0], 
                            'shift_mats_1': shift_mats[1], 'shift_mats_2': shift_mats[2], 'shift_mats_4': shift_mats[4], 'shift_mats_8': shift_mats[8],
                            'trans_mats_withnoise': trans_mats[8.0], 'shift_mats_withnoise': shift_mats[8]}
            ret = self.detector.run(preprocessed_Data, self.img_index)
            self.img_index+=1
            # cv2.imshow('BEV', cv_image)
            # cv2.waitKey(3)
            rets = ret['results']
            detections = self.Visualization_results(rets)
            self.img_transition(trans_mats_np[0.0], shift_mats_np[1], cv_image, detections)
            # print('result: ',rets)
        except CvBridgeError as e:
            print(e)
    
    def get_trans_mat(self,Odometry,z=0):
        UAV_height = self.im_position[-1]
        im_position = [self.im_position[0],self.im_position[1], UAV_height - z]
        im_position = np.array(im_position).reshape((3, 1))
        extrinsic_mat = np.hstack((self.rotation, - self.rotation @ im_position))
        # reverse_matrix = np.eye(3)
        # reverse_matrix[0, 0] = -1
        # mat = reverse_matrix @ Quaternion([0.5, -0.5, 0.5, -0.5]).rotation_matrix.T
        project_mat = self.camera_intrinsic @ extrinsic_mat
        project_mat = np.delete(project_mat, 2, 1) @ self.worldgrid2worldcoord_mat
        return project_mat
    
    def get_crop_shift_mat(self, map_scale_w=1, map_scale_h=1, world_X_left=200, world_Y_left=200):
        im_position = [self.im_position[0], self.im_position[1], 1.]
        world_mat = np.array([[1/map_scale_w, 0, 0], [0, 1/map_scale_h, 0], [0, 0, 1]]) @ \
                np.array([[1, 0, world_X_left], [0, 1, world_Y_left], [0, 0, 1]])
        grid_center = world_mat @ im_position
        # print('grid_center: ',grid_center,'scale: ',map_scale_w)
        yaw, _, _ = Quaternion(matrix=self.rotation).yaw_pitch_roll
        yaw = -yaw + math.pi
        x_shift = 60/map_scale_w
        y_shift = 60/map_scale_h
        shift_mat = np.array([[1, 0, -x_shift], [0, 1, -y_shift], [0, 0, 1]])
        rotat_mat = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]) + \
                    np.array([[0, 0, grid_center[0]], [0, 0, grid_center[1]], [0, 0, 0]])
        trans_mat = np.linalg.inv(rotat_mat @ shift_mat)
        return trans_mat, int(Quaternion(matrix=self.rotation).yaw_pitch_roll[0]*180/math.pi)
    
    def Visualization_results(self,detection_results):
        detections = []
        for detection_result in detection_results:
            category_id = self._valid_ids[0]
            for bbox in detection_result[category_id]:
                if len(bbox) > 5:
                    bbox_out = [float("{:.2f}".format(bbox[i])) for i in range(len(bbox-1))]
                    score = bbox[-1]
                else:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = [float("{:.2f}".format(bbox[i])) for i in range(4)]
                detection = {
                    "image_id": int(self.img_index),
                    "category_id": int(category_id),
                    "bbox": bbox_out,
                    "score": float("{:.2f}".format(score))
                }
                detections.append(detection)
        return detections
    
    def euler2quaternion(self, yaw, pitch, roll):
        cy = np.cos(yaw * 0.5 * np.pi / 180.0)
        sy = np.sin(yaw * 0.5 * np.pi / 180.0)
        cp = np.cos(pitch * 0.5 * np.pi / 180.0)
        sp = np.sin(pitch * 0.5 * np.pi / 180.0)
        cr = np.cos(roll * 0.5 * np.pi / 180.0)
        sr = np.sin(roll * 0.5 * np.pi / 180.0)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return w, x, y, z
    def euler_from_quaternion(self, x, y, z, w):

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
    
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
    
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z    
    def img_transition(self,trans_mat_input,shift_mats_input,image,detections):
        # image_u = cv2.resize(image, (720, 480))
        trans_mat = trans_mat_input.copy()
        shift_mat = shift_mats_input.copy()
        # translation = np.zeros(3,1)
        image_g = self.CoordTrans(image.copy(), trans_mat.copy(), shift_mat.copy())
        image_g = self.vis_cam(image_g.copy(), detections, color=(0, 0, 255), vis_thre=self.vis_score_thre)
        cv2.imshow('BEV',image_g)
        cv2.waitKey(3)
    
    def CoordTrans(self, image, project_mat, rotat_mat, mode='L2G'):
        if mode == 'L2G':
            print(project_mat)
            trans_mat = np.linalg.inv(project_mat)
        else:
            trans_mat = project_mat
        
        feat_mat = np.array(np.diag([4, 4, 1]), dtype=np.float32)
        #tans is from RV to world grid
        #rotat_mat is from grid world  to camera world
        trans_mat = feat_mat @ rotat_mat @ trans_mat
        data = kornia.image_to_tensor(image, keepdim=False)
        data_warp = kornia.warp_perspective(data.float(),
                                            torch.tensor(trans_mat).repeat([1, 1, 1]).float(),
                                            dsize=(self.image_size[0]*4, self.image_size[1]*4))
        img_warp = kornia.tensor_to_image(data_warp.byte())
        return img_warp
    def vis_cam(self, image, annos, color=(127, 255, 0), vis_thre=-1):
    # image = np.ones_like(image) * 255
    # image = np.array(image * 0.85, dtype=np.int32)
    # alpha = np.ones([image.shape[0], image.shape[1], 1]) * 100
    # image = np.concatenate([image, alpha], axis=-1)
    # color = (255, 255, 255)
        for anno in annos:
            if (anno["score"] > vis_thre):
                bbox = anno["bbox"]
                if len(bbox) == 4:
                    bbox = [x*4 for x in bbox]
                    x, y, w, h = bbox
                    image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                else:
                    polygon = np.array(get_2d_polygon(np.array(bbox[:8]).reshape([4,2]).T)).reshape([4,2])
                    # for index, vaule in enumerate(polygon[:,1]):
                    #     polygon[index,1] = vaule -15
                    polygon = polygon * 4
                    image = cv2.polylines(image, pts=np.int32([polygon.reshape(-1, 1, 2)]), isClosed=True, color=color, thickness=2)
        return image       
if __name__ == '__main__':
    opt = opts().parse()
    ROS_MultiAgentDetector(opt)
