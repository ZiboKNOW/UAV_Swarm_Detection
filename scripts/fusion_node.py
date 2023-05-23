from __future__ import absolute_import
from __future__ import division
from torch._C import dtype
import sys
import rospy
import cv2
from std_msgs.msg import String, Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import message_filters
from drone_detection.msg import drone_sensor
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from pyquaternion import Quaternion
import math
import torch
import time
from typing_extensions import OrderedDict
import kornia
import threading


class features_fusion():
    def __init__(self):
        rospy.init_node('Features_Fusion', anonymous = True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.decoder_sub = rospy.Subscriber("/ego/feature_data", drone_sensor, self.msg_decoder)
        self.set_pub = rospy.Publisher("ego/feature_data", drone_sensor, queue_size=10)
        self.lock = threading.Lock()
        self.round_id = 0
        self.data_dict = OrderedDict()
        self.down_ratio = rospy.get_param('/down_ratio')
        self.comm_round = rospy.get_param('/comm_round')
        self.feat_shape = rospy.get_param('/feat_shape')
        self.trans_layer = rospy.get_param('/trans_layer')
        self.agent_num = rospy.get_param('/agent_num')
        rospy.spin()

    def msg_decoder(self, data):
        layout = data.data.layout
        channels_num = int(len(layout.dim)/3)
        print('channels_num: ',channels_num)
        data_msg =list(data.data.data)
        for i in range(channels_num):
            c = layout.dim[0 + i*3].size
            h = layout.dim[1 + i*3].size
            w = layout.dim[2 + i*3].size
            data_len = int(layout.dim[0 + i*3].stride)
            # print('data_len: ',data_len)
            data_list = data_msg[:data_len]
            # print('data list: ', len(data_list))
            origin_data =  torch.tensor (np.array(data_list, dtype=np.float32).reshape(c,h,w)).unsqueeze(0)
            # print("origin data size: ",origin_data.size())
            # print('origin_data type: ',type(origin_data))
            del data_msg[:data_len]
        round_id = int(data_msg[0])
        # if round_id == self.round_id:

    def trans_message_generation(self, x, shift_mats, require_maps, round_id): 
        results_dict = {}
        b, c, h, w = x[0].shape
        results = self.decoder(x, round_id)
        results_dict.update(results)
        confidence_maps = results['hm_single_r{}'.format(round_id)].clone().sigmoid()
        confidence_maps = confidence_maps.reshape(b, 1, confidence_maps.shape[-2], confidence_maps.shape[-1])
        require_maps_list = [0,0,0,0]
        ego_request = 1 - confidence_maps.contiguous()
        if round_id > 0:
            require_maps_list[self.trans_layer[0]] = torch.cat([confidence_maps.unsqueeze(1).contiguous(),require_maps],dim=1) # (b, num_agents, 1, h, w)
            require_maps_BEV = self.communication_processor.get_colla_feats(require_maps_list, shift_mats, self.trans_layer) # (b, num_agents, c, h, w) require_maps in BEV maps

        else:
            require_maps_list[self.trans_layer[0]] = confidence_maps.unsqueeze(1).contiguous().expand(-1, self.agent_num, -1, -1, -1).contiguous() # (b, num_agents, 1, h, w)
        val_feats_to_send, _, _, _= self.communication_processor.communication_graph_learning(x[0], confidence_maps, require_maps_BEV[0:], self.agent_num , round_id=0, thre=0.03, sigma=0)
        return val_feats_to_send, ego_request, round_id

if __name__ == '__main__':
    agent_comm = features_fusion()