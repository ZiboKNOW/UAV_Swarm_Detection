from __future__ import absolute_import
from __future__ import division
from torch._C import dtype
import sys
import rospy
import cv2
from std_msgs.msg import String, Float32MultiArray, MultiArrayLayout, MultiArrayDimension, Header
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import message_filters
from drone_detection.msg import drone_sensor, reset
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from pyquaternion import Quaternion
import math
import torch
import time
from typing_extensions import OrderedDict
import kornia
import threading
from Conet.lib.Multi_detection_factory.communication_msg import communication_msg_generator
from Conet.lib.Multi_detection_factory.dla34 import decoder
from Conet.lib.tcp_bridge.tensor2Commsg import msg_process
from tcp_bridge.msg import ComMessage, Mat2d_33, Mat2d_conf, Mat3d
# temp = sys.stdout
# f = open('fusion_node.log', 'w')
# sys.stdout = f
class features_fusion():
    def __init__(self):
        rospy.init_node('Features_Fusion', anonymous = True)
        self.name_space = rospy.get_namespace().strip('/')
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.decoder_sub = rospy.Subscriber("/{}/ego/feature_data".format(self.name_space), drone_sensor, self.ego_msg_decoder)
        self.tcp_trans = msg_process
        ###################### TO DO #####################
        #              the sub of the TCP                #
        ##################################################
        ###################### TO DO #####################
        #              the pub of the TCP                #
        ##################################################
        self.reset_pub = rospy.Publisher("/reset_topic", reset, queue_size= 1)
        self.lock = threading.Lock()
        # self.features_map = OrderedDict()
        # self.shift_map = OrderedDict()
        self.round_id = 0
        self.fusion_buffer = OrderedDict()
        self.lock = threading.Lock()
        self.buffer_lock = False

        self.down_ratio = rospy.get_param('/{}/down_ratio'.format(self.name_space))
        self.comm_round = rospy.get_param('/{}/comm_round'.format(self.name_space))
        self.feat_shape = rospy.get_param('/{}/feat_shape'.format(self.name_space))
        self.trans_layer = rospy.get_param('/{}/trans_layer'.format(self.name_space))
        self.agent_num = rospy.get_param('/{}/agent_num'.format(self.name_space))
        self.drone_id = rospy.get_param('/{}/Drone_id'.format(self.name_space))
        self.time_gap_threshold = rospy.get_param('/{}/time_gap_threshold'.format(self.name_space))
        self.channels = rospy.get_param('/{}/channels'.format(self.name_space))
        self.communication_module = communication_msg_generator(self.feat_shape,self.drone_id)
        self.heads = rospy.get_param('/{}/heads'.format(self.name_space))
        self.in_fusion = False
        self.next_id = 0
        if self.drone_id == 0:
            self.next_id = 1
        self.tcp_pub = rospy.Publisher("/drone_{}_to_drone_{}_sending".format(self.drone_id,self.next_id), ComMessage, queue_size=1)
        self.feature_sub = rospy.Subscriber('/drone_{}_recive'.format(self.drone_id), ComMessage, self.drones_msg_decoder)
        self.decoder = decoder(self.heads, self.channels, self.down_ratio).to(self.device)
        rospy.Timer(rospy.Duration(0.1), self.Check_Buffer)
        self.round_start_time = time.time()
        rospy.spin()
    def send_reset(self, send_new_msg):
        msg = reset()
        msg.drone_id.data = self.drone_id
        msg.reset = send_new_msg
        msg_Header = Header()
        msg_Header.stamp = rospy.get_rostime()
        msg.header = msg_Header
        if send_new_msg:
            self.round_id = 0
            self.lock.acquire()
            self.fusion_buffer.clear()
            self.lock.release()
        self.reset_pub.publish(msg)
    def Check_Buffer(self,event):
        self.lock.acquire()
        if len(self.fusion_buffer) == self.agent_num:
            for k in self.fusion_buffer.keys():
                if 'round_{}'.format(self.round_id) not in k:
                    print('Error: Wrong msg in buffer.')
                    self.send_reset(True)
                    self.lock.release()
                    return
            self.lock.release()
            self.feature_fusion()  

        else:
            self.lock.release()
            return

    def drones_msg_decoder(self, ComMessage):
        drone_id, agent_round_id, mat33, matconf, mat3d = self.tcp_trans.Commsg2tensor(ComMessage)
        print('decoder size ','round_id: ',agent_round_id,' mat33: ',mat33.shape,' matconf: ',matconf.shape,' mat3d: ',mat3d.shape)
        update_dict = False
        if self.round_id == 0:
            if agent_round_id == self.round_id and not self.in_fusion:
                update_dict = True
            elif agent_round_id > self.round_id and self.in_fusion:
                update_dict = True
            elif agent_round_id == self.round_id and self.in_fusion:
                print('In fusion: stop pub image')
                self.send_reset(False)
                return
            else:
                print('error: reset the iamge')
                self.send_reset(True)
                return
        else:
            if self.round_id <= agent_round_id:
                update_dict = True
            elif self.in_fusion :
                print('In fusion: stop pub image')
                self.send_reset(False)
            else:
                print('Error: late msg')
                self.send_reset(True)             
        if update_dict:
            drone_msg_data= OrderedDict()
            drone_msg_data['features_map'] = [mat3d] #[b, c, h, w]
            drone_msg_data['shift_mat'] = [mat33]
            drone_msg_data['require_mat'] = [matconf]
            drone_msg_data['round_id'] = agent_round_id
            drone_msg_data['header'] = ComMessage.header
            self.lock.acquire()
            self.fusion_buffer.update({'drone_{}_round_{}_msg'.format(drone_id,agent_round_id):drone_msg_data})
            self.lock.release()
            # print('decoded shift_mat: ',drone_msg_data['shift_mat'])
            # print('ego shape ','features_map: ',drone_msg_data['features_map'][0].shape,' shift_mat: ',drone_msg_data['shift_mat'][0].shape)
        else:
            return

    def feature_fusion(self):
        print('start fusion')
        fusion_start_time = time.time()
        if self.round_id == 0:
            self.round_start_time = time.time()
        self.in_fusion = True
        time_stamp_list = []
        features_map_list = []
        self.lock.acquire()
        for k in self.fusion_buffer.keys():
            time_stamp_list.append(self.fusion_buffer[k]['header'].stamp.secs)
        self.lock.release()
        if self.round_id == 0 and max(time_stamp_list) - min(time_stamp_list) > self.time_gap_threshold:
            print('Error: too large stamp gap(secs) ',max(time_stamp_list) - min(time_stamp_list))
            self.send_reset(True)
            self.in_fusion = False
            return
        
        self.lock.acquire()
        features_map_list = self.fusion_buffer['drone_{}_round_{}_msg'.format(self.drone_id,self.round_id)]['features_map'] 
        shift_mat_list = self.fusion_buffer['drone_{}_round_{}_msg'.format(self.drone_id,self.round_id)]['shift_mat']
        self.lock.release()
        for layer in self.trans_layer:
            fustion_features = features_map_list[layer]
            b ,c, h, w = fustion_features.shape
            require_maps = torch.zeros(b, self.agent_num, h, w).to(self.device)
            fustion_features = fustion_features.unsqueeze(1).expand(-1, self.agent_num, -1, -1, -1).contiguous()
            shift_mat = shift_mat_list[layer] #[3,3]
            shift_mat = shift_mat.unsqueeze(0).contiguous()
            shift_mat = shift_mat.unsqueeze(1).expand(-1, self.agent_num, -1, -1).contiguous() #[1,n,3,3]
            for agent in range(self.agent_num):
                if agent != self.drone_id:
                    b, n ,c, h, w = fustion_features.shape
                    self.lock.acquire()
                    fustion_features[0, agent] = self.fusion_buffer['drone_{}_round_{}_msg'.format(agent, self.round_id)]['features_map'][layer]
                    shift_mat[0,agent] = self.fusion_buffer['drone_{}_round_{}_msg'.format(agent, self.round_id)]['shift_mat'][layer]
                    require_maps[0,agent] = self.fusion_buffer['drone_{}_round_{}_msg'.format(agent, self.round_id)]['require_mat'][layer].to(self.device)
                    self.lock.release()
            features_map_list[layer] = fustion_features
            shift_mat_list[layer] = shift_mat
        features_map_list = [x.to(self.device) for x in features_map_list]
        shift_mat_list = [x.to(self.device) for x in shift_mat_list]
        fused_feature_list, _, _ = self.communication_module.COLLA_MESSAGE(features_map_list, shift_mat_list) 
        # self.lock.acquire()
        # temp_shift_mat = self.fusion_buffer['drone_{}_round_{}_msg'.format(self.drone_id,self.round_id)]['shift_mat']
        # self.lock.release()
        if self.round_id >= self.comm_round - 1:
            results = self.decoder(fused_feature_list, self.round_id)
            print("End fusion, require for new image")
            round_end_time = time.time()
            print('round time: ',round_end_time - self.round_start_time,'s')
            self.send_reset(True)
            self.in_fusion = False
            return
        
        val_feats_to_send, ego_request = self.trans_message_generation(fused_feature_list, shift_mat_list, require_maps, self.round_id)
        # clear the buffer
        self.lock.acquire()
        for k in list(self.fusion_buffer.keys()):
            if 'round_'.format(self.round_id) in k:
                del self.fusion_buffer[k]
        self.round_id +=1
        self.lock.release()

        ego_msg = self.pack_ego_msg(fused_feature_list, shift_mat_list)
        self.fusion_buffer.update({'drone_{}_round_{}_msg'.format(self.drone_id,self.round_id):ego_msg})

        # require_maps [1,n,h,w] 其中self.require 为 zeros
        self.in_fusion = False
        fusion_end_time = time.time()
        print('fusion_time: ',fusion_end_time - fusion_start_time,'s')
        print('end fusion')
        self.send_tcp_msg(val_feats_to_send.to('cpu').detach().contiguous(), ego_request.to('cpu').detach().contiguous(), self.next_id)

        return
                
                


    def pack_ego_msg(self, fused_feature,shift_mat):
        # shift_mat
        drone_msg_data= OrderedDict()
        drone_msg_data['features_map'] = fused_feature #[b, c, h, w]
        drone_msg_data['shift_mat'] = shift_mat
        # print('shift_size: ',shift_mat[0].shape ,'len: ',len(shift_mat))
        for i in self.trans_layer:
            drone_msg_data['shift_mat'][i] = shift_mat[i][0,self.drone_id]
            # print('drone_msg_data shift_mat: ', drone_msg_data['shift_mat'][i].shape)
        drone_msg_data['round_id'] = self.round_id
        msg_header = Header()
        msg_header.stamp = rospy.Time.now()
        drone_msg_data['header'] = msg_header
        return drone_msg_data

    def ego_msg_decoder(self, data):
        update_dict = False
        if self.round_id == 0:
            if data.round_id == self.round_id and not self.in_fusion:
                update_dict = True
            elif data.round_id == self.round_id and self.in_fusion:
                print('In fusion stop revice image')
                self.send_reset(False)
            else:
                self.send_reset(False)
        else:
            self.send_reset(True)
            return
        
        layout = data.data.layout
        channels_num = int(len(layout.dim)/3)
        # print('channels_num: ',channels_num)
        data_msg = list(data.data.data)
        feature_list = []
        for i in range(channels_num):
            c = layout.dim[0 + i*3].size
            h = layout.dim[1 + i*3].size
            w = layout.dim[2 + i*3].size
            data_len = int(layout.dim[0 + i*3].stride)
            # print('data_len: ',data_len)
            data_list = data_msg[:data_len]
            # print('data list: ', len(data_list))
            origin_data =  torch.tensor(np.array(data_list, dtype=np.float32).reshape(c,h,w)).unsqueeze(0)
            feature_list.append(origin_data)
            # print("origin data size: ",origin_data.size())
            # print('origin_data type: ',type(origin_data))
            del data_msg[:data_len]
        layout_shift = data.shift_matrix.layout
        shift_msg = list(data.shift_matrix.data)
        h_shift = layout_shift.dim[0].size   
        w_shift = layout_shift.dim[1].size
        origin_shift_matrix = torch.tensor(np.array(shift_msg, dtype=np.float32).reshape(h_shift,w_shift))
        # for v in feature_list:
        #     print('feature_size: ',v.size())
        if update_dict:
            drone_msg_data= OrderedDict()
            drone_msg_data['features_map'] = feature_list #[b, c, h, w]
            drone_msg_data['shift_mat'] = [origin_shift_matrix]
            drone_msg_data['round_id'] = data.round_id
            drone_msg_data['header'] = data.header
            self.lock.acquire()
            self.fusion_buffer.update({'drone_{}_round_{}_msg'.format(data.drone_id,data.round_id):drone_msg_data})
            self.lock.release()
            # print('decoded shift_mat: ',drone_msg_data['shift_mat'])
            print('ego shape ','features_map: ',drone_msg_data['features_map'][0].shape,' shift_mat: ',drone_msg_data['shift_mat'][0].shape)
        else:
            return
        # if round_id == self.round_id:

    def trans_message_generation(self, fused_features, shift_mats, agent_require_maps, round_id): 
        results_dict = {}
        b, c, h, w = fused_features[0].shape
        results = self.decoder(fused_features, round_id)
        results_dict.update(results)
        confidence_maps = results['hm_single_r{}'.format(round_id)].clone().sigmoid()
        confidence_maps = confidence_maps.reshape(b, 1, confidence_maps.shape[-2], confidence_maps.shape[-1])
        # print('confidence_maps size: ',confidence_maps.shape) #[1, 1, 96, 192]
        require_maps_list = [0,0,0,0]
        ego_request = 1 - confidence_maps.contiguous()  #[1, 1, 96, 192]
        require_maps = agent_require_maps
        require_maps = require_maps.unsqueeze(2).contiguous() #[1,2,1,96, 192]
        require_maps[0,self.drone_id,0] = ego_request[0,0]
        require_maps_list[self.trans_layer[0]] = require_maps # (b, num_agents, h, w)
        require_maps_BEV = self.communication_module.get_colla_feats(require_maps_list, shift_mats, self.trans_layer) # (b, num_agents, c, h, w) require_maps in BEV maps
        # else:
        #     require_maps_list[self.trans_layer[0]] = confidence_maps.unsqueeze(1).contiguous().expand(-1, self.agent_num, -1, -1, -1).contiguous() # (b, num_agents, 1, h, w)
        # print('require_maps_BEV size: ',require_maps_BEV[0].shape, 'len: ',len(require_maps_BEV)) # [1, 2, 1, 96, 192]
        val_feats_to_send, _, _= self.communication_module.communication_graph_learning(fused_features[0], confidence_maps, require_maps_BEV[0], self.agent_num , round_id , thre=0.03, sigma=0)
        # print('val_feats_to_send: ', val_feats_to_send.shape, 'ego_request: ',ego_request.shape)
        return val_feats_to_send, ego_request

    def send_tcp_msg(self, val_feats_to_send, ego_request,next_id):
        self.lock.acquire()
        ego_msg = self.fusion_buffer['drone_{}_round_{}_msg'.format(self.drone_id,self.round_id)]
        self.lock.release()
        tcp_msg = self.tcp_trans.tensor2Commsg(self.drone_id, self.round_id, ego_msg['shift_mat'][0].to('cpu').detach().unsqueeze(0).contiguous(), ego_request[0,0], val_feats_to_send[0,next_id], rospy.get_rostime())
        
        # self.tcp_trans.tensor2Commsg(self.drone_id, self.round_id, shift_mat_tcp, require_maps, features_map_tcp, time_stamp)
        self.tcp_pub.publish(tcp_msg)
    ###################### TO DO #####################
    #       callback function of the TCP-decoder     #
    ##################################################
if __name__ == '__main__':
    agent_comm = features_fusion()