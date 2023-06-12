#! /usr/bin/env python
import rospy
from std_msgs.msg import Header
from tcp_bridge.msg import ComMessage, Mat2d_33, Mat2d_conf, Mat3d
import numpy as np
import torch
class msg_process:
    def __init__(self) -> None:
        pass
    def tensor2Commsg(drone_id, round_id, tensor_2d_33 : torch.Tensor, tensor_2d_conf : torch.Tensor, tensor_3d : torch.Tensor, time_stamp):
        msg = ComMessage()
        msg.drone_id = drone_id
        msg.turns = round_id
        msg.header = Header()
        msg.header.stamp = time_stamp
        msg.mat2d_num = tensor_2d_33.shape[0]
        for i in range(9):
            tmp = Mat2d_33()
            tmp.val = [x[i // 3][i % 3] for x in tensor_2d_33.numpy().tolist() if x[i // 3][i % 3] != 0]
            tmp.index = [j for j in range(tensor_2d_33.shape[0]) if tensor_2d_33.numpy().tolist()[j][i // 3][i % 3] != 0]
            tmp.num = len(tmp.val)
            msg.mat2d_33[i] = tmp

        tmp = Mat2d_conf()
        tmp.size = list(tensor_2d_conf.shape)
        conf = [x[i] for x in tensor_2d_conf.numpy().tolist() for i in range(tensor_2d_conf.shape[1])]
        tmp.num = len(conf) - conf.count(0)
        tmp.index = [[i // tmp.size[1], i % tmp.size[1]] for i in range(len(conf)) if conf[i] != 0]
        tmp.index = [x[i] for x in tmp.index for i in range(2)]
        tmp.val = [x for x in conf if x != 0]
        msg.mat2d_conf = tmp

        tmp = Mat3d()
        mat3d = [x[i][j] for x in tensor_3d.numpy().tolist() for i in range(tensor_3d.shape[1]) for j in range(tensor_3d.shape[2])]
        shape = list(tensor_3d.shape)
        tmp.val = [x for x in mat3d if x != 0]
        tmp.num = len(tmp.val)
        tmp.index = [[(i % (shape[1] * shape[2])) // shape[1], (i % (shape[1] * shape[2])) % shape[1], i // (shape[1] * shape[2])] for i in range(len(mat3d))]
        tmp.index = [x[i] for x in tmp.index for i in range(3)]
        tmp.size = shape
        msg.mat3d = tmp
        
        return msg

    def Commsg2tensor(msg : ComMessage):
        drone_id = msg.drone_id
        turns = msg.turns

        mat33 = []
        for i in range(msg.mat2d_num):
            tmp = []
            for j in range(9):
                if msg.mat2d_33[j].index.count(i) != 0:
                    num = msg.mat2d_33[j].index.index(i)
                    tmp.append(msg.mat2d_33[j].val[num])
                else :
                    tmp.append(0)
            tmp = [tmp[:3], tmp[3:6], tmp[6:9]]
            mat33.append(tmp)
        mat33 = torch.tensor(tmp)

        size = msg.mat2d_conf.size
        matconf = [[0 for _ in range(size[1])] for _ in range(size[0])]
        for i in range(msg.mat2d_conf.num):
            index = [msg.mat2d_conf.index[2 * i], msg.mat2d_conf.index[2 * i + 1]]
            val = msg.mat2d_conf.val[i]
            matconf[index[0]][index[1]] = val
        matconf = torch.tensor(matconf)

        size = msg.mat3d.size
        mat3d = [[[0 for _ in range(size[2])] for _ in range(size[1])] for _ in range(size[0])]
        for i in range(msg.mat3d.num):
            index = [msg.mat3d.index[3 * i], msg.mat3d.index[3 * i + 1], msg.mat3d.index[3 * i + 2]]
            val = msg.mat3d.val[i]
            mat3d[index[2]][index[0]][index[1]] = val
        mat3d = torch.tensor(mat3d)

        return drone_id, turns, mat33, matconf, mat3d

    

# tensor1 = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]) #shift matrix
# tensor2 = torch.tensor([[4, 5, 6], [4, 5, 6]]) #confidence map
# tensor3 = torch.randn(3,3,3) #features_map
# print(tensor1.shape, tensor2.shape, tensor3.shape)
# print(tensor3.numpy().tolist())

# msg = tensor2Commsg(1, 1, tensor1, tensor2, tensor3)
# print(Commsg2tensor(msg))
# print(msg.mat3d)

    






