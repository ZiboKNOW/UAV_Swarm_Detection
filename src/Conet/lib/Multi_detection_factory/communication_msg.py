
import torch
import kornia
import numpy as np
import torch.nn.functional as F
class communication_msg_generator:
    def __init__(self, feat_shape, drone_id, comm_thre = 0, trans_layer = 0, message_mode = 'Max', comm_round = 2):
        self.feat_H, self.feat_W = feat_shape
        self.comm_thre = comm_thre
        self.drone_id = drone_id
        self.trans_layer = trans_layer
        self.message_mode = message_mode
        self.comm_round = comm_round
    def get_colla_feats(self, x, shift_mats, trans_layer):
        val_feats = []
        for c_layer in trans_layer:
            feat_map = x[c_layer] #是confidence map 或者是 BEV下的features,都是在无人机视角下的。 是confidence map的时候  with_pos=False，# (b, num_agents, 1, h, w)            
            b, num_agents, c, h, w = feat_map.size()
            ori_shift_mats = shift_mats[c_layer]
            # noisy_shift_mats = shift_mats[-1]

            # Get the value mat (shift global feature to current agent coord) # val_feat: (b, k_agents, q_agents, c, h, w)
            ego_shift_mats = ori_shift_mats[:,self.drone_id].unsqueeze(1).expand(-1, num_agents, -1, -1).contiguous() #(b,n,3,3)
            # for agent_i in range(num_agents):
            # noisy_shift_mats_k[:,0] = shift_mats_k[:,0] # i to i 没有 noisy
            # shift_mats_k = shift_mats_k.view(num_agents*num_agents, 3, 3).contiguous()  #  (b, num_agents , 3, 3)
            ego_shift_mats = ego_shift_mats.view(b * num_agents, 3, 3).contiguous()  # (b * num_agents , 3, 3)
            shift_mats_q = torch.inverse(ori_shift_mats.contiguous()).contiguous().view(b*num_agents, 3, 3).contiguous()   # (b, num_agents , 3, 3)
            # cur_shift_mats = shift_mats_k @ shift_mats_q    # (b*k_agents*q_agents, 3, 3)
            cur_shift_mats = ego_shift_mats @ shift_mats_q    # (b * num_agents , 3, 3)

            # global_feat = feat_map.view(num_agents, c, h, w).contiguous().unsqueeze(1).expand(-1, num_agents, -1, -1, -1)
            global_feat = feat_map.contiguous().view(b * num_agents, c, h, w).contiguous()
            
            val_feat = kornia.warp_perspective(global_feat, cur_shift_mats, dsize=(h, w)) # (b*num_agents, c, h, w)
            val_feat = val_feat.view(b, num_agents, c, h, w).contiguous() # (b, num_agents, c, h, w) 
            
            # if with_qual:
            #     global_qual_map = global_qual_map.view(b, num_agents, 1, h, w).contiguous().unsqueeze(2).expand(-1, -1, num_agents, -1, -1, -1)
            #     global_qual_map = global_qual_map.contiguous().view(b*num_agents*num_agents, 1, h, w).contiguous()
                
            #     val_qual_map = kornia.warp_perspective(global_qual_map, cur_shift_mats, dsize=(h, w)) # (b*num_agents*num_agents, c, h, w)
            #     val_qual_map = val_qual_map.view(b, num_agents, num_agents, 1, h, w).contiguous() # (b, k_agents, q_agents, c, h, w)

            #     for i in range(num_agents):
            #         q_qual_map = val_qual_map[:,i:i+1,i]
            #         k_qual_maps = val_qual_map[:,:,i]
            #         rel_qual_maps = q_qual_map + k_qual_maps
            #         val_feat[:,:,i] = (k_qual_maps / (rel_qual_maps + 1e-6)) * val_feat[:,:,i]

            val_feats.append(val_feat)
        return val_feats
    def COLLA_MESSAGE(self, x, shift_mats):
        val_feats = self.get_colla_feats(x, shift_mats, self.trans_layer) #(b,n,c,h,w) n个agent的features或者confidence都到ego agent的BEV view下
        # val_feats (b，n,c,h,w)
        weight_mats = []
        for i, c_layer in enumerate(self.trans_layer):
            feat_map = x[c_layer] 
            val_feat = val_feats[i]
            b, num_agents, c, h, w = feat_map.size()
            # query_feat = feat_map.unsqueeze(0).expand(num_agents, -1, -1, -1, -1).contiguous()   # (b*num_agents, c, h, w) --> (b, k_agents, q_agents, c, h, w)
            if self.message_mode == 'Mean':
                # Mean
                feat_map_mask = torch.where(val_feat>0, torch.ones_like(val_feat).to(val_feat.device), torch.zeros_like(val_feat).to(val_feat.device))
                feat_fuse = val_feat.sum(dim=1) / (feat_map_mask.sum(dim=1)+1e-6)
            elif self.message_mode == 'Max':
                # Max
                feat_fuse = val_feat.max(dim=1)[0] # (b, c, h, w)
            # elif self.message_mode == 'Pointwise':
            #     # Pointwise: Attention Mode 1: Relu(MLP([q, k]))
            #     weight_mat = torch.cat([query_feat, val_feat], dim=3).view(b*num_agents*num_agents, c*2, h, w)
            #     weight_mat = F.relu(eval('self.weight_net'+str(c_layer))(weight_mat))    # (b*k_agents*q_agents, c, h, w)
            #     weight_mat = weight_mat.view(b, num_agents, num_agents, 1, h, w).softmax(dim=1)
            #     feat_fuse = (weight_mat * val_feat).sum(dim=1)    # (b*num_agents, c, h, w)
            #     weight_mats.append(weight_mat)
            elif self.message_mode == 'Concat':
                mean_feat = val_feat.sum(dim=1)# (b, num_agents, c, h, w)
                mean_feat = (mean_feat - feat_map)/4.0
                cat_feat = torch.cat([feat_map, mean_feat], dim=-3).reshape(num_agents, 2*c, h, w).contiguous() # (b*num_agents, 1, 2*c, h, w)
                feat_fuse = eval('self.weight_net'+str(c_layer))(cat_feat)
                feat_fuse = feat_fuse.view(num_agents, c, h, w)

            # 4. Return fused feat
            post_commu_feats = feat_fuse * 0.5 + feat_map[:,0] * 0.5

            # if self.message_mode in ['Pointwise']:
            #     x[c_layer] = post_commu_feats[:,:,:-2,:,:]
            # else:
            x[c_layer] = post_commu_feats
        return x, weight_mats, val_feats

    def communication_graph_learning(self, val_feats, confidence_maps, require_maps, num_agents , round_id=0, thre=0.03, sigma=0):
        # val_feats: (b, c, h, w) 
        # confidence_maps: (b, 1, h, w)
        # require_maps: (b, agents_num - 1, 1, h, w)
        # thre: threshold of objectiveness
        # a_ji = (1 - q_i)*q_ji
        confidence_maps = confidence_maps.unsqueeze(1).contiguous().expand(-1, num_agents - 1, -1, -1, -1).contiguous()
        b, _, _ , h, w  = confidence_maps.shape
        if round_id == 0:
            communication_maps = confidence_maps
        else: # more then 1 round the is prev confidence_maps
            # beta = 0
            communication_maps = confidence_maps * require_maps
            communication_maps = F.relu(communication_maps)

        ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
        zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
        communication_mask = torch.where((communication_maps - thre)>1e-6, ones_mask, zeros_mask) #(b, agents_num - 1, 1, h, w)
        # communication_mask, communication_maps_topk = _bandwidthAware_comm_mask(communication_maps, B=1)

        if round_id > 0:
            # Local context
            communication_mask = self.get_local_mask(communication_mask, kernel_size=11, stride=1) #max pooling (b, agents_num - 1 , 1, h, w)
            communication_maps = communication_maps * communication_mask
            communication_thres = [0.0, 0.001, 0.01, 0.03, 0.06, 0.08, 0.1, 0.13, 0.16, 0.20, 0.24, 0.28, 1.0]
            for thre_idx, comm_thre in enumerate(communication_thres):
                if (comm_thre == thre) and (thre_idx>=2):
                    if round_id == 1:
                        thre = communication_thres[thre_idx-1]
                    else:
                        thre = communication_thres[thre_idx-2]
                    break
            # thre = min(0.001, thre)
            # thre = 0.08
            communication_mask = torch.where((communication_maps - thre)>1e-6, ones_mask, zeros_mask)
            # if round_id == 2:
            #     communication_mask = F.relu(communication_mask - prev_communication_mask)
            # communication_mask = _get_topk(communication_maps, k=400)
        # Range
        # communication_mask_0 = torch.where((communication_maps - 1e-5)>1e-6, ones_mask, zeros_mask)
        # communication_mask_1 = torch.where((0.001 - communication_maps)>1e-6, ones_mask, zeros_mask)
        # communication_mask = communication_mask_0 * communication_mask_1

        # if round_id == 0:
            # if sigma > 0:
            #     confidence_maps = confidence_maps.view(b*q_agents, 1, h, w)
            #     confidence_maps = self.__getattr__('gaussian_filter'+str(int(10*sigma)))(confidence_maps)
            #     confidence_maps = confidence_maps.view(b, q_agents, 1, h, w)
            # confidence_mask = torch.where((confidence_maps - thre)>1e-6, ones_mask[:,:,0], zeros_mask[:,:,0])
            # Local context
            # confidence_mask = _get_local_mask(confidence_mask, kernel_size=11, stride=1)
            # confidence_maps = confidence_maps * confidence_mask
            # confidence_mask = torch.where((confidence_maps - 0.001)>1e-6, ones_mask[:,:,0], zeros_mask[:,:,0])
            # communication_rate = confidence_mask.sum()/(b*h*w*q_agents)
        # else:
        # communication_mask_list = []
        # for i in range(num_agents):
        #     communication_mask_list.append(torch.cat([communication_mask[:,:i,i],communication_mask[:,i+1:,i]], dim=1).unsqueeze(2)) #即自己和自己的通信量不计算在其中
        # communication_mask_list = torch.cat(communication_mask_list, dim=2) # (1, 4, 5, 1, h, w)
        communication_rate = communication_mask.sum()/(b*h*w*(num_agents - 1))

        communication_mask_nodiag = communication_mask.clone()
        # communication_maps_topk_nodiag = communication_maps_topk.clone()
        # for q in range(num_agents - 1):
        #     communication_mask_nodiag[:,q,q] = ones_mask[:,q,q] #self comminication mask = 1
            # communication_maps_topk_nodiag[:,q,q] = ones_mask[:,q,q]
            # val_feats_aftercom[:,:q,q] = val_feats[:,:q,q] * communication_mask[:,:q,q]
            # val_feats_aftercom[:,q+1:,q] = val_feats[:,q+1:,q] * communication_mask[:,q+1:,q]
        val_feats_to_send = val_feats.unsqueeze(1).contiguous().expand(-1, num_agents - 1, -1, -1, -1).contiguous() * communication_mask_nodiag #(b, agents_num - 1, 1, h, w)
        return val_feats_to_send, communication_mask, communication_rate

        
        

    def get_local_mask(self, x, kernel_size=3, stride=1): 

        def get_padded_feat(x, kernel_size=3, stride=1): 
            b, c, h, w = x.shape
            padding_w = torch.zeros([b, c, h, (kernel_size-1)*stride//2], dtype=torch.float32, device=x.device) #for same size
            padding_h = torch.zeros([b, c, (kernel_size-1)*stride//2, w+(kernel_size-1)*stride], dtype=torch.float32, device=x.device)
            x = torch.cat([padding_w, x, padding_w], dim=-1)    # (b, num_agents, c, h, w+kernel_size-1)
            x = torch.cat([padding_h, x, padding_h], dim=-2)    # (b, num_agents, c, h+kernel_size-1, w+kernel_size-1)
            return x

        # mask: (b, k_agents, q_agents, 1, h, w)
        shape_len = len(x.shape)
        if shape_len == 6:
            b, k_agents, q_agents, c, h, w = x.shape
            x = x.flatten(0, 2)
        else:
            b, q_agents, c, h, w = x.shape
            x = x.flatten(0, 1)
        x = get_padded_feat(x, kernel_size, stride) #padding，横竖边加上zeros， （b*n*n,1,h+kernel_size-1, w+kernel_size-1）
        x_local = []
        for i in range(kernel_size):
            for j in range(kernel_size):
                x_local.append(x[:,:,i*stride:h+i*stride,j*stride:w+j*stride].unsqueeze(1))
        
        x_local = torch.cat(x_local, dim=1) # (b, kernel**2, 1, h, w)
        x_local = x_local.max(dim=1)[0] # (b, 1, h, w) #
        if shape_len == 6:
            x_local = x_local.view(b, k_agents, q_agents, 1, h, w)
        else:
            x_local = x_local.view(b, q_agents, 1, h, w)
        return x_local
