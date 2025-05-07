import torch.nn as nn
import torch
from torch.nn.functional import normalize


# class Network(nn.Module):
#     def __init__(self, resnet, feature_dim, class_num):
#         super(Network, self).__init__()
#         self.resnet = resnet
#         self.feature_dim = feature_dim
#         self.cluster_num = class_num
#         self.instance_projector = nn.Sequential(
#             nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
#             nn.ReLU(),
#             nn.Linear(self.resnet.rep_dim, self.feature_dim),
#         )
#         self.cluster_projector = nn.Sequential(
#             nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
#             nn.ReLU(),
#             nn.Linear(self.resnet.rep_dim, self.cluster_num),
#             nn.Softmax(dim=1)
#         )

#     def forward(self, x_i, x_j, x_s):
#         h_i = self.resnet(x_i)
#         h_j = self.resnet(x_j)
#         h_s = self.resnet(x_s)

#         z_i = normalize(self.instance_projector(h_i), dim=1)
#         z_j = normalize(self.instance_projector(h_j), dim=1)
#         z_s = normalize(self.instance_projector(h_s), dim=1)

#         c_i = self.cluster_projector(h_i)
#         c_j = self.cluster_projector(h_j)
#         c_s = self.cluster_projector(h_s)

#         return z_i, z_j, z_s, c_i, c_j, c_s

#     def forward_cluster(self, x):
#         h = self.resnet(x)
#         c = self.cluster_projector(h)
#         c = torch.argmax(c, dim=1)
#         return c


class Network(nn.Module):
    def __init__(self, clip, feature_dim, class_num):
        super(Network, self).__init__()
        self.clip = clip
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim),

        )

        # Additional classification head for instance features
        self.instance_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

        
        self.cluster_projector = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j, x_s):
        h_i = self.clip(x_i)
        h_j = self.clip(x_j)
        h_s = self.clip(x_s)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)
        z_s = normalize(self.instance_projector(h_s), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)
        c_s = self.cluster_projector(h_s)

        return z_i, z_j, z_s, c_i, c_j, c_s

    # def forward_cluster(self, x):
    #     h = self.clip(x)
    #     c = self.cluster_projector(h)
    #     c = torch.argmax(c, dim=1)
    #     return c

    def forward_cluster(self, x, use_instance_classifier=False):
        h = self.clip(x)
        if use_instance_classifier:
            # Use instance projector + classifier for pseudo labels
            z = normalize(self.instance_projector(h), dim=1)
            return self.instance_classifier(z)
        else:
            # Use original cluster projector
            return self.cluster_projector(h)



