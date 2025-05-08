from CLIP import clip
import torch
import torch.nn as nn


class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        
        # 加载 CLIP 模型和其 ViT 编码器
        self.model, _ = clip.load("ViT-B/32", device='cuda')  # 或者 'cuda' 选择设备
        
        # CLIP的图像编码器输出是 512 维
        self.image_encoder = self.model.visual
        

    def forward(self, x):

        # 图像编码器生成的特征向量
        x = self.image_encoder(x)
        
        return x


