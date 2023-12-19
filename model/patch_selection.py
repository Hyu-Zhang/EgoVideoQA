# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
from functools import partial

class CustomModule(nn.Module):
    def __init__(self, input_dim, num_heads, p, num_patches):
        super(CustomModule, self).__init__()
        self.num_heads = num_heads
        self.p = p
        self.num_patches = num_patches
        self.num_frames = 16
        self.dim = input_dim

        # Prior Weight
        self.prior_tensor = self.get_prior_tensor(self.num_patches, 'prior')
#         self.prior_tensor = nn.Parameter(self.prior_tensor)
        
        self.filter1 = torch.tensor([[0, 1, 0],
                                [1, 0, 1],
                                [0, 1, 0]])
    
    def get_prior_tensor(self, num_patches, flag):
        n = int(math.sqrt(num_patches))
        # Create an empty square matrix
        matrix = torch.zeros(n, n)
        
        # # Calculate the center position's row and column indices
        # center = (9, 7)
        # # Define the value for the center region
        # center_value = 10
        # # Define the decrement step size
        # decrement = 2
        
        # # Assign values to the matrix
        # for i in range(n):
        #     for j in range(n):
        #         # Calculate the distance from the current position to the center position
        #         distance = max(abs(i - center[0]), abs(j - center[1]))
        #         # Calculate the value for the current position
        #         value = center_value - decrement * distance
        #         # Assign the value to the corresponding position in the matrix
        #         matrix[i, j] = value
        
        if flag == 'prior':
            matrix[4:, 2:-2] = 1
        else:
            matrix[3, 2:-2] = 1
            matrix[4:,1] = matrix[4:,-2] = 1
        return matrix
    
    def vector_to_matrix(self, vector):
        n = vector.size(0)
        m = int(math.sqrt(n))
        matrix = vector.reshape(m, m)
        return matrix
    
    def forward(self, score, topn=196):
        batch_size = score.size(0)
#         B = score.softmax(dim=-1) # 64, 16, 196
        
        b = self.prior_tensor.repeat(batch_size, 1, 1).to(score.device)
        # Use a 2-channel filter
        filter1 = self.filter1.unsqueeze(0).repeat(batch_size, 1, 1).to(score.device)
        
        patch_mask = [torch.ones(batch_size, 1, device=score.device)]
        for i in range(self.num_frames):
            a = score[:, i, :]
            
            top_values, top_indices = torch.topk(a, k=1, dim=1)

            a.zero_()

            a.scatter_(1, top_indices, 1)
            
            a = a.view(batch_size, b.size(1), b.size(2))

            # Convolution operation
            neighbor_mask = torch.nn.functional.conv2d(b.unsqueeze(1).float(), filter1.unsqueeze(1).float(), padding=1)

            # Use logical_and for boolean masking
            neighbor_mask = torch.logical_and(neighbor_mask[:, 0].bool(), a.bool())
            
            # Use logical_or for modification
            b = torch.logical_or(b, neighbor_mask).float()
            
            patch_mask.append(b.view(batch_size, -1))
            
        output = torch.cat(patch_mask, dim=1)

        return output


if __name__ == "__main__":
    batch_size = 2
    num_frames = 2
    num_patches = 4
    dim = 16
    # model
    model = CustomModule(dim, 4, 0.25, num_patches)
    # video
    a = torch.randn(batch_size, num_frames, num_patches, dim)
    # question
    b = torch.randn(batch_size, dim)
    _ = model(a, b)
