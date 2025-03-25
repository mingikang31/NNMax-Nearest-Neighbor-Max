import torch
import torch.nn as nn
import torch.nn.functional as F
import math



def scaled_dot_product_attention(Q, K, V, mask=None, d_k=1):
    
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, V)
    return output



def nnMax_dot_product_attention(Q, K, V, mask=None, d_k = 1):
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
    attn_probs = NNMax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, V)
    return output
      

class NNMax(nn.Module):
    def __init__(self, dim=None, K = 3):
        super(NNMax, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        if self.dim is None: 
            dim = -1 # Apply softmax over the last dimension if dim is not specified
        else: 
            dim = self.dim
            
        x = x - torch.max(x, dim=dim, keepdim=True)[0]
        
        
            
def calculate_distance_matrix(matrix):
    """Calculates distance matrix of the input matrix"""
    norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True)
    dot_product = torch.bmm(matrix.transpose(2, 1), matrix)
    dist_matrix = norm_squared + norm_squared.transpose(2, 1) - 2 * dot_product
    return torch.sqrt(dist_matrix)

def calculate_similarity_matrix(matrix): 
    """Calculates similarity matrix of the input matrix"""
    normalized_matrix = F.normalize(matrix, p=2, dim=1) # p=2 (L2 Norm - Euclidean Distance), dim=1 (across the channels)
    dot_product = torch.bmm(normalized_matrix.transpose(2, 1), normalized_matrix)
    similarity_matrix = dot_product 
    return similarity_matrix

def prime_vmap_2d(matrix, magnitude_matrix, num_nearest_neighbors, maximum): 
    """Vectorization / Vmap Implementation for Nearest Neighbor Tensor 2D"""
    batched_process = torch.vmap(process_batch, in_dims=(0, 0, None), out_dims=0)
    prime = batched_process(matrix, magnitude_matrix, num_nearest_neighbors, flatten=True, maximum=maximum)
    return prime 

def prime_vmap_3d(matrix, magnitude_matrix, num_nearest_neighbors, maximum): 
    """Vectorization / Vmap Implementation for Nearest Neighbor Tensor 3D"""
    batched_process = torch.vmap(process_batch, in_dims=(0, 0, None), out_dims=0)
    prime = batched_process(matrix, magnitude_matrix, num_nearest_neighbors, flatten=False, maximum=maximum)
    return prime

def process_batch(matrix, magnitude_matrix, num_nearest_neighbors, flatten, maximum): 
    """Process the batch of matrices by finding the K nearest neighbors with reshaping."""
    ind = torch.topk(magnitude_matrix, num_nearest_neighbors, largest=maximum).indices 
    neigh = matrix[:, ind]
    if flatten: 
        reshape = torch.flatten(neigh, start_dim=1)
        return reshape
    return neigh


def prime(matrix, similarity_matrix, num_nearest_neighbors, maximum=False):
    """Implementation of Nearest Neighbor Tensor"""
    
    stack_list = []
    for i in range(matrix.shape[0]): # Iterate through the batch
        
        concat_list = [] 
        for j in range(matrix.shape[2]):
            # Get the indices of the nearest neighbors 
            indices = torch.topk(similarity_matrix[i, j, :], num_nearest_neighbors, largest=maximum).indices
            
            # Get the nearest neighbors 
            nearest_neighbors = matrix[i, :, indices]
            
            # Concatenate the nearest neighbors
            concat_list.append(nearest_neighbors)
            
        # Concatenate the tensor list to create the convolution matrix 
        concat = torch.cat(concat_list, dim=1)
        stack_list.append(concat)
    
    prime = torch.stack(stack_list, dim = 0)
    return prime

def prime_vectorization(matrix, similarity_matrix, num_nearest_neighbors, maximum=False):
    ### Vectorization Implementation for Nearest Neighbor Tensor
    stacked_list = []
    
    for i in range(matrix.shape[0]):
        similarity = similarity_matrix[i, :, :] # each similarity matrix for each batch -> iterating through the batch dimension
        
        indices = torch.topk(similarity, num_nearest_neighbors, largest=maximum).indices
        
        matrix1 = matrix[i, :, :]
        
        neighbor = matrix[: indices]
        
        reshape = torch.flatten(neighbor, start_dim=1)
        stacked_list.append(reshape)
        
    prime = torch.stack(stacked_list, dim = 0)
    
    return prime



