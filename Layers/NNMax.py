import torch




@staticmethod
def calculate_distance_matrix(matrix):
    """Calculates distance matrix of the input matrix"""
    norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True)
    dot_product = torch.bmm(matrix.transpose(2, 1), matrix)
    dist_matrix = norm_squared + norm_squared.transpose(2, 1) - 2 * dot_product
    return torch.sqrt(dist_matrix)

@staticmethod 
def calculate_similarity_matrix(matrix): 
    """Calculates similarity matrix of the input matrix"""
    normalized_matrix = F.normalize(matrix, p=2, dim=1) # p=2 (L2 Norm - Euclidean Distance), dim=1 (across the channels)
    dot_product = torch.bmm(normalized_matrix.transpose(2, 1), normalized_matrix)
    similarity_matrix = dot_product 
    return similarity_matrix

@staticmethod 
def prime_vmap_2d(matrix, magnitude_matrix, num_nearest_neighbors, maximum): 
    """Vectorization / Vmap Implementation for Nearest Neighbor Tensor 2D"""
    batched_process = torch.vmap(Conv1d_NN.process_batch, in_dims=(0, 0, None), out_dims=0)
    prime = batched_process(matrix, magnitude_matrix, num_nearest_neighbors, flatten=True, maximum=maximum)
    return prime 

@staticmethod 
def prime_vmap_3d(matrix, magnitude_matrix, num_nearest_neighbors, maximum): 
    """Vectorization / Vmap Implementation for Nearest Neighbor Tensor 3D"""
    batched_process = torch.vmap(Conv1d_NN.process_batch, in_dims=(0, 0, None), out_dims=0)
    prime = batched_process(matrix, magnitude_matrix, num_nearest_neighbors, flatten=False, maximum=maximum)
    return prime

@staticmethod 
def process_batch(matrix, magnitude_matrix, num_nearest_neighbors, flatten, maximum): 
    """Process the batch of matrices by finding the K nearest neighbors with reshaping."""
    ind = torch.topk(magnitude_matrix, num_nearest_neighbors, largest=maximum).indices 
    neigh = matrix[:, ind]
    if flatten: 
        reshape = torch.flatten(neigh, start_dim=1)
        return reshape
    return neigh


