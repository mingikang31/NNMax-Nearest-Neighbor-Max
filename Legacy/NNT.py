'''Nearest Neighbor Tensor (NNT) Class'''

import torch 
import torch.nn as nn 

class NNT: # Nearest Neighbor  
    def __init__(self, matrix, num_nearest_neighbors):
        self.matrix = matrix.to(torch.float32) 
        
        self.num_nearest_neighbors = int(num_nearest_neighbors)
        
        self.dist_matrix = self.matrix 
        
        self.dist_matrix_vectorized = self.matrix
        
        self.prime = self.prime() 
        
    '''Getters for the NNT object'''
    @property
    def matrix(self): 
        '''Returns the matrix of the NNT object'''
        return self._matrix
    @property
    def num_nearest_neighbors(self): 
        '''Returns the number of nearest neighbors to be used in the convolution matrix'''
        return self._num_nearest_neighbors
    @property
    def dist_matrix(self): 
        '''Returns the distance matrix of the NNT object'''
        return self._dist_matrix
    @property 
    def dist_matrix_vectorized(self): 
        '''Returns the distance matrix (vectorized)of the NNT object'''
        return self._dist_matrix_vectorized
    @property
    def prime(self): 
        '''Returns the convolution matrix of the NNT object'''
        return self._prime
    
    '''Setters for the NNT object'''
    @matrix.setter
    def matrix(self, value): 
        # Check if the matrix is a torch.Tensor
        if not isinstance(value, torch.Tensor): 
            raise ValueError("Matrix must be a torch.Tensor")
        self._matrix = value
        
    @num_nearest_neighbors.setter
    def num_nearest_neighbors(self, value): 
        # Check if the number of nearest neighbors is an integer
        if not isinstance(value, int): 
            raise ValueError("Number of nearest neighbors must be an integer")
        self._num_nearest_neighbors = value
        
    @dist_matrix.setter
    def dist_matrix(self, matrix): 
        # Calculate the distance matrix using for-loops 
        self._dist_matrix = torch.zeros(matrix.shape[0], matrix.shape[2], matrix.shape[2])
        
        for i in range(matrix.shape[0]): 
            for j in range(matrix.shape[2]): 
                for k in range(matrix.shape[2]): 
                    self._dist_matrix[i, j, k] = torch.norm(matrix[i, :, j] -  matrix[i, :, k])
        
        # Calculate the distance matrix using broadcasting
        # self._dist_matrix = torch.cdist(matrix, matrix)
        
    @dist_matrix_vectorized.setter
    def dist_matrix_vectorized(self, matrix):
        # Calculate the distance matrix using vectorization 
        
        # Calculate the squared norms of each vector
        norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True)

        # Calculate the dot product of the vectors
        dot_product = torch.bmm(matrix.transpose(2, 1), matrix)

        # Calculate the distance matrix using the formula for squared Euclidean distance
        dist_matrix = norm_squared + norm_squared.transpose(2, 1) - 2 * dot_product

        # Take the square root to get the Euclidean distance
        self._dist_matrix_vectorized  = torch.sqrt(dist_matrix)
        
        

        
        
    def prime(self): 
        # stack_list = [] 
        
        # for i in range(self._matrix.shape[0]): 
            
        #     concat_list = [] 
        #     for j in range(self._matrix.shape[2]): 
        #         # Get the indices of the nearest neighbors
        #         indices = torch.topk(self.dist_matrix_vectorized[i, j, :], self.num_nearest_neighbors, largest=False).indices
                
        #         # Get the nearest neighbors
        #         nearest_neighbors = self._matrix[i, :, indices]
                
        #         # Concatenate the nearest neighbors
        #         concat_list.append(nearest_neighbors)
            
        #     # Concatenate the tensor list to create the convolution matrix 
        #     concat = torch.cat(concat_list, dim=1)
        #     stack_list.append(concat)
        # prime = torch.stack(stack_list, dim= 0)
        
        
        ### Vectorization 
        stacked_list = []
        for i in range(self.matrix.shape[0]): 
            
            dist = self.dist_matrix_vectorized[i, : , :]
            
            ind = torch.topk(dist, self.num_nearest_neighbors, largest=False).indices
            
            matrix = self.matrix[i, :, :]
            neig = matrix[:, ind]
            reshape = torch.flatten(neig, start_dim=1)
            stacked_list.append(reshape)
        prime = torch.stack(stacked_list, dim=0)
        
        return prime
        
        
        
'''EXAMPLE USAGE'''

# Example 
ex = torch.rand(32, 3, 40) # 3 samples, 2 channels, 10 tokens
                          # 3 batches, 2 sentences, 10 words
closest_neighbors = 3 # 3 closest neighbors
nnt = NNT(ex, closest_neighbors) 
print(nnt.prime.shape) # (3, 2, 10) -> (3, 2, 30) 
# print(nn.prime)

# Vectorized Distance Matrix
torch.set_printoptions(sci_mode=True)

# print("-"*50)
# print("Distance Matrix - forloop: ", nn.dist_matrix.shape) # (3, 10, 10)
# print(nn.dist_matrix)
# print("-"*50)
# print("Distance Matrix - vectorized: ", nn.dist_matrix_vectorized.shape) # (3, 2, 2)
# print(nn.dist_matrix_vectorized)
# print('-'*50)
# print(nn.dist_matrix == nn.dist_matrix_vectorized)

