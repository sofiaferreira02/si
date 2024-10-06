import numpy as np

def cosine_distance(x, y):
    """
    Computes the cosine distance between a single sample x and multiple samples y.
    
    Arguments:
    x -- a 1D numpy array representing a single sample
    y -- a 2D numpy array where each row represents a sample
    
    Returns:
    distances -- a 1D numpy array containing the cosine distances between x and each sample in y
    """
    
    # Normalize the vector x
    norm_x = np.linalg.norm(x)
    
    # Normalize each row in y
    norm_y = np.linalg.norm(y, axis=1)
    
    # Compute the dot product between x and each row in y
    dot_product = np.dot(y, x)
    
    # Compute the cosine similarities
    cosine_similarities = dot_product / (norm_x * norm_y)
    
    # Compute the cosine distances
    distances = 1 - cosine_similarities
    
    return distances