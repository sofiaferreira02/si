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
    
    # Normaliza o vetor x
    norm_x = np.linalg.norm(x)
    
    # Normaliza cada linha de y
    norm_y = np.linalg.norm(y, axis=1)
    
    # Calcula o produto escalar entre x e cada linha de y
    dot_product = np.dot(y, x.T)  # Transpõe x para que a multiplicação seja correta
    
    # Calcula a similaridade cosseno
    cosine_similarities = dot_product / (norm_x * norm_y)
    
    # Calcula as distâncias cosseno
    distances = 1 - cosine_similarities
    
    return distances