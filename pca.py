"""
Principal component analysis (PCA).
Method one: compute eigen vectors directly.
Method two: use singular value decomposition (SVD) 
References:
1. 
    1.1 PCA: https://youtu.be/fkf4IBRSeEc
    1.2 SVD: https://youtu.be/nbBvuuNVfco
2. https://www.coursera.org/lecture/machine-learning/principal-component-analysis-algorithm-ZYIPa
3. https://www.youtube.com/watch?v=52d7ha-GdV8
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
# %%
class PCA:
    def __init__(self, num_components) -> None:
        self.num_components = num_components

    def transform(self, X, method='eigen'):
        """
        Assume that each row is an example
        """
        self.X_mean = np.mean(X, axis=0)
        ## Mean normalization, i.e., subtract mean of each feature.
        X -= self.X_mean

        ## Two methods to compute covariance matrix, method 1: use numpy
        ## np.cov assumes that each column is an example, we transform X below.
        X_cov = np.cov(X.T)
        ## method 2:
        #X_cov = 1/X.shape[0] * X.T @ X

        if method == 'eigen':
            ## column eigen_val[:,i] is the eigenvector corresponding \
            ## to the eigenvalue eigen_vectors[i].
            eigen_val, eigen_vectors = np.linalg.eig(X_cov)
            ## each eigen vector is a column vector
            ## Sort eigen_vectors by eigen_vals, max first
            indices = np.argsort(eigen_val)[::-1] ## Order from largest to smallest
            ## Keep the first num_components largest eigen values
            indices = indices[:self.num_components]
            ## Save eigen_val for debugging purpose. We don't need it for PCA
            self.eigen_val = eigen_val[indices]
            ## store only the eigen vectors corresponding to the num_components\
            ##  largest eigen values
            self.eigen_vectors = eigen_vectors[:, indices]
            #self.principal_component = X @ self.eigen_val
            #return X @ self.eigen_vectors

        elif method == 'svd':
            ## Eacu column of U is an eigen vector.
            ## Each element of S is an eigen value, sorted in descending order
            U, S, VT = np.linalg.svd(X_cov.T,full_matrices=True)
            ## Keep the first num_components largest eigen values
            ## Save eigen_val for debugging purpose. We don't need it for PCA
            self.eigen_val = S[0:self.num_components]
            ## Save the first num_components eigen vectors. 
            self.eigen_vectors = U[:, 0:self.num_components]
            ## Project X onto the coordinate of the eigen vectors
            #self.principal_component = U @ S
        
        return X @ self.eigen_vectors
    
    def reconstruct(self, Z):
        """
        Reconstruct the compressed data back to the original dimension
        Z is the compressed data. Each row of Z is an example.
        """
        return Z @ self.eigen_vectors.T

 #### test 1.
 #%%
if __name__ == '__main__':
    ## Generate a 2D cloud point.
    m = 10_000
    center = np.array([2, 1]).reshape(1,2) ## Center of data
    sig = [2, 0.5] #np.array([2, 0.5])#.reshape(1,2) ## principal axis
    theta = np.pi/3 ## Rotate cloud by pi/3
    R = np.array([[np.cos(theta), -np.sin(theta)],\
        [np.sin(theta), np.cos(theta)]]).reshape(2,2)
    X = R @ ((np.random.randn(m, 2)) @ np.diag(sig)).T + center.T
    X = X.T
    plt.rcParams['figure.figsize'] = [8, 8]
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(X[:,0], X[:,1], color='b')
    ax1.set_xlim((-6, 8))
    ax1.set_ylim((-6,8))
    
    ## Add a dimension to X. Transform the 2D cloud point to 3D
    z = (X[:,0] + X[:,1]).reshape(-1, 1)
    X = np.hstack((X, z))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(X[:,0], X[:,1], X[:,2], color='r')


    pca = PCA(2)
    #X_trans = pca.transform(X)
    X_trans = pca.transform(X, method='svd')
    

    #project the 3D data cloud to 2D
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.scatter(X_trans[:,0], X_trans[:,1], color='g')
    ax3.set_xlim((-6, 8))
    ax3.set_ylim((-6,8))

    ## Reconstruct the compressed (2D) data back to 3D.
    ## Plot the reconstructed data and the original data.
    Xp = pca.reconstruct(X_trans)
    ax2.scatter(X[:,0], X[:,1], X[:,2], color='c')
    ax2.set_xlim((-6, 8))
    ax2.set_ylim((-6,8))

    plt.show()

    

# %%
