import numpy as np
from matplotlib import pylab as plt
from torch.linalg import svd
from PIL import Image
import math
import torch
from skimage import io

# from tensorflow.python.ops.gen_array_ops import diag, transpose

torch.random.seed()
device = torch.device("cuda")
def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

class TRPCA:

    def converged(self, L, E, X, L_new, E_new):
        '''
        judge convered or not
        '''
        condition1 = torch.max(abs(L_new - L))
        condition2 = torch.max(abs(E_new - E))
        condition3 = torch.max(abs(L_new + E_new - X))



        # obj1, obj2 = 0, 0
        # for j in range(X.shape[2]):
        #     obj1 +=torch.norm(E[:,:,j],p=1)
        #     obj2 +=torch.norm(dY[:,:,j])
        # print(max(condition1,condition2,condition3))
        return max(condition1,condition2,condition3)

    def SoftShrink(self, X, tau):
        '''
        apply soft thesholding
        '''
        z = torch.sign(X) * (abs(X) - tau) * ((abs(X) - tau) > 0)
        # z = np.clip(X-tau,a_min=0,a_max=None)+np.clip(X+tau,a_min=None,a_max=0)
        return z

    def SVDShrink2(self, Y, tau):
        '''
        apply tensor-SVD and soft thresholding
        '''
        [n1, n2, n3] = Y.shape
        XX = torch.complex(torch.empty(size= Y.shape),torch.empty(size= Y.shape)).to(device)
        X = torch.fft.fft(Y)
        # Z =torch.matmul(X[:,:,0],X[:,:,0])
        tnn = 0
        trank = 0

        U, S, V = svd(X[:,:,0])
        r = torch.count_nonzero(S>tau)
        S = S.type(torch.complex64)
        if r>=1:
            S = torch.diag(S[0:r]-tau)
            # print(XX[:,:,0].dtype,U[:,0:r].dtype, S.dtype,V[0:r,:].dtype)
            XX[:,:,0]  = torch.matmul(torch.matmul(U[:,0:r], S),  V[0:r,:])
            tnn  += tnn+torch.sum(S)
            trank = max(trank,r)

        halfn3 = round(n3/2)
        for i in range(1,halfn3):
            U, S, V = svd(X[:,:,i])
            r = torch.count_nonzero(S>tau)
            S = S.type(torch.complex64)
            if r>=1:
                S = torch.diag(S[0:r]-tau)
                XX[:,:,i] = torch.matmul(torch.matmul(U[:,0:r], S),  V[0:r,:])
                tnn  += tnn+torch.sum(S)*2
                trank = max(trank,r)
            
            XX[:,:,n3-i] = XX[:,:,i].conj()

        if n3%2 == 0:
            i = halfn3
            U, S, V = svd(X[:,:,i])
            r = torch.count_nonzero(S>tau)
            S = S.type(torch.complex64)
            if r>=1:
                S = torch.diag(S[0:r]-tau)
                XX[:,:,i] = torch.matmul(torch.matmul(U[:,0:r], S), V[0:r,:])
                tnn  += tnn+torch.sum(S)
                trank = max(trank,r)

        tnn = tnn/n3
        XX = torch.fft.ifft(XX).real
        return XX, tnn


    def T_SVD_2(self,X, k=50):
        '''
        apply tensor-SVD and soft thresholding
        '''
        W_bar = np.empty((X.shape[0], X.shape[1], 0), complex)
        D = np.fft.fft(X)
        for i in range (3):
            if i < 3:
                U, S, V = svd(D[:, :, i], full_matrices = False)
                S = np.diag(S[:k])
                w = np.dot(np.dot(U[:,:k], S), V[:k,:])
                W_bar = np.append(W_bar, w.reshape(X.shape[0], X.shape[1], 1), axis = 2)
            if i == 3:
                W_bar = np.append(W_bar, (w.conjugate()).reshape(X.shape[0], X.shape[1], 1))
        return np.fft.ifft(W_bar).real


    def SVDShrink(self, X, tau):
        '''
        apply tensor-SVD and soft thresholding
        '''
        W_bar = torch.complex(torch.empty(size = (X.shape[0], X.shape[1], 0)), torch.empty(size = (X.shape[0], X.shape[1], 0)))
        D = torch.fft.fft(X)
        r = r
        for i in range (3):
            if i < 3:
                U, S, V = torch.svd(D[:, :, i])
                S = self.SoftShrink(S[0:r], tau)
                w = torch.matmul(torch.matmul(U[:,0:r], S), V[0:r,:])
                W_bar = torch.append(W_bar, w.reshape(X.shape[0], X.shape[1], 1), axis = 2)
            if i == 3:
                W_bar = torch.append(W_bar, (w.conjugate()).reshape(X.shape[0], X.shape[1], 1))
        return torch.fft.ifft(W_bar).real


    def T_SVD(self, Y, k = 50):
        print('=== t-SVD: rank={} ==='.format(k))
        [n1, n2, n3] = Y.shape
        XX = torch.complex(torch.empty(size= Y.shape),torch.empty(size= Y.shape)).to(device)
        X = torch.fft.fft(Y)

        U, S, V = torch.svd(X[:,:,0])
        # print(U,S,V)
        print("rank_before = {}".format(len(S)))
        S = S.type(torch.complex64)
        if k>=1:
            S = torch.diag(S[0:k])
            # print(XX[:,:,0].dtype,U[:,0:r].dtype, S.dtype,V[0:r,:].dtype)
            XX[:,:,0]  = torch.matmul(torch.matmul(U[:,0:k], S), V[:,:k].T)

        halfn3 = round(n3/2)
        for i in range(1,halfn3):
            U, S, V = torch.svd(X[:,:,i])
            S = S.type(torch.complex64)
            if k>=1:
                S = torch.diag(S[0:k])
                XX[:,:,i] = torch.matmul(torch.matmul(U[:,0:k], S), V[:,:k].T)
            
            XX[:,:,n3-i] = XX[:,:,i].conj()

        if n3%2 == 0:
            i = halfn3
            U, S, V = torch.svd(X[:,:,i])
            S = S.type(torch.complex64)
            if k>=1:
                S = torch.diag(S[0:k])
                XX[:,:,i] = torch.matmul(torch.matmul(U[:,0:k], S), V[:,:k].T)

        XX = torch.fft.ifft(XX).real
        # XX[XX<0]=0
        print("rank_after = {}".format(k))
        return XX




    def ADMM(self, X):
        '''
        Solve
        min (nuclear_norm(L)+lambda*l1norm(E)), subject to X = L+E
        L,E
        by ADMM
        '''
        m, n, l = X.shape
        eps = 1e-4
        rho = 1.1
        mu = 1e-4
        mu_max = 1e10
        max_iters = 200
        lamb = 1/math.sqrt(max(m, n) * l)
        L = torch.zeros((m, n, l)).to(device)
        E = torch.zeros((m, n, l)).to(device)
        Y = torch.zeros((m, n, l)).to(device)
        iters = 0
        while True:
            iters += 1
            # update L(recovered image)
            L_new,tnn = self.SVDShrink2(X - E - (1/mu) * Y, 1/mu)
            # print(L_new)
            # update E(noise)
            E_new = self.SoftShrink(X - L_new - (1/mu) * Y, lamb/mu)
            # print(E_new)
            dY = L_new + E_new - X
            Y += mu * dY
            mu = min(rho * mu, mu_max)

            if self.converged(L, E, X, L_new, E_new).item()<eps or iters >= max_iters:
                return L_new, E_new
            else:
                L, E = L_new, E_new
                obj1, obj2 = 0, 0
                for j in range(l):
                    obj1 +=torch.norm(E[:,:,j],p=1)
                    obj2 +=torch.norm(dY[:,:,j],p=2)

                if iters == 1 or iters%10 == 0:
                    print(iters,": ", "mu=",mu, "obj=", (tnn+lamb*obj1).item()
                    ,"err=",obj2.item(),"chg=",self.converged(L, E, X, L_new, E_new).item())


if __name__ == "__main__":
    # Load Data
    X = torch.tensor(np.array(Image.open(r'lena.bmp'))).to(device)

    # add noise(make some pixels black at the rate of 10%)
    k = torch.rand(X.shape[0],X.shape[1],X.shape[2]) > 0.1
    k = k.to(device)
    # K = np.empty((X.shape[0], X.shape[1], 0), np.uint8)
    # for i in range (X.shape[2]):
    #     K = np.append(K, k.reshape(X.shape[0], X.shape[1], 1), axis = 2)
    X_bar = X * k

    # image denoising
    model = TRPCA()
    print("trpcaing")
    # model = model.to(device)
    L, E = model.ADMM(X_bar)

    L = L.cpu().numpy()
    E = E.cpu().numpy()
    X = X.cpu().numpy()
    L = np.array(L).astype(np.uint8)
    E = np.array(E).astype(np.uint8)
    io.imsave('trpca_L.jpg',L)
    image_output = Image.fromarray(L)
    # save image
    image_output.save("new_panda.jpg")
    X_bar = np.array(X_bar.cpu().numpy()).astype(np.uint8)
    print("psnr: ",psnr(L,X))
    plt.subplot(131)
    plt.imshow(X)
    plt.title('original image')
    plt.subplot(132)
    plt.imshow(X_bar)
    plt.title('image with noise')
    plt.subplot(133)
    plt.imshow(L)
    plt.title('recovered image')
    plt.show()
