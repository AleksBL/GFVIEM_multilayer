import numpy as np
import matplotlib.pyplot as plt
def inner(v,u):
    return (v.conj()*u).sum()

def Conj_Grad_Alg(x,yinit,tol,maxit,COP,COP_T):
    cond=True
    it=0
    z=COP_T(x); g=z-COP_T(COP(yinit))
    p=np.zeros(g.shape,dtype=g.dtype); p+=g; w=COP(yinit); v=COP(p)
    y=np.zeros(yinit.shape,dtype=yinit.dtype); y+=yinit
    while cond:
        a=inner(g,g)/inner(v,v); y=y+a*p
        if np.mod(it,3)==0: w=COP(y)
        else: w=w+a*v; 
        g2=z-COP_T(w)
        b=inner(g2,g2)/inner(g,g)
        del g; g=g2; del g2
        p=g+b*p; 
        if np.mod(it,10)==0: v=COP(p)
        else: v=COP(g)+b*v
        
        error=1-inner(w,x)*inner(x,w)/(inner(w,w)*inner(x,x))
        it+=1
        if np.mod(it,20)==0:
            print(error)
            #plt.matshow((y-x)[:,:,y.shape[2]//2,2].real); plt.colorbar()
            fig, (ax1, ax2) = plt.subplots(1, 2)
            #fig.suptitle('log10(error): '+str(np.round(np.log10(error),5)), + 'Iteration: ', str(it))
            img = ax1.matshow((y)[:,:,y.shape[2]//2,0].real);fig.colorbar(img)
            ax2.plot((y)[y.shape[0]//2,y.shape[1]//2,:,0].real)
            plt.pause(0.01)
        if error<tol or it>maxit:
            cond=False
    return y

def fftZP(A):
    Nx=A.shape[0]; Ny=A.shape[1]; Nz=A.shape[2]; 
    t1=0; t2=0; t3=0
    if np.mod(Nx,2)==0: t1=1
    if np.mod(Ny,2)==0: t2=1
    if np.mod(Nz,2)==0: t3=1
    p00=Nx//2-t1; p01=Nx//2
    p10=Ny//2-t2; p11=Ny//2
    p20=Nz//2-t3; p21=Nz//2
    return np.pad(A,((p00,p01),(p10,p11),(p20,p21)))

def Pil_ud(A,Nx,Ny,Nz):
    dim=A.shape
    t1=0; t2=0; t3=0
    if np.mod(Nx,2)==0: t1=1
    if np.mod(Ny,2)==0: t2=1
    if np.mod(Nz,2)==0: t3=1
    p00=Nx//2-t1; p01=Nx//2
    p10=Ny//2-t2; p11=Ny//2
    p20=Nz//2-t3; p21=Nz//2
    return A[p00:dim[0]-p01,p10:dim[1]-p11,p20:dim[2]-p21]    
