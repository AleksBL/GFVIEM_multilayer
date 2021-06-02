#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 17:26:16 2020

@author: aleks
"""
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import trange
from time import time
import LorteFly
import dill

norm=np.linalg.norm; linsp=np.linspace
Arr=np.array
### Test on box ####
lx=5; ly=5; lz=5
verts_kasse=[(0,0,0),(0,ly,0),(lx,ly,0),(lx,0,0),(0,0,lz),(0,ly,lz),(lx,ly,lz),(lx,0,lz)]
faces_kasse=[(0,1,2,3),(7,6,5,4),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)]

verts_trekant=[(0,0,0),(0,ly,0),(lx,ly,0),(lx,0,0),(0,0,lz),(0,ly,lz)]
faces_trekant=[(0,1,2,3),(0,1,5,4)[::-1],(2,3,4,5)[::-1],(0,3,4),(1,2,5)[::-1]]


verts_L=[(0,0,0),(5,0,0),(5,0,5),(4,0,5),(4,0,1),(0,0,1),(0,5,0),(5,5,0),(5,5,5),(4,5,5),(4,5,1),(0,5,1)]
faces_L=[(0,1,4,5),(1,2,3,4),(8,9,3,2),(2,1,7,8),(4,3,9,10),(5,4,10,11),(0,1,7,6)[::-1],(0,5,11,6),(6,11,10,7),(10,9,8,7)]

class Structure:
    def __init__(self,v,f,e_ref=np.nan,ep=np.nan,k0=np.nan,uv=np.array([1,0,0]),name='noname'):
        #print('Tal, det er bare bogstaver for mig')
        coefs=np.zeros((len(f),4))
        B1213=[]
        B1314=[]
        faces_list=[]
        verts_list=[]
        for i in range(len(f)):
            r12 = Arr(v[f[i][1]])-Arr(v[f[i][0]])
            r13 = Arr(v[f[i][2]])-Arr(v[f[i][0]])
            if len(f[i])==4:
                r14 = Arr(v[f[i][3]])-Arr(v[f[i][0]])
            n = np.cross(r12,r13)
            n = n/norm(n)
            coefs[i,0:3] =  n
            coefs[i,3]   =  -(n[0]*v[f[i][0]][0]+n[1]*v[f[i][0]][1]+n[2]*v[f[i][0]][2])
            B1213+=[np.array([r12,r13,n]).T]
            if len(f[i])==4:
                B1314+=[np.array([r13,r14,n]).T]
            elif len(f[i])==3:
                B1314+=[None]
            
            faces_list+=[f[i]]
            #verts_list+=[v[i]]
        
        self.B1213  = B1213; self.B1314  = B1314
        self.coefs  = coefs
        self.verts  = v
        self.faces  = f
        self.Nfaces = len(faces_list)
        self.Nverts = len(v)
        self.Grid   = None
        self.e_ref  = e_ref
        self.ep     = ep
        self.k      = k0*e_ref**0.5
        self.faces_discrete_points = []
        MAX=0
        for v1 in v:
            for v2 in v:
                d=norm(Arr(v1)-Arr(v2))
                if d>MAX:
                    MAX=d
        self.maxdistance=MAX*1.1
        self.outsideboxes       = None
        self.BoundaryPoints     = None
        self.BoundaryNvecs      = None
        self.BoundaryVolFrac    = None
        self._BoundaryIndsRough_= None
        self.MarkCorners        = None
        self.CellCornerDists    = None
        self.InsideOutside      = None
        self.Check_Ordering_of_Facepoints()
        self.Check_Direction_of_Plane_Normal_Vectors()
        self.name=name
        self.BoundaryLimits=None
        self.EpsilonMat=None
        
    
    def Set_Epsilon(self,E_REF=np.nan,EP=np.nan):
        self.e_ref = E_REF
        self.ep    = EP

    def Dist(self,pin):
        #Vectorised distance function, returns signed distance to all faces of the N x 3 array of points given.
        p=np.zeros(pin.shape); p+=pin
        if p.shape==(3,): p=np.repeat(p[np.newaxis,:],1,axis=0)
        lp=len(p[:,0])
        d_infplan=np.zeros((lp,self.Nfaces))
        sign_dot_nvec=np.zeros((lp,self.Nfaces))
        in_out_plane=np.ones((lp,self.Nfaces))
        
        for i in range(self.Nfaces):
            vfi0=self.verts[self.faces[i][0]]
            Zero=np.repeat(Arr(vfi0)[np.newaxis,...],lp,axis=0)
            b1213inv=np.linalg.inv(self.B1213[i])
            if self.B1314[i] is None:
                b1314inv = None
            else:
                b1314inv=np.linalg.inv(self.B1314[i])
            
            M=np.repeat(self.coefs[i,0:3][np.newaxis,:],lp,axis=0)
            dot=(M*p).sum(axis=1)
            d_infplan[:,i]=dot+self.coefs[i,3]
            p2=p-Zero
            dot2=(M*p2).sum(axis=1); sign_dot_nvec[:,i]=np.sign(dot2)
            planp=p2-np.outer(dot2,self.coefs[i,0:3])
            planp_1213 = np.zeros((lp,3))
            planp_1314 = np.zeros((lp,3))
            
            for j in range(3):
                planp_1213[:,j] = (b1213inv[j,0]*planp[:,0]+b1213inv[j,1]*planp[:,1]+b1213inv[j,2]*planp[:,2])
                if b1314inv is None: pass
                else: planp_1314[:,j] = (b1314inv[j,0]*planp[:,0]+b1314inv[j,1]*planp[:,1]+b1314inv[j,2]*planp[:,2])
            
            ind1=np.where(planp_1213[:,0]<0)
            ind2=np.where(planp_1213[:,1]<0)
            ind3=np.where(planp_1213[:,0]+planp_1213[:,1]>1)
            ind1213=np.union1d(ind1,np.union1d(ind2,ind3))
            if b1314inv is None:
                ind1314=[]
                outind=ind1213.copy()
            else:
                ind1=np.where(planp_1314[:,0]<0)
                ind2=np.where(planp_1314[:,1]<0)
                ind3=np.where(planp_1314[:,0]+planp_1314[:,1]>1)
                ind1314=np.union1d(ind1,np.union1d(ind2,ind3))
                outind=np.intersect1d(ind1213,ind1314)
            
            d_infplan[outind,i]=np.inf
            
            ##### Projection outside 4-gon #####
            
            pout=p[outind]
            npo=len(pout[:,0])
            inds_out=np.arange(0,npo)
            
            if b1314inv is None:
                R1=Arr(self.verts[self.faces[i][0]]); R2=Arr(self.verts[self.faces[i][1]])
                R3=Arr(self.verts[self.faces[i][2]]);
                LIST=Arr([[R1,R2],[R2,R3],[R3,R1]])
                t=np.zeros((npo,3))
                dout=np.zeros((npo,3))
                COUNT=3
            else:
                R1=Arr(self.verts[self.faces[i][0]]); R2=Arr(self.verts[self.faces[i][1]])
                R3=Arr(self.verts[self.faces[i][2]]); R4=Arr(self.verts[self.faces[i][3]])
                LIST=Arr([[R1,R2],[R2,R3],[R3,R4],[R4,R1]])
                t=np.zeros((npo,4))
                dout=np.zeros((npo,4))
                COUNT=4
            
            for j in range(COUNT):
                P=LIST[j,0,:]; Q=LIST[j,1,:]
                line_vec=(Q-P)/np.linalg.norm(Q-P)
                t=-(((P[0]-pout[:,0])*(Q[0]-P[0]))+
                        ((P[1]-pout[:,1])*(Q[1]-P[1]))+
                        ((P[2]-pout[:,2])*(Q[2]-P[2])))/np.linalg.norm(Q-P)**2
                PP=np.zeros((npo,3)); PQ=np.zeros((npo,3))
                
                PP[:,0]=P[0]-pout[:,0]; PP[:,1]=P[1]-pout[:,1]; PP[:,2]=P[2]-pout[:,2]
                PQ[:,0]=Q[0]-pout[:,0]; PQ[:,1]=Q[1]-pout[:,1]; PQ[:,2]=Q[2]-pout[:,2]
                
                NPP=np.sqrt(PP[:,0]**2+PP[:,1]**2+PP[:,2]**2)
                NPQ=np.sqrt(PQ[:,0]**2+PQ[:,1]**2+PQ[:,2]**2)
                
                PP=PP*(1/np.repeat(NPP[...,np.newaxis],3,axis=1))
                PQ=PQ*(1/np.repeat(NPQ[...,np.newaxis],3,axis=1))
                
                cosangles1=PP[:,0]*line_vec[0]+PP[:,1]*line_vec[1]+PP[:,2]*line_vec[2]
                cosangles2=PQ[:,0]*line_vec[0]+PQ[:,1]*line_vec[1]+PQ[:,2]*line_vec[2]
                
                inds_online = np.where(np.sign(cosangles1*cosangles2)==-1)[0]
                otherinds   = np.setdiff1d(inds_out,inds_online)
                
                d_online=np.sqrt( ( (P[0]-pout[inds_online,0])+(Q[0]-P[0]) * t[inds_online] )**2
                                + ( (P[1]-pout[inds_online,1])+(Q[1]-P[1]) * t[inds_online] )**2
                                + ( (P[2]-pout[inds_online,2])+(Q[2]-P[2]) * t[inds_online] )**2 )
                
                d_otherinds=np.min(np.array([NPP[otherinds],NPQ[otherinds]]).T,axis=1)
                
                dout[inds_online,j] = d_online
                dout[otherinds  ,j] = d_otherinds
            
            d_infplan[outind,i]=np.min(dout,axis=1)
            in_out_plane[outind,i]=0
        
        TEMP = np.zeros(sign_dot_nvec.shape)
        TEMP+= sign_dot_nvec; TEMP[np.where(TEMP==0)]=1
        
        return np.abs(d_infplan)*TEMP, in_out_plane
    
    def Inside_Struc_Convex(self,p):
        print('Structure needs to be convex for this function to work')
        d,io=self.Dist(p)
        return (np.sum(np.sign(d),axis=1)==-self.Nfaces)
    
    def Outside_Really_Easy(self, points, distances):
        if self.outsideboxes!=None:
            ### Do clever stuff ####
            pass
        return np.where(np.max(np.abs(distances),axis=1)>self.maxdistance)[0]
    
    def Inside_Struc_Line_Check(self,pin,res=2000):
        dist=self.maxdistance
        p=np.zeros(pin.shape); p+=pin
        if p.shape==(3,): p=np.repeat(p[np.newaxis,:],1,axis=0)
        #### Not Done
        dists, ips = self.Dist(p)
        
        r_vec=-np.random.random(3); r_vec=r_vec/norm(r_vec); r_vec=r_vec*dist
        t_line=linsp(0,1,res); line_vecs=np.outer(t_line,r_vec)
        tolerance=dist/res
        # points farther away than self.maxdistance
        inds_really_easy=self.Outside_Really_Easy(p,dists) #np.where(np.max(np.abs(dists),axis=1)>self.maxdistance)[0]
        
        # points not close to boundary compared to step size
        inds_easy=np.where(np.min(np.abs(dists),axis=1)>tolerance)[0]
        inds_easy=np.setdiff1d(inds_easy,inds_really_easy)
        # points close to boundary compared to step size
        inds_hard=np.setdiff1d(np.arange(0,len(p[:,0])),inds_easy)
        inds_hard=np.setdiff1d(inds_hard,inds_really_easy)
        p_easy=p[inds_easy]
        pl=Repeat_Arr_Add_vec(p_easy,line_vecs,res)
        dl,in_out_l=self.Dist(pl)
        dl=Translate_to_NxResx3(len(p_easy[:,0]),res,dl)
        in_out_l=Translate_to_NxResx3(len(p_easy[:,0]),res,in_out_l)
        abs_dist=np.abs(dl); sign_dl=np.sign(dl)
        sign_change       =  np.abs(np.roll(sign_dl,-1,axis=1)-sign_dl)[:,0:res-1,:]/2
        neighbor_in_plane =  in_out_l[:,1:,:]*in_out_l[:,0:res-1,:]
        #print(sign_change,neighbor_in_plane)
        intersects=np.full(len(p[:,0]),np.nan)
        intersects[inds_easy]=np.sum(sign_change*neighbor_in_plane,axis=(1,2))
        intersects[inds_really_easy]=0
        
        det_count=0
        
        for i in inds_hard:
            ic = np.where(np.abs(dists[i,:])==np.abs(dists[i,:]).min())[0]
            for iic in ic:
                iic_count=0
                if ips[i,iic]==1 and np.sign(dists[i,iic])==-1:
                    intersects[i]=1
                    det_count+=1
                    break
                elif ips[i,iic]==1 and np.sign(dists[i,iic])==1:
                    intersects[i]=0
                    det_count+=1
                    break
                elif ips[i,iic]==0:
                    pass # print(i, 'is undetermined')
        
        # print('\n N really easy: ', len(inds_really_easy),
        #       '\n N easy: ', len(inds_easy),
        #       '\n N hard: ', det_count,
        #       '\n N undetermined: ', np.sum(np.[isnan(intersects)))
        
        return np.mod(intersects,2)
    
    def Set_Grid(self,Grid):
        self.Grid = Grid
    
    def Set_Discrete_Plane(self,res=5):
        if self.Grid==None:
            pass
        else:
            dx=self.Grid[0][1,0,0]-self.Grid[0][0,0,0]
            Disc_Faces=[]
            for f in range(self.Nfaces):
                #print(f)
                C=np.zeros((4,3))
                for i in range(len(self.faces[f])):
                    C[i,:]=Arr(self.verts[self.faces[f][i]])
                C12=C[1,:]-C[0,:]
                C13=C[2,:]-C[0,:]
                if self.B1314[f] is not None:
                    C14=C[3,:]-C[0,:]
                cos1213=np.dot(C12,C13)/(norm(C12)*norm(C13))
                if self.B1314[f] is not None:
                    cos1314=np.dot(C13,C14)/(norm(C13)*norm(C14))
                u1213=np.linspace(0,1,int(res*norm(C12)/dx))
                t1213=np.linspace(0,1,int(res*(1-cos1213**2)**(1/2)*norm(C13)/dx))
                
                if self.B1314[f] is not None:
                    u1314=np.linspace(0,1,int(res*norm(C13)/dx))
                    t1314=np.linspace(0,1,int(res*(1-cos1314**2)**(1/2)*norm(C14)/dx))
                
                U,V=np.meshgrid(u1213,t1213)
                inds=np.where(U+V<=1)
                U=U[inds]; V=V[inds]
                plan1=np.repeat(C[0,:][np.newaxis,...],len(U),axis=0)+np.outer(U,C12)+np.outer(V,C13)
                if self.B1314[f] is not None:
                    U,V=np.meshgrid(u1314,t1314)
                    inds=np.where(U+V<=1)
                    U=U[inds]; V=V[inds]
                    inds=np.where(U>0)
                    U=U[inds]; V=V[inds]
                    plan2=np.repeat(C[0,:][np.newaxis,...],len(U),axis=0)+np.outer(U,C13)+np.outer(V,C14)
                    plan=np.vstack([plan1,plan2])
                    Disc_Faces+=[plan]
                else:
                    Disc_Faces+=[plan1]
            
            self.faces_discrete_points=Disc_Faces
    
    def Check_Direction_of_Plane_Normal_Vectors(self):
        # Normal vectors must point out of the scatterer
        BOOL=np.zeros(self.Nfaces,dtype=bool)
        f_new=[]
        for i in range(self.Nfaces):
            n=self.coefs[i,0:3]
            Zero=np.zeros(3)
            for j in range(len(self.faces[i])):
                v=self.verts[self.faces[i][j]]
                Zero+=Arr(v)/len(self.faces[i])
            test_p=Zero-n*self.maxdistance/600
            #print(test_p)
            Truth=self.Inside_Struc_Line_Check(test_p,res=40000)[0]
            #print(Truth)
            if Truth==1:
                f_new+=[self.faces[i]]
                BOOL[i]=0
            elif Truth==0:
                f_new+=[self.faces[i][::-1]]
                BOOL[i]=1
        print('Normal vectors of ',np.where(BOOL),' needs reversing')
    
    def Check_Ordering_of_Facepoints(self):
        f_new=[]
        BOOL=np.zeros(self.Nfaces, dtype=bool)
        for i in range(self.Nfaces):
            f=self.faces[i]
            
            if self.B1314[i] is None:
                pass
            else:
                Zero=Arr(self.verts[f[0]])
                r01=Arr(self.verts[f[1]])-Zero
                r02=Arr(self.verts[f[2]])-Zero
                r03=Arr(self.verts[f[3]])-Zero
                if np.dot(np.cross(r01,r02),np.cross(r02,r03))>0:
                    f_new+=[f]
                    BOOL[i]=False
                else:
                    f_new+=[(f[0],f[2],f[1],f[3])]
                    BOOL[i]=True
        
        print('Faces ',np.where(BOOL),' is badly written')        
    
    def Detect_Boundaries(self,opl=15,opl_highres=30):
        # equidistance Nx * Ny * Nz grid
        #opl uneven number
        if self.Grid==None:
            print('Set Grid')
        elif self.faces_discrete_points==[]:
            print('Give discrete points of faces')
        else:
            X=self.Grid[0]; Y=self.Grid[1]; Z=self.Grid[2]
            Nx=len(X[:,0,0]); Ny=len(Y[0,:,0]); Nz=len(Z[0,0,:])
            Dx0=X[0,0,0]; Dx1=X[Nx-1,0,0]
            Dy0=Y[0,0,0]; Dy1=Y[0,Ny-1,0]
            Dz0=Z[0,0,0]; Dz1=Z[0,0,Nz-1]
            dx=X[1,0,0]-X[0,0,0]; dy=Y[0,1,0]-Y[0,0,0];  dz=Z[0,0,1]-Z[0,0,0]
            if np.abs(dx-dy)>1e-10 or np.abs(dx-dz)>1e-10 or np.abs(dy-dz)>1e-10: print('Check grid spacing')
            Cornerx, Cornery, Cornerz = np.meshgrid(linsp(Dx0-dx/2,Dx1+dx/2,Nx+1),
                                                    linsp(Dy0-dy/2,Dy1+dy/2,Ny+1),
                                                    linsp(Dz0-dz/2,Dz1+dz/2,Nz+1),
                                                    indexing='ij')
            
            Tal=(Nx+1)*(Ny+1)*(Nz+1)
            R_Corner=np.vstack([Cornerx.reshape(Tal),
                                Cornery.reshape(Tal),
                                Cornerz.reshape(Tal)]).T
            
            
            
            D_Corner, in_out_plane=self.Dist(R_Corner)
            D_Corner=D_Corner.reshape(Nx+1,Ny+1,Nz+1,self.Nfaces)
            in_out_plane=in_out_plane.reshape(Nx+1,Ny+1,Nz+1,self.Nfaces)
            Rough_Bound_inds = RoughBoundary( D_Corner,dx/1.99)
            n_boundary=len(Rough_Bound_inds[0])
            
            xl,yl,zl=np.meshgrid(linsp(-dx/2+1/(2*10000),dx/2-1/(2*10000),opl),
                                   linsp(-dy/2+1/(2*10000),dy/2-1/(2*10000),opl),
                                   linsp(-dz/2+1/(2*10000),dz/2-1/(2*10000),opl),
                                   indexing='ij')
            
            rloc = np.vstack([xl.reshape(opl**3),
                              yl.reshape(opl**3),
                              zl.reshape(opl**3)]).T
            
            
            OC=dx*0.499999 ###(3*((dx/2)**2))**(1/3)
            false_positive=np.zeros(n_boundary)
            Corner_marker=np.zeros(n_boundary)
            nvec=np.zeros((n_boundary,3))
            
            vol_frac=np.zeros(n_boundary)
            Vert_Count=0
            D_CELL=self.Dist(np.vstack([X.reshape(Nx*Ny*Nz),
                                        Y.reshape(Nx*Ny*Nz),
                                        Z.reshape(Nx*Ny*Nz)]).T)[0].reshape(Nx,Ny,Nz,self.Nfaces)
            self.CellCornerDists=D_CELL
            for b in trange(n_boundary):
                i1,i2,i3=Rough_Bound_inds[0][b],Rough_Bound_inds[1][b],Rough_Bound_inds[2][b]
                rb=np.array([X[i1,i2,i3],Y[i1,i2,i3],Z[i1,i2,i3]])
                Vert_in_cell=False
                for V in self.verts:
                    if rb[0]-dx/2<V[0]<rb[0]+dx/2 and rb[1]-dy/2<V[1]<rb[1]+dy/2 and rb[2]-dz/2<V[2]<rb[2]+dz/2:
                        Vert_in_cell=True
                
                d_cell=D_CELL[i1,i2,i3,:]; 
                close=np.where(np.abs(d_cell)<=OC)[0]
                dloc=self.Dist(rloc+rb)[0]
                
                if Vert_in_cell==False:
                    if len(close)==1:
                        vfrac=np.sum(np.sign(dloc[:,close])==-1)/opl**3+np.sum(np.sign(dloc[:,close])==0)/(2*opl**3)
                        if 1>vfrac>0:
                            vol_frac[b]=vfrac
                            nvec[b,:]=self.coefs[close[0]][0:3]
                        else:
                            false_positive[b]=1
                    elif len(close)==2:
                        i_p1=close[0]; i_p2=close[1]
                        np1=self.faces_discrete_points[i_p1].shape[0]
                        np2=self.faces_discrete_points[i_p2].shape[0]
                        d12,ip12=self.Dist(self.faces_discrete_points[i_p1][np1//2])
                        d21,ip21=self.Dist(self.faces_discrete_points[i_p2][np2//2])
                        Situation=Determine_Situation(np.sign(d21[0,i_p1]), np.sign(d12[0,i_p2]))
                        inds_p1=np.where(dloc[:,i_p1]<=0); inds_p2=np.where(dloc[:,i_p2]<=0)
                        inds_inside=Get_Points_inside_planes(inds_p1,inds_p2,Situation)
                        vfrac=len(inds_inside)/opl**3
                        if 1>vfrac>0: 
                            vol_frac[b]=vfrac
                            Ant_1=N_Inside(self.faces_discrete_points[i_p1],rb,dx/2)
                            Ant_2=N_Inside(self.faces_discrete_points[i_p2],rb,dx/2)
                            if Ant_1+Ant_2>0:
                                nvec[b,:]=(Ant_1*self.coefs[i_p1][0:3]+Ant_2*self.coefs[i_p2][0:3])/(Ant_1+Ant_2)
                                nvec[b,:]*=(1/norm(nvec[b,:]))
                        else:
                            false_positive[b]=1
                    
                    elif len(close)==0:
                        print('false pos')
                        false_positive[b]=1
                    elif len(close)>2:
                        print(close)
                
                elif Vert_in_cell==True:
                    #print(np.sum(rloc,axis=0))
                    #dloc_hr,ip_hr=self.Dist(rb+rloc_hr)
                    Corner_marker[b]=1
                    Vert_Count+=1
                    Ant=np.zeros(len(close))
                    for i in range(len(close)):
                        Ant[i]=N_Inside(self.faces_discrete_points[close[i]],rb,2*dx)
                        nvec[b,:]+=Ant[i]*self.coefs[close[i]][0:3]
                    if np.sum(Ant)>0:
                        nvec[b,:]*=(1/np.sum(Ant))
                        nvec[b,:]*=(1/norm(nvec[b,:]))
                    intersects=self.Inside_Struc_Line_Check(rloc+rb,res=500)
                    inds_undet=np.where(np.isnan(intersects))[0]
                    if len(inds_undet)>0:
                        intersects2=self.Inside_Struc_Line_Check((rloc+rb)[inds_undet],res=1000)
                        intersects[inds_undet]=intersects2
                        inds_undet2=np.where(np.isnan(intersects)[0])
                        if len(inds_undet2)>0:
                            intersects3=self.Inside_Struc_Line_Check((rloc+rb)[inds_undet][inds_undet2],res=5000)
                            intersects[inds_undet][inds_undet2]=intersects3
                    intersects[np.isnan(intersects)]=0
                    vfrac=np.sum(intersects)/opl**3
                    if 1>vfrac>0:
                        vol_frac[b]=vfrac
                        print(vfrac, rb)
                    else:
                        false_positive[b]=1 
            
            print('\n Vertex count:', Vert_Count, '\n Actual number of vertecies:', len(self.verts),
                  '\n No. false positives:',  np.sum(false_positive),'\n nvec is nan:', np.isnan(nvec).any(),
                  '\n Boundary points:', n_boundary-np.sum(false_positive))
            
            inds_final=np.zeros((n_boundary-int(np.sum(false_positive)),3),dtype=int)
            nvecs_final=np.zeros((n_boundary-int(np.sum(false_positive)),3))
            vol_frac_final=np.zeros(n_boundary-int(np.sum(false_positive)))
            Rough_Inds=np.zeros((n_boundary,3),dtype=int)
            Corner_marker_final=np.zeros(n_boundary-int(np.sum(false_positive)))
            it=0
            for i in range(n_boundary):
                i1,i2,i3=Rough_Bound_inds[0][i],Rough_Bound_inds[1][i],Rough_Bound_inds[2][i]
                if false_positive[i]==0:
                    inds_final[it,:]=Arr([i1,i2,i3])
                    nvecs_final[it,:]=nvec[i,:]
                    vol_frac_final[it]=vol_frac[i]
                    Corner_marker_final[it]=Corner_marker[i]
                    it+=1
                Rough_Inds[i,:]=Arr([i1,i2,i3])
            self.BoundaryPoints = inds_final
            self.BoundaryNvecs  = nvecs_final
            self.BoundaryVolFrac= vol_frac_final
            self._BoundaryIndsRough_ = Rough_Inds
            self.MarkCorners=Corner_marker_final
    
    #### Optimize this shitty function and make more stabile ######
    def PointsInsideOutside(self):
        if self.BoundaryPoints is None:
            print('Needs Bondaries First')
        else:
            Nx = len(self.Grid[0][:,0,0])
            Ny = len(self.Grid[0][0,:,0])
            I_O_B=np.zeros((Nx,Ny,10))
            for ix in range(Nx):
                for iy in range(Ny):
                    Points=[]
                    indsx=np.where(self.BoundaryPoints[:,0]==ix)[0]
                    indsy=np.where(self.BoundaryPoints[:,1]==iy)[0]
                    inds=np.intersect1d(indsx,indsy)
                    Points=self.BoundaryPoints[inds,:]
                    # for I in range(self.BoundaryPoints.shape[0]):
                    #     p=self.BoundaryPoints[I,:]
                    #     if p[0]==ix and p[1]==iy:
                    #         Points+=[p]
                    if len(Points[:,0])==0:
                        pass
                    elif len(Points[:,0])==1:
                        print('Is there a pointy edge at indecies',Points,'?')
                    else:
                        p_it=0
                        for i in range(len(Points[:,0])-1):
                            p0=Points[i  ,:]
                            p1=Points[i+1,:]
                            if p1[2]-p0[2]>1:
                                m=int((p1[2]+p0[2])//2)
                                rxyz=Arr([self.Grid[0][ix,iy,m],self.Grid[1][ix,iy,m],self.Grid[2][ix,iy,m]])
                                inside_iter=0
                                for ite in range(3):
                                    if self.Inside_Struc_Line_Check(rxyz)==1:
                                        inside_iter+=1
                                if inside_iter<3:
                                    if self.Inside_Struc_Line_Check(rxyz,res=5000)==1:
                                        inside_iter+=1
                                
                                if inside_iter>=3:
                                    I_O_B[ix,iy,p_it]   = p0[2]
                                    I_O_B[ix,iy,p_it+1] = p1[2]
                                    p_it+=2
                                
            
            self.BoundaryLimits=I_O_B
    
    def MakeEpsilon(self):
        if self.BoundaryLimits is None: print('Find boundary limits first')
        else:
            Nx = len(self.Grid[0][:,0,0])
            Ny = len(self.Grid[0][0,:,0])
            Nz = len(self.Grid[0][0,0,:])
            Epsilon=np.ones((Nx,Ny,Nz,3,3),dtype=np.complex128)*self.e_ref
            Epsilon[:,:,:,0,1]=0; Epsilon[:,:,:,0,2]=0; Epsilon[:,:,:,1,0]=0;
            Epsilon[:,:,:,2,0]=0; Epsilon[:,:,:,1,2]=0; Epsilon[:,:,:,2,1]=0
            
            I=np.eye(3)
            
            for ix in range(Nx):
                for iy in range(Ny):
                    izB  = self.BoundaryLimits[ix,iy,:]
                    #print(izB)
                    if (izB>0).any():
                        imax = (np.where(izB>0)[0].max()+1)//2
                        for i in range(imax):
                            zmin=izB[2*i]; zmax=izB[2*i+1]
                            Epsilon[ix,iy,int(zmin+1):int(zmax),:,:]=np.multiply.outer(np.ones(int(zmax-zmin-1)),I*self.ep)
            
            for i in range(self.BoundaryPoints.shape[0]):
                vf=self.BoundaryVolFrac[i]
                nvec=self.BoundaryNvecs[i,:]
                i1,i2,i3=self.BoundaryPoints[i,0],self.BoundaryPoints[i,1],self.BoundaryPoints[i,2]
                print(i1,i2,i3)
                nn=np.outer(nvec,nvec)
                eps_para=(1-vf)*self.e_ref+vf*self.ep
                eps_perp=((1-vf)/self.e_ref+vf/self.ep)**-1
                Epsilon[i1,i2,i3,:,:]=(eps_perp-eps_para)*nn+I*eps_para
        self.EpsilonMat=Epsilon
    
    def Save(self):
        """try load self.name.txt"""
        dill.dump(self, file = open(self.name+".pickle", "wb"))
    

def Load(fm):
    return dill.load(open(fm+".pickle", "rb"))

def Get_Points_inside_planes(inds1,inds2,Sit):
    if Sit==0:#1:
        return np.intersect1d(inds1,inds2)
    elif Sit==1:#0:
        return np.union1d(inds1,inds2)
    elif Sit==1/2:
        return np.intersect1d(inds1,inds2)

def Determine_Situation(inp1,inp2):
    Situation=1/2
    if inp1==-1 and inp2==-1:
        Situation=0 # Sharp Corner < 180*
    elif inp1==1 and inp2==1:
        Situation=1 # inwards curving corner > 180* 
        print('Mærkelig kant')
    elif inp1==0 and inp2==0:
        Situation=1
        print('Could not determine angle between planes??')
    
    return Situation

def Repeat_Arr_Add_vec(A,line,n):
    Res=np.zeros((A.shape[0]*n,A.shape[1]))
    la=A.shape[0]
    for i in range(la):
        Res[i*n:(i+1)*n,:]=np.repeat(A[i,:][np.newaxis,:],n,axis=0)+line
    return Res

@jit
def Translate_to_NxResx3(n_p,res,dl):
    Res=np.zeros((n_p,res,len(dl[0,:])))
    it=0
    for i_point in range(n_p):
        for i_line in range(res):
            Res[i_point,i_line,:]=dl[it,:]
            it+=1
    return Res

@jit
def N_Inside(A,r,d):
    it=0
    for i in range(len(A[:,0])):
        p=A[i,:]
        if r[0]-d<p[0]<r[0]+d and r[1]-d<p[1]<r[1]+d and r[2]-d<p[2]<r[2]+d:
            it+=1
    return it

@jit
def RoughBoundary(DC,Rad):
    Nx=len(DC[:,0,0,0])-1; Ny=len(DC[0,:,0,0])-1;Nz=len(DC[0,0,:,0])-1
    BOOL=np.zeros((Nx,Ny,Nz))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                H=DC[i:i+2,j:j+2,k:k+2,:]
                T=np.sum(np.sum(np.sum(np.sign(H),axis=0),axis=0),axis=0)
                indt=np.where(np.abs(T)<8)[0]
                for I in range(len(indt)):
                    if ((np.abs(H[:,:,:,indt[I]]))<Rad).any():
                        BOOL[i,j,k]=1
    return np.where(BOOL==1)

def Scatter3D(x,y,z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_xlim((0,5))
    #ax.set_ylim((0,5))
    #ax.set_zlim((0,5))
    ax.scatter(x, y, z)

def Get_Grid(xmm,ymm,zmm,d,k=np.nan):
    deltax=xmm[1]-xmm[0]
    deltay=ymm[1]-ymm[0]
    deltaz=zmm[1]-zmm[0]
    
    idx=int(deltax/d);  r_idx=deltax/d-idx
    idy=int(deltay/d);  r_idy=deltay/d-idy
    idz=int(deltaz/d);  r_idz=deltaz/d-idz
    #print(deltax/d,deltay/d,deltaz/d)
    #print(idx,r_idx,idy,r_idy,idz,r_idz)
    x_cor = d*(1-r_idx)/2
    y_cor = d*(1-r_idy)/2
    z_cor = d*(1-r_idz)/2
    xl=linsp(xmm[0]-x_cor,xmm[1]+x_cor,idx+2)
    yl=linsp(ymm[0]-y_cor,ymm[1]+y_cor,idy+2)
    zl=linsp(zmm[0]-z_cor,zmm[1]+z_cor,idz+2)
    
    
    if k is not np.nan:
        print(np.round((2*np.pi/k)/d,5),' cells per wavelength')
        print('xyz/151**3:', (idx+2)*(idy+2)*(idz+2)/151**3)
    
    return np.meshgrid(xl,yl,zl,indexing='ij')
    
    

# X,Y,Z=Get_Grid([-1.9,1.9],[-1.4,0.95],[-0.025,0.4785],0.009,2*np.pi/0.5)#np.meshgrid(linsp(-4.5,5,71),linsp(0,5,71),linsp(0,,71),indexing='ij')
# print(X.shape,Y.shape,Z.shape)

# ShittyPlane=Structure(LorteFly.verts,LorteFly.faces_tri,e_ref=1,ep=10+1j,name='Fly4_hr')
#ShittyPlane.Set_Grid([X,Y,Z])
#ShittyPlane.Set_Discrete_Plane(res=20)
#ShittyPlane.Detect_Boundaries(opl=16)
#ShittyPlane.PointsInsideOutside()
#ShittyPlane.MakeEpsilon()
#ShittyPlane.Save()



    
    
    
    
    
    
    
    
    
    
# elif Vert_in_cell==True:
#     Vert_Count+=1
#     Ant=np.zeros(len(close))
#     for i in range(len(close)):
#         Ant[i]=N_Inside(self.faces_discrete_points[close[i]],rb,dx/2)
#         nvec[b,:]+=Ant[i]*self.coefs[close[i]][0:3]
#     if np.sum(Ant)>0:
#         nvec[b,:]*=(1/np.sum(Ant))
#         nvec[b,:]*=(1/norm(nvec[b,:]))
                #     intersects=self.Inside_Struc_Line_Check(rloc+rb,res=500)
                #     inds_undet=np.where(np.isnan(intersects))[0]
                #     if len(inds_undet):
                #         intersects2=self.Inside_Struc_Line_Check((rloc+rb)[inds_undet],res=5000)
                #         intersects[inds_undet]=intersects2
                #     intersects[np.isnan(intersects)]=0
                #     vfrac=np.sum(intersects)/opl**3
                #     if vfrac>0: 
                #         vol_frac[b]=vfrac
                #     else:
                #         false_positive[b]=1 
    
# @jit
# def Boundary_detector(DC,in_out_plane,Rad):
#     Nx=len(DC[:,0,0,0])-1; Ny=len(DC[0,:,0,0])-1;Nz=len(DC[0,0,:,0])-1
#     BOOL=np.zeros((Nx,Ny,Nz))
#     Bends=np.zeros((Nx,Ny,Nz))
#     Nfaces=len(DC[0,0,0,:])
#     for i in range(Nx):
#         for j in range(Ny):
#             for k in range(Nz):
#                 H=DC[i:i+2,j:j+2,k:k+2,:]
#                 io=in_out_plane[i:i+2,j:j+2,k:k+2,:]
#                 T=np.sum(np.sum(np.sum(np.sign(H),axis=0),axis=0),axis=0)
#                 indt=np.where(np.abs(T)<8)[0]
#                 for I in range(len(indt)):
#                     if ((np.abs(H[:,:,:,indt[I]]))<Rad).all():
#                         BOOL[i,j,k]=1
#                         if (io[:,:,:,indt]==0).any():
#                             Bends[i,j,k]=1
                             
#     return np.where(BOOL==1), np.where(Bends==1)
    
