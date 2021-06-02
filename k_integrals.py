import numpy as np
from scipy.integrate import quad, simpson, romberg, fixed_quad
from scipy.special import j0,j1, binom, jv, ellipk, y0, struve
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d as inte
from scipy.interpolate import griddata
from tqdm import tqdm
from joblib import Parallel, delayed
from numba import njit
import numba_scipy
dz = 0.05








class interp:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
    def interpolate(self, r):
        return griddata((self.x,self.y), self.z,r[:,0:2],method='linear')


def Integral_vs_brute_force(f, r, z , k1,i_max, div, n, shift = 0):
    res_1 = Integral(lambda x,y,z,v: f(x+shift, y, z, v), k1, r, z, n_pi = n, force_fallback = True)
    Ii=0.0
    Ir=0.0
    errors = []
    for i in range(i_max):
        res_r = Quad(lambda x: f(x+shift,r, z, k1).real, i/div,(i+1)/div)
        res_i = Quad(lambda x: f(x+shift,r, z, k1).imag, i/div,(i+1)/div)
        Ir+= res_r[0]
        Ii+= res_i[0]
        errors+=[max(res_i[1], res_r[1])]
    return res_1, Ir + 1j* Ii

def Integral(f, k1, rho, dz, P = [], n_pi = 30, int_tol = 1e-7, force_fallback = False):
    if len(P) == 0:
        P = [0, k1, 5 * k1,  10 * k1, 20 * k1]
    
    Ir = 0.0
    Ii = 0.0
    n = len(P)
    for tæller in range(n-1):
        a,b = P[tæller],P[tæller+1]
        res_r = Quad(lambda x: f(x, rho, dz, k1).real, a, b)
        res_i = Quad(lambda x: f(x, rho, dz, k1).imag, a, b)
        if max(res_r[1], res_i[1]) > int_tol:
            print(max(res_r[1], res_i[1]))
            assert 1 == 0
        Ir += res_r[0]
        Ii += res_i[0]
    tail_r = 0.0#GWA(lambda x: f(x, rho, dz, k1).real, P[-1], rho, dz, n_PI = n_pi)
    tail_i = 0.0#GWA(lambda x: f(x, rho, dz, k1).imag, P[-1], rho, dz, n_PI = n_pi)
    if True:#force_fallback:
        tail_r = np.inf
    
    if np.isinf(tail_r) or np.isinf(tail_i) or np.isnan(tail_r) or np.isnan(tail_i):
        #print('fallback to brute force\n')
        if rho>0:
            div = (2 * rho)
        else:
            div = (2 * dz)
            
        it = 0
        tail_r  = 0.0;  tail_i = 0.0
        cond = True
        counter = 0
        while cond  == True:
            delta_r, e_r = Quad(lambda x: f(x, rho, dz, k1).real, P[-1] + it * np.pi/ div, P[-1] + (it+1) * np.pi/ div)
            delta_i, e_i = Quad(lambda x: f(x, rho, dz, k1).imag, P[-1] + it * np.pi/ div, P[-1] + (it+1) * np.pi/ div)
            tail_r += delta_r
            tail_i += delta_i
            it+=1
            change_frac =(abs(delta_r) + abs(delta_i)) 
            if change_frac < int_tol:
                counter += 1
            else:
                counter = 0
            if counter == 10:
                cond = False
    
    Ir += tail_r
    Ii += tail_i
    
    return Ir + 1j* Ii



def Quad(f,a,b,abstol = 1e-7, max_n = 7):
    res = quad(f,a,b)
    if res[1] > abstol or np.isnan(res[0]):
        for i_sub in range(3,max_n):
            xsub = np.linspace(a, b, 2**i_sub)
            res_sub_val   = []
            res_sub_error = []
            for ii in range(i_sub-1):
                xa_sub, xb_sub = xsub[ii], xsub[ii+1]
                res_sub = quad(f, xa_sub, xb_sub)
                res_sub_val  +=[res_sub[0]]
                res_sub_error+=[res_sub[1]]
            res_sub_val   = np.array(res_sub_val  )
            res_sub_error = np.array(res_sub_error)
            if res_sub_error.sum() < abstol:
                break
        #print(i_sub)
        res = (res_sub_val.sum(), res_sub_error.sum())
    return res

def GWA(f, kp_min, rho, dz, n_PI, abstol = 1e-7, q = -1):
    gamma = dz + 1j*rho
    if rho > 1e-2:
        step_size = np.pi / rho
    elif dz > 1e-2:
        step_size = np.pi / dz
    else:
        step_size = 100
    
    x = np.array([i * step_size for i in range(n_PI + 1)])
    i = np.arange(n_PI)
    Partial_integrals = np.zeros(n_PI)
    for j in range(n_PI):
        xa, xb = x[j] + kp_min, x[j+1] + kp_min
        res = Quad(f, xa, xb)
        Partial_integrals[ j ] = res[0]
    
    PI = [np.sum(Partial_integrals[:(p+1)]) for p in range(n_PI)]
    PI = np.array(PI)
    PI_original = PI.copy()
    
    wi =(-1)**i * np.exp( gamma * x[:-1] ) * binom(n_PI, i) * x[:-1]**(n_PI - 2 - q)
    Res = (wi * PI).sum() / (wi.sum())
    n_pi_inside  = 0
    n_pi_inside += n_PI
    
        
    i  = i[:-20]
    x  = x[:-20]
    PI = PI[:-20]
    n_pi_inside-=20
    wi = (-1)**i * np.exp( gamma * x[:-1] ) * binom(n_pi_inside, i) * x[:-1]**(n_pi_inside - 2 - q)
    Res_2 = (wi * PI).sum()/(wi.sum())
    if n_pi_inside < 3 or abs(Res_2 - Res)/Res>abstol or np.isnan(Res) or np.isinf(Res):
        Res = np.inf
    
    if np.isinf(Res) or np.isnan(Res):
        cauchy_count = 0
        for i in range(n_PI-1):
            if abs(PI_original[i]-PI_original[i+1])<abstol:
                cauchy_count += 1
            else:
                cauchy_count = 0
            
            if cauchy_count==3:
                Res = PI_original[i+1]
                break
    if np.isinf(Res) or np.isnan(Res):
        print('GWA failed')
    
    return  Res

@njit
def r_s(kx,k0, ep1 = 1.0, ep2 = 3.5):
    ky1 = np.sqrt(k0**2 * ep1 - kx**2 + 0j)
    ky2 = np.sqrt(k0**2 * ep2 - kx**2 + 0j)
    return ( ky1 - ky2 ) / ( ky1 + ky2 )

@njit
def r_p(kx,k0, ep1 = 1.0, ep2 = 3.5):
    ky1 = np.sqrt(k0**2 * ep1 - kx**2 + 0j)
    ky2 = np.sqrt(k0**2 * ep2 - kx**2 + 0j)
    return ( ep2 * ky1 - ep1 * ky2 ) / ( ep2 * ky1 + ep1 * ky2 )

# def r_s(kx, k0, ep1, ep2, ep3 = 1.5**2)
# def f(x):
#     return I1(x, rho, dz, k1).imag
# np.exp(-dz*x) * j0( rho *  np.sqrt(x**2-1) )


@njit
def w(kp,z,k1):
    kz= np.sqrt(k1**2 - kp**2 + 0j )
    return (1 /(1j * kz)) * (np.exp(1j * kz*(z+dz/2)) -  np.exp(1j * kz*(z-dz/2)) ) * kp / (kz * dz)

# def iw(kp,z,k1):
#     kz= np.sqrt(k1**2 - kp**2 + 0j )
#     return -(1j/kz) *( -1  + np.exp(1j * kz*z)) * kp / kz

@njit
def j0_p(x):
    return -j1(x)

@njit
def j0_pp(x):
    return j1(x)/x - j0(x)

# All integrals have k_x * rho as argument and rho as integration variable
@njit
def I_r_j0(k, a):
    return a * j1(a * k) / k

@njit
def I_r_j0_p(k, a):
    return - np.pi * a * (struve(0.0, a * k) * j1( a * k) - struve(1.0, a * k ) * j0(a * k) ) / ( 2 * np.abs(k) )

@njit
def I_j0_p(k, a):
    return (j0(a * k) - 1) / k

@njit
def I_r_j0_pp(k, a):
    return - ( j0(a*k) + a * j1(a * k) - 1)/k

@njit
def i_i1(kp,r,z,k1, iRZ = False):
    if iRZ==False:
        return r_p(kp, k1) *     j0(kp * r) * kp**2 *  w(kp, z, k1) / k1**2
    if iRZ==True:
        return r_p(kp, k1) * I_r_j0(kp , r) * kp**2 * w(kp, z, k1) / k1**2

@njit
def i_i2(kp,r,z,k1, iRZ = False):
    kz1= np.sqrt(k1**2 - kp**2 + 0j )
    if iRZ == False:
        return r_p(kp, k1) *   j0_p(kp * r)/(kp * r) * kz1**2 * w(kp, z, k1) / k1 **2
    if iRZ == True:
        return r_p(kp, k1) * I_j0_p(kp , r)/(kp    ) * kz1**2 * w(kp, z, k1) / k1 **2

@njit
def i_i3(kp,r,z,k1, iRZ = False):
    kz1= np.sqrt(k1**2 - kp**2 + 0j )
    if iRZ == False:
        return r_p(kp, k1) *     j0_pp( kp * r )*  w(kp, z, k1) *  kz1 ** 2 / k1**2
    if iRZ == True:
        return r_p(kp, k1) * I_r_j0_pp( kp , r )*  w(kp, z, k1) *  kz1 ** 2 / k1**2

@njit    
def i_i4(kp,r,z,k1, iRZ = False):
    kz1= np.sqrt(k1**2 - kp**2 + 0j )
    if iRZ == False:
        return 1j * r_p(kp, k1) * kp * kz1 / k1**2  *     j0_p(kp*r)*  w(kp, z, k1)
    if iRZ == True:
        return 1j * r_p(kp, k1) * kp * kz1 / k1**2  * I_r_j0_p(kp,r)*  w(kp, z, k1)

@njit
def i_i5(kp,r,z,k1, iRZ = False):
    if iRZ == False:
        return r_s(kp, k1) *     j0_pp(kp * r) *  w(kp, z, k1)
    if iRZ == True:
        return r_s(kp, k1) * I_r_j0_pp(kp , r) *  w(kp, z, k1)

@njit
def i_i6(kp,r,z,k1, iRZ = False):
    if iRZ == False:
        return r_s(kp, k1) *   j0_p(kp * r)/(kp * r) * w(kp, z, k1)
    if iRZ == True:
        return r_s(kp, k1) * I_j0_p(kp , r)/(kp    ) * w(kp, z, k1)




def test_1(kp, r, z, k1, gamma = 0):
    return np.exp(- z * kp) * j0(r * np.sqrt(kp**2 + 2 * gamma * kp))
def test_1_exact(r,z, gamma):
    return 1 / np.sqrt(r**2 + z**2)* np.exp(gamma * (z - np.sqrt(r**2 + z**2)))
def test_2(kp,r,z,k1, v = 5):
    return np.exp(-z * kp) * jv(v, r * kp)
def test_2_exact(r,z,v):
    return r ** (-v) * (np.sqrt(r**2 + z**2) -z)**v/((r**2+ z**2)**0.5)
def test_3(kp, r, z, k1):
    return np.exp(-2 * z * kp ) * j0(kp) * y0(kp)
def test_3_exact(z):
    return ellipk((z * ( z**2 + 1) ** (-0.5))**2)/ ( np.pi * (z**2 + 1)**0.5 )




def cart2pol(r):
    x = r[:,0]
    y = r[:,1]
    z = r[:,2]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.array((rho, phi, z)).T

def pol2cart(c):
    rho = c[:,0]
    phi = c[:,1]
    z   = c[:,2]
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array((x, y, z)).T
def rho_vec(c):
    return np.array([ np.cos(c[:,1]), np.sin(c[:,1]), np.zeros(len(c))])

def phi_vec(c):
    return np.array([-np.sin(c[:,1]), np.cos(c[:,1]), np.zeros(len(c))])

def s_linsp(xmin, xmax, n, fr = 1/2):
    def f(x):
        return (1-fr) * x**2 + x*fr
    def df(x):
        return 2 * (1-fr) * x  + fr
    
    return f(np.linspace(xmin/xmax, 1, n)) * xmax



class Layered_gf_interpolator:
    def __init__(self, r_max, z_max, nr, nz, wavenumber,sing_box, z_offset_indirect = 0):
        self.r_max = r_max
        self. z_max = z_max
        self.nr = nr#int(r_max/dr)+1
        self.nz = nz#int(r_max/dz)+1
        self.z_offset_indirect = z_offset_indirect
        
        self.grid_r, self.grid_z = np.meshgrid(s_linsp(0.0,r_max,self.nr ), s_linsp(0.0, z_max,self.nz), indexing = 'ij')
        
        self.k0 = np.round(wavenumber,7)
        self.sing_box = sing_box
        self.vol = np.pi * self.sing_box[0] ** 2 * self.sing_box[1]
    
    def Template(self, f, Out, ns = 40):
        for i in tqdm(range(self.nr)):
            for j in range(self.nz):
                r_ij = self.grid_r[i,j]
                z_ij = self.grid_z[i,j]
                if i==0 and j == 0:
                    Out[i,j] = 2 * np.pi * f(self.sing_box[0], z_ij, iRZ = True)
                elif i==0 and j!=0:
                    Out[i,j] = 2 * np.pi * f(self.sing_box[0], z_ij, iRZ = True)
                else:
                    Out[i,j] = f(r_ij, z_ij)
        return Out
    
   
    def Gi_zz(self, n_pi):
        def f(r, z, iRZ = False):
            return 1j/(4 * np.pi) * Integral(lambda x1,x2,x3,x4 : i_i1(x1,x2,x3,x4, iRZ = iRZ), 
                                             self.k0, r, z + self.z_offset_indirect, n_pi = n_pi,
                                             force_fallback = iRZ)
        
        self.GI_ZZ =  np.zeros((self.nr, self.nz), dtype=complex)
        self.Template(f, self.GI_ZZ)
        self.interp_GI_ZZ = interp(self.grid_r.ravel(),
                                   self.grid_z.ravel(), 
                                   self.GI_ZZ.ravel())
    def Gi_pp(self, n_pi):
        def f(r, z, iRZ = False):
            return 1j / (4 * np.pi) * ( Integral(lambda x1,x2,x3,x4: i_i2(x1,x2,x3,x4, iRZ = iRZ), 
                                                 self.k0, r, z + self.z_offset_indirect, n_pi = n_pi,
                                                 force_fallback = iRZ) - 
                                        Integral(lambda x1,x2,x3,x4: i_i5(x1,x2,x3,x4, iRZ = iRZ), 
                                                 self.k0, r, z + self.z_offset_indirect, n_pi = n_pi,
                                                 force_fallback = iRZ)
                                      )
        
        self.GI_PP = np.zeros((self.nr, self.nz), dtype=complex)    
        self.Template(f, self.GI_PP)
        self.interp_GI_PP = interp(self.grid_r.ravel(),
                                   self.grid_z.ravel(), 
                                   self.GI_PP.ravel())
    def Gi_rr(self, n_pi):
        def f(r, z, iRZ = False):
            return 1j/ (4 * np.pi )* ( Integral(lambda x1,x2,x3,x4: i_i3(x1,x2,x3,x4, iRZ = iRZ), 
                                                self.k0, r, z + self.z_offset_indirect, n_pi = n_pi,
                                                force_fallback = iRZ) - 
                                       Integral(lambda x1,x2,x3,x4: i_i6(x1,x2,x3,x4, iRZ = iRZ), 
                                                self.k0, r, z + self.z_offset_indirect, n_pi = n_pi,
                                                force_fallback = iRZ) 
                                     )
        
        self.GI_RR = np.zeros((self.nr, self.nz), dtype=complex)
        self.Template(f, self.GI_RR)
        
        self.interp_GI_RR = interp(self.grid_r.ravel(),
                                   self.grid_z.ravel(), 
                                   self.GI_RR.ravel())
    def Gi_rz(self, n_pi):
        def f(r, z, iRZ = False):
            return 1j / (4 * np.pi) * Integral(lambda x1,x2,x3,x4: i_i4(x1,x2,x3,x4,iRZ = iRZ), 
                                               self.k0, r, z + self.z_offset_indirect, n_pi = n_pi,
                                               force_fallback = iRZ)
        
        self.GI_RZ  = np.zeros((self.nr, self.nz), dtype=complex)
        self.Template(f, self.GI_RZ)
        self.GI_ZR = - self.GI_RZ
        self.interp_GI_ZR = interp(self.grid_r.ravel(),
                                   self.grid_z.ravel(), 
                                   self.GI_ZR.ravel())
        self.interp_GI_RZ = interp(self.grid_r.ravel(),
                                   self.grid_z.ravel(), 
                                   self.GI_RZ.ravel())
    
    
    def Calc_all(self, n_pi = 80):
        self.Gi_rr(n_pi)
        self.Gi_pp(n_pi)
        self.Gi_zz(n_pi)
        self.Gi_rz(n_pi)
    
    def Indirect_Greens(self, p):
        cyl   = cart2pol(p)
        phi = cyl[:,1]
        n = len(p)
        cos = np.cos(phi)
        sin = np.sin(phi)
        GI = np.zeros((n,3,3), dtype = complex)
        
        gi_rr = self.interp_GI_RR.interpolate(cyl[:,[0,2]])
        gi_pp = self.interp_GI_PP.interpolate(cyl[:,[0,2]])
        gi_zz = self.interp_GI_ZZ.interpolate(cyl[:,[0,2]])
        gi_rz = self.interp_GI_RZ.interpolate(cyl[:,[0,2]])
        GI[:,0,0] = sin**2 * gi_pp[:] + cos**2 * gi_rr
        GI[:,1,1] = cos**2 * gi_pp[:] + sin**2 * gi_rr
        GI[:,2,2] = 1 * gi_zz[:]
        GI[:,0,1] = sin * cos * (gi_rr - gi_pp)
        GI[:,1,0] = GI[:,0,1]
        GI[:,0,2] = cos * gi_rz
        GI[:,1,2] = sin * gi_rz
        GI[:,2,0] = cos * (-1 * gi_rz)
        GI[:,2,1] = sin * (-1 * gi_rz) 
        return GI
    def save(self):
        np.save('ML_greens_functions/GI_RR'+str(self.k0), self.GI_RR)
        np.save('ML_greens_functions/GI_PP'+str(self.k0), self.GI_PP)
        np.save('ML_greens_functions/GI_ZZ'+str(self.k0), self.GI_ZZ)
        np.save('ML_greens_functions/GI_RZ'+str(self.k0), self.GI_RZ)
    def load(self,k0):
        self.GI_RR  = np.load('ML_greens_functions/GI_RR'+str(k0)+'.npy')
        self.GI_PP  = np.load('ML_greens_functions/GI_PP'+str(k0)+'.npy')
        self.GI_ZZ  = np.load('ML_greens_functions/GI_ZZ'+str(k0)+'.npy')
        self.GI_RZ  = np.save('ML_greens_functions/GI_RZ'+str(k0)+'.npy')
        
        
        
        

    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    # def _Gd_rr(self, n_pi = 60):
    #     Gd_rr = np.zeros((self.nr, self.nz), dtype = complex)
    #     def f(r, z):
    #         return 1j/(4*np.pi) * (Integral(d_i1, self.k0, r, z, n_pi = n_pi) + 
    #                                Integral(d_i2, self.k0, r, z, n_pi = n_pi)  )
    #     for i  in range(self.nr):
    #         for j in range(self.nz):
    #             r_ij = self.grid_r[i,j]; z_ij = self.grid_z[i,j]
    #             if i==0 and j == 0:
    #                 for r in np.linspace(self.dr/10, self.dr,  10):
    #                     for z in np.linspace(self.dz/10, self.dz, 10):
    #                         Gd_rr[i,j]+= 2 * np.pi * f(r,z)*r * (self.dr/10) * (self.dz/10)/self.vol
    #             elif i==0:
    #                 Gd_rr[i,j] = f(1e-3, z_ij)
    #             elif j==0:
    #                 Gd_rr[i,j] = f(r_ij,  1e-3)
    #             else:
    #                 Gd_rr[i,j] = f(r_ij, z_ij)
    #     self. GD_RR = Gd_rr
    # def _Gd_pp(self, n_pi = 60):
    #     Gd_pp = np.zeros((self.nr, self.nz), dtype = complex)
    #     def f(r, z):
    #         return 1j/(4*np.pi) * (Integral(d_i1, self.k0, r, z, n_pi = n_pi) + 
    #                                Integral(d_i3, self.k0, r, z, n_pi = n_pi)  )
    #     for i  in range(self.nr):
    #         for j in range(self.nz):
    #             r_ij = self.grid_r[i,j]; z_ij = self.grid_z[i,j]
    #             if i==0 and j == 0:
    #                 for r in np.linspace(self.dr/10, self.dr,  10):
    #                     for z in np.linspace(self.dz/10, self.dz, 10):
    #                         Gd_pp[i,j]+= 2 * np.pi * f(r,z)*r * (self.dr/10) * (self.dz/10)/self.vol
    #             elif i==0:
    #                 Gd_pp[i,j] = f(1e-3, z_ij)
    #             elif j==0:
    #                 Gd_pp[i,j] = f(r_ij,  1e-3)
    #             else:
    #                 Gd_pp[i,j] = f(r_ij, z_ij)
    #     self. GD_PP = Gd_pp
    
    # def _Gd_zz(self, n_pi = 60):
    #     Gd_zz = np.zeros((self.nr, self.nz), dtype = complex)
    #     def f(r, z):
    #         return 1j/(4*np.pi) * (Integral(d_i4, self.k0, r, z, n_pi = n_pi)  )
    #     for i  in range(self.nr):
    #         for j in range(self.nz):
    #             r_ij = self.grid_r[i,j]; z_ij = self.grid_z[i,j]
    #             if i==0 and j == 0:
    #                 for r in np.linspace(self.dr/10, self.dr,  10):
    #                     for z in np.linspace(self.dz/10, self.dz, 10):
    #                         Gd_zz[i,j]+= 2 * np.pi * f(r,z)*r * (self.dr/10) * (self.dz/10)/self.vol
    #             elif i==0:
    #                 Gd_zz[i,j] = f(1e-3, z_ij)
    #             elif j==0:
    #                 Gd_zz[i,j] = f(r_ij,  1e-3)
    #             else:
    #                 Gd_zz[i,j] = f(r_ij, z_ij)
    #     self. GD_ZZ = Gd_zz
    
    # def _Gd_rz(self, n_pi = 60):
    #     Gd_rz = np.zeros((self.nr, self.nz), dtype = complex)
    #     def f(r, z):
    #         return 1j/(4*np.pi) * (Integral(d_i5, self.k0, r, z, n_pi = n_pi)  )
    #     for i  in range(self.nr):
    #         for j in range(self.nz):
    #             r_ij = self.grid_r[i,j]; z_ij = self.grid_z[i,j]
    #             if i==0 and j == 0:
    #                 for r in np.linspace(self.dr/10, self.dr,  10):
    #                     for z in np.linspace(self.dz/10, self.dz, 10):
    #                         Gd_rz[i,j]+= 2 * np.pi * f(r,z)*r * (self.dr/10) * (self.dz/10)/self.vol
    #             elif i==0:
    #                 Gd_rz[i,j] = f(1e-3, z_ij)
    #             elif j==0:
    #                 Gd_rz[i,j] = f(r_ij,  1e-3)
    #             else:
    #                 Gd_rz[i,j] = f(r_ij, z_ij)
    #     self. GD_RZ = Gd_rz
    
    
    
    
    
# Gf = Layered_gf_interpolator(10, 10,1,1, wavenumber = 2 * np.pi / 2000, sing_box = [1,1], z_offset_indirect = 2.5 * 2)

# for i in range(10):
#     Integral(i_i2, Gf.k0, 350.960954385274774, 1.32891195693938, n_pi = 10)


# def Gd_T1(rho, dz, k1, exp_tol = 1e-7, int_tol = 1e-5):
#     lim_int = np.sqrt(k1**2 + (-np.log(exp_tol)/(dz+1e-10))**2)
#     error = 1
#     it = 2
#     while error > int_tol:
#         split = np.hstack((np.linspace(0,k1, it), np.linspace((k1 + (k1 + lim_int)/2),lim_int,it)))
#         #return split
        
#         errors = []
#         RES_r = []
#         RES_i = []
#         for i in range(2*it-1):
#             w1 = split[i]
#             w2 = split[i+1]
#             res_r = quad(lambda x: I1(x, rho, dz, k1).real, w1*lim_int, w2*lim_int )
#             res_i = quad(lambda x: I1(x, rho, dz, k1).imag, w1*lim_int, w2*lim_int )
#             errors += [max(res_r[1], res_i[1])]
#             res_r = res_r[0]
#             res_i = res_i[0]
#             RES_r+=[res_r]
#             RES_i+=[res_i]
        
#         error = sum(errors)/(it-1)
#         it+=1
    
#     res = (sum(RES_r) + 1j * sum(RES_i), error)
#     return res
