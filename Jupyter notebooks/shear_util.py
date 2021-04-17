import os
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import minimize
import math
from galsim import _galsim
import pandas as pd
from scipy.interpolate import interp1d
# from mpmath import gammainc

absdir = os.path.dirname(os.path.abspath(__file__))

class HLRShearModel:
    """ fully vectorized - blazing fast
    ... > hsm = HLRShearModel('hlrshear.csv')
    ... > hlr = hsm.get_hlr_preshear(p.gal_hlr, e, magnify=True)
    magnify=True corresponds to galsim.lens() 
    magnify=False corresponds to galsim.shear() 
    """
    def __init__(self,training_file=f'{absdir}/hlrshear.csv',magnify=None):
        # training_file should be csv
        # it has the sheared hlr of initially round shape with hlr=1.0 given ellipticity
        # ------------
        # passing magnify=False makes the returned hlrs appropriate inputs for galsim.shear()
        # for the rest of the run using hsm but you can pass it later too (recommended)
        self.training_file = training_file
        self.magnify = magnify
        self.df_train = pd.read_csv(self.training_file)
        
    def train(self,ref,magnify):
        # ref: 'analytic', 'galsim'
        self.magnify = magnify
        colname = f'scale_{ref}'
        if self.magnify:
            colname += '_magnified'
        self.finterp = interp1d(self.df_train['e'], self.df_train[colname], kind='cubic', bounds_error=False, fill_value='extrapolate')
        
    def get_hlr_preshear(self,hlr_postshear,e,magnify=None):
        ''' get the hlr of the "round" preshear shape from the sheared shape '''
        if hasattr(self, 'finterp') and self.magnify==magnify:
            pass
        else:
            if magnify is None:
                raise RuntimeError('Pass True or False for magnify')
            self.train('analytic',magnify)
        e, hlr_postshear = np.asarray(e), np.asarray(hlr_postshear)
        scale = self.finterp(e)
        hlr_preshear = hlr_postshear / scale
        return hlr_preshear 
    
    def get_hlr_postshear(self,hlr_preshear,e,magnify=None):        
        ''' get the hlr of the sheared shape from the "round" preshear shape '''
        if hasattr(self, 'finterp') and self.magnify==magnify:
            pass
        else:
            if magnify is None:
                raise RuntimeError('Pass True or False for magnify')
            self.train('analytic',magnify)
        e, hlr_preshear = np.asarray(e), np.asarray(hlr_preshear)
        scale = self.finterp(e)
        hlr_postshear = hlr_preshear * scale
        return hlr_postshear

# [for testing] taken from https://github.com/LSSTDESC/WeakLensingDeblending/blob/master/descwl/model.py#L48
def moments_size_and_shape(Q):
    """Calculate size and shape parameters from a second-moment tensor.
    If the input is an array of second-moment tensors, the calculation is vectorized
    and returns a tuple of output arrays with the same leading dimensions (...).
    Args:
        Q(numpy.ndarray): Array of shape (...,2,2) containing second-moment tensors,
            which are assumed to be symmetric (only the [0,1] component is used).
    Returns:
        tuple: Tuple (sigma_m,sigma_p,a,b,beta,e1,e2) of :class:`numpy.ndarray` objects
            with shape (...). Refer to :ref:`analysis-results` for details on how
            each of these vectors is defined.
    """
    trQ = np.trace(Q,axis1=-2,axis2=-1)
    detQ = np.linalg.det(Q)
    sigma_m = np.power(detQ,0.25)
    sigma_p = np.sqrt(0.5*trQ)
    asymQx = Q[...,0,0] - Q[...,1,1]
    asymQy = 2*Q[...,0,1]
    asymQ = np.sqrt(asymQx**2 + asymQy**2)
    a = np.sqrt(0.5*(trQ + asymQ))
    b = np.sqrt(0.5*(trQ - asymQ))
    beta = 0.5*np.arctan2(asymQy,asymQx)
    e_denom = trQ + 2*np.sqrt(detQ)
    e1 = asymQx/e_denom
    e2 = asymQy/e_denom
    return sigma_m,sigma_p,a,b,beta,e1,e2 

# modified from https://github.com/LSSTDESC/WeakLensingDeblending/blob/master/descwl/model.py#L14
def sersic_second_moments(hlr,e1=None,e2=None,n=0.5,q=None,beta=None,magnify=True,out_unit='arcsec'): # just to generate training data for the shear model
    """Calculate the second-moment tensor of a "sheared" Sersic radial profile.
    >> EN: you should either pass (e1,e2) or (q,beta) | e1,e2 should be epsilon-ellipticity
    Args:
        n(int): Sersic index of radial profile. Only n = 1 and n = 4 are supported.
        hlr(float): Radius of 50% isophote before shearing, in arcseconds.
        q(float): Ratio b/a of Sersic isophotes after shearing.
        beta(float): Position angle of sheared isophotes in radians, measured anti-clockwise
            from the positive x-axis.
    Returns:
        numpy.ndarray: Array of shape (2,2) with values of the second-moments tensor
            matrix, in units of square arcseconds.
    Raises:
        RuntimeError: Invalid Sersic index n.
    """
        
    if n == 0.5:
        # for gaussians the calculation is analytic
        r02hlr =  1/np.sqrt(np.log(2))
    else:
        r02hlr = 1/_galsim.SersicHLR(n,1.0) # 1.0 implies flux_untruncated==True
        
    cn = 0.5*(r02hlr)**2 * math.gamma(4*n)/math.gamma(2*n)

    if e1 is None and e2 is None:
        e_mag = (1.-q)/(1.+q)
        e_mag_sq = e_mag**2
        e1 = e_mag*np.cos(2*beta)
        e2 = e_mag*np.sin(2*beta)
    else:
        e_mag_sq=e1**2+e2**2
    
    Q11 = 1 + e_mag_sq + 2*e1
    Q22 = 1 + e_mag_sq - 2*e1
    Q12 = 2*e2
    
    Q = np.array(((Q11,Q12),(Q12,Q22)))*cn*hlr**2
    
    if magnify:
        # mu = 1/(1-e_mag_sq)
        Q /= (1-e_mag_sq)**2
    else:
        Q /= (1-e_mag_sq)
    
    if out_unit.startswith('deg'):
        Q /= 3600**2 # now in degrees
    elif out_unit == 'arcsec':
        pass
    else:
        raise RuntimeError('Invalid `out_unit`')

    return Q
    
def gaussian_second_moments(galhlr,e1,e2,magnify=True,out_unit='arcsec'): # just to generate training data for the shear model
    """
    Returns the covariance matrix of the lensing shears given the two components
    Author: Erfan Nourbakhsh

    [assuming N galaxies in the blended system]

    e1 : the array of the first component of the shape for N galaxies, epsilon ellipticity
    e2 : the array of the second component of the shape for N galaxies, epsilon ellipticity

    with magnify=True it agrees with WeakLensingDeblending's sersic_second_moments(n,hlr,q,beta)
    https://github.com/LSSTDESC/WeakLensingDeblending/blob/master/descwl/model.py#L14
    """
    
    # absolute magnitude of ellipticity
    e = np.sqrt(e1**2+e2**2)
    
    # galhlr: FLUX_RADIUS in arcsec -- half-light radius of the round version of the "gaussian" profile before shearing
    sigma_round = galhlr/np.sqrt(2.*np.log(2)) # in arcsec
    
    if out_unit.startswith('deg'):
        sigma_round /= 3600 # now in degrees
    elif out_unit == 'arcsec':
        pass
    else:
        raise RuntimeError('Invalid `out_unit`')

    # a and b are deviations from a circle of radius r=sigma_round
    if magnify: # equivalent of galsim.lens()
        # https://galsim-developers.github.io/GalSim/_build/html/gsobject.html#galsim.GSObject.lens
        a = sigma_round/(1-e)
        b = sigma_round/(1+e)
    else: # equivalent of galsim.shear() -- area or "total" flux preseved
        # https://galsim-developers.github.io/GalSim/_build/html/gsobject.html#galsim.GSObject.shear
        a = sigma_round * np.sqrt((1+e)/(1-e))
        b = sigma_round * np.sqrt((1-e)/(1+e))

    theta = 0.5*np.arctan2(e2,e1) # in radians

    if np.isscalar(theta):
        R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        Sigma_0 = np.array([[a**2,0],[0,b**2]])
        Sigma   = np.dot(R,np.dot(Sigma_0,R.T))
    else:
        N = theta.size
        R = [np.array([[np.cos(theta[k]),-np.sin(theta[k])],[np.sin(theta[k]),np.cos(theta[k])]]) for k in range(N)]
        Sigma_0 = [np.array([[a[k]**2,0],[0,b[k]**2]]) for k in range(N)]
        Sigma   = [np.dot(R[k],np.dot(Sigma_0[k],R[k].T)) for k in range(N)]
    return np.array(Sigma)

def I(x,a): # watch the order! # just to generate training data for the shear model
    # a>0, x>=0 only with scipy
    gamma_a, gammainc_a_x = scipy.special.gamma(a), (scipy.special.gammainc(a, x))
    # ans = gamma_a*gammainc_a_x # has problem at large a>170 but it's vectorized
    if not isinstance(gammainc_a_x, (tuple,list,np.ndarray)):
        if np.isposinf(gamma_a):
            ans=0 # this is just to prevent nan outputs, not an actual solution; TODO
        else:
            ans=gamma_a*gammainc_a_x
    else:
        ispinf = np.isposinf(gamma_a)
        ans[ispinf]=0
        ans[~ispinf]=gamma_a[~ispinf]*gammainc_a_x[~ispinf]
    return ans
    # return np.float(gammainc(a,a=0,b=x,regularized=False)) # same but not vectoized

def get_shape_covmat(hlr_postshear,e1,e2,out_unit='arcsec'):
    # assumes hlr_postshear in arcsec
    magnify = True # magnify=True or False doesn't change the outcome it just needs to be consistent throughout this function
    e = np.sqrt(e1**2+e2**2)
    hlr_preshear = get_hlr_preshear(hlr_postshear, e, magnify=magnify)
    Q = sersic_second_moments(hlr_preshear,e1,e2,magnify=magnify,out_unit=out_unit) # both gaussian_second_moments and sersic_second_moment with n=0.5 work
    return Q

def get_shape_covmat_fast(hlr_postshear,e1,e2,hsm=HLRShearModel(),out_unit='arcsec'):
    ''' returns the actual second moments tensor that describes the shape given the
    current ellipticity and hlr. It uses interpolation on pre-computd data to increase
    efficiency while vectorizing the function.
    Note: if you invoke this function multiple times and don't want to train the model every time,
          you can execute hsm=HLRShearModel() once and pass the hsm class object to this function.
    '''
    # assumes hlr_postshear in arcsec
    magnify = False # magnify=True or False doesn't change the outcome it just needs to be consistent throughout this function
    e = np.sqrt(e1**2+e2**2)
    hlr_preshear = hsm.get_hlr_preshear(hlr_postshear, e, magnify=magnify)
    Q = gaussian_second_moments(hlr_preshear,e1,e2,magnify=magnify,out_unit=out_unit) # both gaussian_second_moments and sersic_second_moment with n=0.5 work
    return Q

def integral_bivariate_gaussian(r, sigma_x, sigma_y, rtol=1e-10, miniter=3, maxiter=150): # just to generate training data for the shear model
    # Fully vectorized!
    # high maxiter leads to div by zero runtime warning, keep it low like at 50
    # Using eq. (5) of https://www.jstor.org/stable/pdf/1266181.pdf
    # Used in the Circular Error Probable (CEP) problem
    # miniter and maxiter override EPS
    a = (sigma_x**2+sigma_y**2)/(2*sigma_x*sigma_y)**2
    b = (sigma_y**2-sigma_x**2)/(2*sigma_x*sigma_y)**2
    A = 1/(2*a*sigma_x*sigma_y)
    sum_0 = 2*rtol
    n, sum_ = 0, 0
    while n<miniter or np.all(np.abs(sum_-sum_0)/np.abs(sum_0))>rtol: # vectorized with np.all(...)!
        sum_0 = sum_
        sum_ += (b/(2*a))**(2*n) * (1/np.math.factorial(n))**2 * I(a*r**2, 2*n+1)
        n += 1
        if n>=maxiter:
            break
    Fr = A*sum_
    # (F(r)], r=0...R is gives the probability of a random point falling within the distant R from the center
    return Fr

# https://scicomp.stackexchange.com/questions/15907/why-does-matlabs-integral-outperform-integrate-quad-in-scipy
# from numba import complex128,float64,jit
# @jit(flot64(float64, float64), nopython=True, cache=True)
def bivariate_gaussian(r, sigma_x, sigma_y): # just to generate training data for the shear model
    # we can conveniently assume \rho=0 (uncorrelated) w/o loss of generality
    # Using eq. (4) of https://www.jstor.org/stable/pdf/1266181.pdf
    a = (sigma_x**2+sigma_y**2)/(2*sigma_x*sigma_y)**2
    b = (sigma_y**2-sigma_x**2)/(2*sigma_x*sigma_y)**2
    return (r/(sigma_x*sigma_y)) * np.exp(-a*r**2)*scipy.special.iv(0,b*r**2)

# def bivariate_gaussian_derivative(r, sigma_x, sigma_y):
#     # we can conveniently assume \rho=0 (uncorrelated) w/o loss of generality
#     # Taking the derivative of eq. (4) of https://www.jstor.org/stable/pdf/1266181.pdf
#     a = (sigma_x**2+sigma_y**2)/(2*sigma_x*sigma_y)**2
#     b = (sigma_y**2-sigma_x**2)/(2*sigma_x*sigma_y)**2
#     return np.exp(-a*r**2) * ( (1-2*a*r**2)*scipy.special.iv(0,b*r**2) + 2*b*r**2*scipy.special.iv(1,b*r**2) ) / (sigma_x*sigma_y)

def prob_bivariate_gaussian(r1, r2, sigma_x, sigma_y, method='quad'): # just to generate training data for the shear model
    # quad is not fully vectorized; use the infinite series
    if method=='quad': # not vectorized
        prob = quad(bivariate_gaussian, r1, r2, args=(sigma_x, sigma_y), epsrel=1e-10)[0]
        #prob = prob_[0]
        #print(f'int and its error in quad: {prob_}')
    elif method=='infinite-series': # vectorized
        raise RuntimeError(f"The {method} method doesn't work well for high ellipticities (~>0.7) due to some\
                             numerical issues with the gammainc, i.e. I(x,a) function at high `a`. This should be\
                             fixed before you are able to use it. Use the `quad` method which works well.")
        if r1==0:
            prob = integral_bivariate_gaussian(r2, sigma_x, sigma_y) - 0 # just to save some computation time
        else:
            prob = integral_bivariate_gaussian(r2, sigma_x, sigma_y) - integral_bivariate_gaussian(r1, sigma_x, sigma_y)
    else:
        raise RuntimeError('Illegal `method`')
    return prob

def hlr_diff(hlr_preshear,hlr_postshear,e,magnify=False): # just to generate training data for the shear model
    # circular gaussian sigma before shearing
    sigma_round = hlr_preshear/np.sqrt(np.log(4))
    
    # after shearing, we will have sigma_x and sigma_y which are switchable without changing the results
    if magnify:
        # equivalent of galsim.lens() = galsim.shear() + galsim.magnify() which doesn't preserve the total flux
        # this is the what happens from the physical standpoint due to lensing
        sigma_x = sigma_round/(1-e)
        sigma_y = sigma_round/(1+e)
    else:
        # equivalent of galsim.shear() which preserves the "total" flux where sigma_x * sigma_y == sigma_circular**2
        # magnification: mu = 1/(1-e**2)
        # simply multiply the magnified sigma_x/y by mu**(-1/2)
        sigma_x = sigma_round * np.sqrt((1+e)/(1-e))
        sigma_y = sigma_round * np.sqrt((1-e)/(1+e))        
    enclosed_flux_frac = prob_bivariate_gaussian(0, hlr_postshear, sigma_x, sigma_y)
    return enclosed_flux_frac-0.5 # 0.5 for hlr!

def hlr_diff_derivative(hlr_preshear,hlr_postshear,e,magnify=False): # just to generate training data for the shear model
    # circular gaussian sigma before shearing
    sigma_circular = hlr_preshear/np.sqrt(np.log(4))
    
    # after shearing, we will have sigma_x and sigma_y which are switchable without changing the results
    if magnify:
        # equivalent of galsim.lens() = galsim.shear() + galsim.magnify() which doesn't preserve the total flux
        # this is the what happens from the physical standpoint due to lensing
        sigma_x = sigma_circular/(1-e)
        sigma_y = sigma_circular/(1+e)
    else:
        # equivalent of galsim.shear() which preserves the "total" flux where sigma_x * sigma_y == sigma_circular**2
        # magnification: mu = 1/(1-e**2)
        # simply multiply the magnified sigma_x/y by mu**(-1/2)
        sigma_x = sigma_circular * np.sqrt((1+e)/(1-e))
        sigma_y = sigma_circular * np.sqrt((1-e)/(1+e))        
    return bivariate_gaussian(hlr_postshear, sigma_x, sigma_y)

def hlr_absdiff(hlr_preshear,hlr_postshear,e,magnify=False): # just to generate training data for the shear model
    return np.abs(hlr_diff(hlr_preshear,hlr_postshear,e,magnify=magnify))

def hlr_absdiff_with_sigmas(hlr_in,sigma_x,sigma_y): # just to generate training data for the shear model
    enclosed_flux_frac = prob_bivariate_gaussian(0, hlr_in, sigma_x, sigma_y)
    return np.abs(enclosed_flux_frac-0.5) # 0.5 for hlr!

def get_hlr_preshear(hlr_postshear, e, magnify=False, method='Nelder-Mead'): # just to generate training data for the shear model; later you should use the fast approach of hsm.get_hlr_preshear()
    ''' NOTE: hlr_postshear can't be an array because this function is not vectorized; use np.vectorize() or list comprehension if desired '''
    # A very good approximation for the initial guess using the Wilson-Hilferty transformation
    # eq. 18 of https://apps.dtic.mil/dtic/tr/fulltext/u2/1043284.pdf
    # hlr_wh = np.sqrt( (sigma_x**2+sigma_y**2)*(1-2*(sigma_x**4+sigma_y**4)/9/(sigma_x**2+sigma_y**2)**2)**3 )
    # let's use a less accurate approximation that only takes e: hlr/(1-e**2) w/ magnification; hlr/(1-e**2)**0.5 w/o magnification
    hlr_guess = hlr_postshear*(1-e**2) if magnify else hlr_postshear*(1-e**2)**0.5 # magnification: mu = 1/(1-e**2)
    if method=='Newton':
        raise NotImplementedError('get_hlr_preshear() does not support the `newton` method yet. TODO for completeness. Please use the `Nelder-Mead` method.')
        res = scipy.optimize.newton(lambda x: hlr_diff(x,hlr_postshear,e,magnify), hlr_guess,
                                    fprime=lambda x: hlr_diff_derivative(x,hlr_postshear,e,magnify))
        return res
    elif method=='Nelder-Mead': # for some reason only this works for this function not newton
        res = minimize(hlr_absdiff, hlr_guess, args=(hlr_postshear, e, magnify), method='Nelder-Mead', tol=1e-18)
        return res.x[0]
    else:
        raise RuntimeError('Invalid method')

def get_hlr_preshear_fast(hlr_postshear, e, hsm=HLRShearModel(), magnify=False):
    ''' vectorized and efficient based on pre-computed interpolated data '''
    return hsm.get_hlr_preshear(hlr_postshear, e, magnify)

def get_hlr_postshear(hlr_preshear, e, magnify=False, method='Newton'): # just to generate training data for the shear model; later you should use the fast approach of hsm.get_hlr_postshear()
    ''' NOTE: hlr_preshear can't be an array because this function is not vectorized; use np.vectorize() or list comprehension if desired '''
    # A very good approximation for the initial guess using the Wilson-Hilferty transformation
    # eq. 18 of https://apps.dtic.mil/dtic/tr/fulltext/u2/1043284.pdf
    # hlr_wh = np.sqrt( (sigma_x**2+sigma_y**2)*(1-2*(sigma_x**4+sigma_y**4)/9/(sigma_x**2+sigma_y**2)**2)**3 )
    # let's use a less accurate approximation that only takes e: hlr/(1-e**2) w/ magnification; hlr/(1-e**2)**0.5 w/o magnification
    hlr_guess = hlr_preshear/(1-e**2) if magnify else hlr_preshear/(1-e**2)**0.5 # magnification: mu = 1/(1-e**2)
    if method=='Newton':
        res = scipy.optimize.newton(lambda x: hlr_diff(hlr_preshear,x,e,magnify), hlr_guess,
                                    fprime=lambda x: hlr_diff_derivative(hlr_preshear,x,e,magnify))
        return res
    elif method=='Nelder-Mead':
        res = minimize(hlr_absdiff, hlr_guess, args=(hlr_preshear, e, magnify), method='Nelder-Mead', tol=1e-18)
        return res.x[0]
    else:
        raise RuntimeError('Invalid method')

def get_hlr_postshear_fast(hlr_preshear, e, hsm=HLRShearModel(), magnify=False):
    ''' vectorized and efficient based on pre-computed interpolated data '''
    return hsm.get_hlr_postshear(hlr_preshear, e, magnify)

def calc_moment_radius(Q, method='det'):
    # assumes symmetric Q and doesn't check for it
    if method=='det':
        sigma_round = np.linalg.det(Q)**0.25
    elif method=='trace':
        trace = np.trace(Q, axis1=-2, axis2=-1)
        sigma_round = np.sqrt(0.5*trace)
    else:
        raise RuntimeError('Invalid method')
    return sigma_round

def shape_from_moments(Q, return_emag=True):
    # vectorized! assumes epsilon-ellipticity
    # not necessarily for gaussian profiles, use for whatever profiles
    # assumes symmetric Q and doesn't check for it
    Q = np.asarray(Q)
    Q11, Q22, Q12 = Q[...,0,0], Q[...,1,1], Q[...,0,1]
    e1 = (Q11-Q22)/(Q11+Q22+2*(Q11*Q22-Q12**2)**0.5)
    e2 = 2.0*Q12/(Q11+Q22+2*(Q11*Q22-Q12**2)**0.5)
    e = np.sqrt(e1**2+e2**2)
    if return_emag:
        return e1, e2, e
    else:
        return e1, e2

def diagonalize(Q):
    """
    Q(..., M, M) array: Matrices for which the diagonalized version will be computed
    It diagonalizes a stack of matrices of any order (2 by 2 for our application)
    """
    # theoretically the eigenvectors (P) are linearly independent and a can be
    # diagonalized by a similarity transformation using v, i.e, inv(v) @ Q @ v is diagonal
    Q = np.asarray(Q)
    _, v = np.linalg.eig(Q)
    v_inv = np.linalg.inv(v)
    Qdiag = v_inv @ Q @ v # Python 3.5+
    # The Einstein summation convention works too!
    # X = np.einsum("...ij, ...jk -> ...ik", v_inv, Q)
    # Qdiag = np.einsum("...ij, ...jk -> ...ik", X, v)
    assert Q.shape == Qdiag.shape
    return Qdiag

def hlr_from_moments(Q):
    # for the vectorizrd version (recommended) use `hlr_from_moments_fast()`
    # not vectorized because of the minimize function (along with the quad funcion inside)
    # you can use scipy.optimize.newton (vectorized) instead of minimize but still the integral quad remains non-vectorized
    # assumes symmetric Q and doesn't check for it
    # IMPORTANT: the diagonals always give us the two variances even in the presence of non-zero off-diagonals BUT we
    # need the ones after the Q matrix is diagonalized here!!
    
    Qdiag = diagonalize(Q)
    sigma_x, sigma_y = np.sqrt(Qdiag[...,0,0]), np.sqrt(Qdiag[...,1,1])
    # A very good approximation for the initial guess using the Wilson-Hilferty transformation
    # eq. 18 of https://apps.dtic.mil/dtic/tr/fulltext/u2/1043284.pdf
    hlr_wh = np.sqrt( (sigma_x**2+sigma_y**2)*(1-2*(sigma_x**4+sigma_y**4)/9/(sigma_x**2+sigma_y**2)**2)**3 )
    res = minimize(hlr_absdiff_with_sigmas, hlr_wh, args=(sigma_x, sigma_y), method='Nelder-Mead', tol=1e-15)
    return res.x[0]

def hlr_from_moments_fast(Q, hsm=HLRShearModel(), return_shape=False):
    # vectorized!
    # It uses interpolation on pre-computed data with high precission
    # recommended over `hlr_from_moments()`
    # assumes gaussian profiles
    # assumes symmetric Q and doesn't check for it
    Q = np.asarray(Q)
    sigma_round = calc_moment_radius(Q)
    hlr_round = sigma_round*np.sqrt(np.log(4))
    e1, e2, e = shape_from_moments(Q)
    hlr_interp = hsm.get_hlr_postshear(hlr_round,e,magnify=False)
    if return_shape:
        return hlr_interp, e1, e2
    else:
        return hlr_interp

def convolve_with_PSF(e1,e2,hlr,PSF_FWHM=0,hsm=HLRShearModel(),mode='convolve',return_T2Tpsf=False,illegal_moments_flag=-11): # correct version!
    nobj = len(hlr)
    hlr_psf = PSF_FWHM/2 # only true for Gaussians
    Q = get_shape_covmat_fast(hlr,e1,e2,out_unit='arcsec')    # for the galaxy
    P = get_shape_covmat_fast(hlr_psf,0,0,out_unit='arcsec')  # for the PSF -- 0, 0 -> circular       
    if mode=='convolve':
        Q+=P
    elif mode=='deconvolve':
        Q-=P
    else:
        raise RuntimeError('Invalid mode')
    if return_T2Tpsf: # T of output (convolved/deconvolved) to T of PSF
        T2Tpsf = get_T_from_Q(Q)/get_T_from_Q(P)
    bm = np.linalg.det(Q)<0 # bad_moments
    nbm = sum(bm)
    if nbm>0: # it never happened for convolution
        print(f"Unable to run the code in the `{mode}` mode for {nbm} galaxies ({100*nbm/nobj:.2f}%) -- but don't worry we will flag their (e1, e2, hlr) as {illegal_moments_flag}")

    ok = ~bm
    hlr_new, e1_new, e2_new = np.zeros_like(hlr), np.zeros_like(hlr), np.zeros_like(hlr)
    hlr_new[bm] = e1_new[bm] = e2_new[bm] = illegal_moments_flag
    hlr_new[ok],  e1_new[ok],  e2_new[ok] = hlr_from_moments_fast(Q[ok],hsm=hsm,return_shape=True)
    
    if return_T2Tpsf:
        return e1_new, e2_new, hlr_new, T2Tpsf
    else:
        return e1_new, e2_new, hlr_new

def get_T_from_Q(Q):
    T = Q.trace(axis1=-2, axis2=-1)
    return T

def get_T(e1,e2,hlr):
    Q = get_shape_covmat_fast(hlr,e1,e2,out_unit='arcsec')
    T = get_T_from_Q(Q)
    return T
