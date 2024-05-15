import jax
import jaxlib
import jaxlib.xla_extension
from jax import grad
from iminuit import Minuit
import numpy as np  # original numpy still needed, since jax does not cover full API
from numpy import ndarray, float64
from collections.abc import Callable
import scipy as sc
import matplotlib.pyplot as plt
from scipy.special import factorial
# Add the following line to import jaxlib
import jaxlib

jax.config.update("jax_enable_x64", True)  # enable float64 precision, default is float32

def final_val(x,sigma,decimals = 2,exp = 0, udm: str = '') -> str:
    x = np.round(x*np.power(10.0,-exp),decimals)
    sigma = np.round(sigma*np.power(10.0,-exp),decimals)
    return f'{x} ± {sigma} {udm}' if exp == 0 else f'({x} ± {sigma})e{exp} {udm}'


class Interpolazione:
    def __init__(self,X: ndarray[float64], Y: ndarray[float64],
                 sigmaY: ndarray[float64] | float64,sigmaX: ndarray[float64] | float64,
                 model: Callable,guess: ndarray[float64],
                 names: list[str] = None, iteration = 5) -> None:
        self.f = model
        self.Y = Y.astype('float64')
        self.X = X.astype('float64')
        self.N = len(X)
        self.iteration = iteration
        self.sigmaY = sigmaY
        self.sigmaX = sigmaX
        self.names = names
                        
        @jax.jit
        def cost(par):
            result = 0.0
            for xi, yi, sy, sx in zip(self.X, self.Y, self.sigmaY, self.sigmaX):
                # y_var = self.__error(xi, par)
                y_var = sy ** 2
                
                # yvar:
                x_var = 0
                d = self.f
                for i in range(1,self.iteration+1):
                    d = jax.jit(grad(d)) # compute derivative
                    x_var += (sx**i * d(xi,*par)) / factorial(i) # 1/n| d^n/dx^n f(x) * dx^n
                
                z_var = y_var + x_var**2                
                result += (yi - self.f(xi, *par)) ** 2 / z_var
            return result
        
        
        cost.errordef = Minuit.LEAST_SQUARES

        self.m = Minuit(cost,guess,name=names)
        self.m.migrad()
        self.m.hesse()
        
        self.chi2 = np.round(self.m.fval,2)
        self.dof = len(X) - len(guess)
        self.rchi2 = np.round(self.chi2/self.dof)
        
        self.values = self.m.values.to_dict()
        self.errors = self.m.errors.to_dict()
        self.covariance = np.array(self.m.covariance)
        
    def draw(self,xscale='linear',N=1000):
        if xscale == 'log':
            x = np.logspace(np.log10(self.X.min()),np.log10(self.X.max()),N)
        else:
            x = np.linspace(self.X.min(),self.X.max(),N)
        
        return x, self.f(x,*self.values.values())
        
    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        # s1 = str(self.m) + '\n\n
        s0 = '----------------- VALORI FIT: -----------------\n'
        s2 = f"dof: {self.dof}\nchi2: {self.chi2}\nchi2 ridotto: {self.rchi2}"
        
        exponents = np.array(np.floor(np.log10(np.abs(list(self.values.values())))),dtype=np.int32)
                
        s3 = '\n'.join([n + ':' + final_val(v,s,3,e) for n,v,s,e in zip(self.names,self.values.values(),self.errors.values(),exponents)])
         
        s4 = '\n------------------------------------------------\n'
        return s0 + s3 + '\n\n' + s2 + s4
    


if __name__ == '__main__':
   
    y = np.array([2.25, 4.25, 6.5, 8.75, 10.75, 12.75])
    x = np.array([10.,20.,30.,40.,50.,60.])
    y_err = np.array([0.21, 0.21, 0.25, 0.21, 0.21, 0.21])
    x_err = (np.ones(len(x))).astype(float)

    def func1(x, a,b,c): # delta_N = ... * delta_p
        return a*x**2 + b*x + c

    f_prime = jax.jit(grad(func1))

    @jax.jit
    def cost(m):
        result = 0.0
        for xi, yi, y_erri in zip(x, y, y_err):
            y_var = y_erri ** 2 + (f_prime(xi, *m)*1) ** 2 # 1 è l'errore sulla pressione
            result += (yi - func1(xi, *m)) ** 2 / y_var
        return result

    cost.errordef = Minuit.LEAST_SQUARES

    my_minuit = Minuit(cost,(1,2,0),name=('a','b','c'))
    my_minuit.migrad()
    my_minuit.hesse()

    # print(my_minuit)
    fit = Interpolazione(x,y,y_err,x_err,func1,[1,2,0],names=['a','b','c'])
        
    # plt.errorbar(x,y,y_err,x_err,fmt='ok')
    # plt.plot(*fit.draw())
    # plt.show()
    
    print(fit)