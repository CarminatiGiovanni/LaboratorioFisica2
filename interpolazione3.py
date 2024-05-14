import jax
from jax import grad
from iminuit import Minuit
import numpy as np  # original numpy still needed, since jax does not cover full API
from numpy import ndarray, float64
from collections.abc import Callable
import scipy as sc
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)  # enable float64 precision, default is float32


class Interpolazione:
    def __init__(self,X: ndarray[float64], Y: ndarray[float64],
                 sigmaY: ndarray[float64] | float64,sigmaX: ndarray[float64] | float64,
                 model: Callable,guess: ndarray[float64],
                 weights: ndarray[float64] = None, names: list[str] = None,
                 iteration = 10) -> None:
        self.f = model
        self.Y = Y.astype('float64')
        self.X = X.astype('float64')
        self.N = len(X)
        self.iteration = iteration
        self.sigmaY = sigmaY
        self.sigmaX = sigmaX
        
        @jax.jit
        def cost(par):
            print('PARAMETROOOOOOo',par)
            result = 0.0
            for xi, yi in zip(X, Y):
                y_var = self.__error(xi, par)
                result += (yi - self.f(xi, par)) ** 2 / y_var
            return result
            
        
        cost.errordef = Minuit.LEAST_SQUARES

        self.m = Minuit(cost,guess)
        self.m.migrad()
        self.m.hesse()
        
    def __error(self,xi,par):
            
        print('PARAMETROOOOOOOOOOOOOO ',par)
        fact = 1/np.array([sc.special.factorial(i) for i in range(1, self.iteration+1)]) # 1/n!
        
        d = [jax.jit(grad(self.f))] # derivatives
        for _ in range(1, self.iteration):
            d.append(jax.jit(grad(d[-1])))  

        d = np.array([i(xi,par) for i in d])
        dx = np.array([self.sigmaX**i for i in range(1,self.iteration+1)])
        t = d*fact*dx # 1/n! * d^n/dx^n f(x) * dx^n
        print(t)
        
        return self.sigmaY**2 + np.sum(t)**2

    def __chi2(self):
        pass

    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        pass
    
def f():
    return 0





'''
y = np.array([2.25, 4.25, 6.5, 8.75, 10.75, 12.75])
x = np.array([10.,20.,30.,40.,50.,60.]) # [kPa]
#y_err = np.array([0.43, 0.43, 0.5, 0.43, 0.43, 0.43])
y_err = np.array([0.21, 0.21, 0.25, 0.21, 0.21, 0.21])
x_err = (np.ones(len(x))).astype(float)

print(y_err)

import jax
from iminuit import Minuit
from jax import grad

jax.config.update("jax_enable_x64", True)

def func1(x, m): # delta_N = ... * delta_p
    return (2*d*m/Lambda)*x

f_prime = jax.jit(grad(func1))

@jax.jit
def cost(m):
    result = 0.0
    for xi, yi, y_erri in zip(x, y, y_err):
        y_var = y_erri ** 2 + (f_prime(xi, m)*1) ** 2 # 1 è l'errore sulla pressione
        result += (yi - func1(xi, m)) ** 2 / y_var
    return result

cost.errordef = Minuit.LEAST_SQUARES

my_minuit = Minuit(cost,m=0)
my_minuit.migrad()
my_minuit.hesse()
display(my_minuit)


'''

import jax.numpy as jnp

if __name__ == '__main__':
    # x = jnp.array([1,2,3,4,5])
    # f = lambda x,a: a*x # **2 + b*x + c
    # y = f(x,2) + 3*np.random.normal(0,2,len(x))
    
    # plt.errorbar(x,y,yerr=1,fmt='ok')
    # # plt.show()
    
    # i = Interpolazione(x,y,1,1,f,1.0)
    # print(i.m)
    
    
    y = np.array([2.25, 4.25, 6.5, 8.75, 10.75, 12.75])
    x = np.array([10.,20.,30.,40.,50.,60.]) # [kPa]
    #y_err = np.array([0.43, 0.43, 0.5, 0.43, 0.43, 0.43])
    y_err = np.array([0.21, 0.21, 0.25, 0.21, 0.21, 0.21])
    x_err = (np.ones(len(x))).astype(float)

    print(y_err)

    import jax
    from iminuit import Minuit
    from jax import grad

    jax.config.update("jax_enable_x64", True)

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
    print(my_minuit)