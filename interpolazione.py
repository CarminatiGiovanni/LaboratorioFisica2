import numpy as np
from numpy import ndarray, float64
from scipy.optimize import curve_fit
from scipy.stats import norm
import scipy.stats as sc

'''
Classe per le interpolazioni (sfrutta sc.curve_fit)

__init__:
    X: valori sull'asse delle ascisse
    Y: valori sull'asse delle ordinate
    f: funzione per l'interpolazione
    sigmaY_strumento: incertezza delle Y
    p0: initial guess per i parametri di f
    weights: pesi per minimi quadrati pesati

NOTA: 
    la classe stima il chiquadro e il pvalue solo se è dato un certo sigmaY_strumento,
    altrimenti si limita a stimare a posteriori il sigmaY.
    Se il parametro weights non è nullo il sigmaY viene calcolato con il metodo dei minimi quadrati pesati
'''

class Interpolazione:
    def __init__(self,X: ndarray[float64], Y: ndarray[float64],f,sigmaY_strumento: ndarray[float64] | float64 = None,p0 = None,weights: ndarray[float64] = None, names: list[str] = None) -> None:
        self.f = f
        self.Y = Y.astype('float64')
        self.X = X.astype('float64')
        self.N = len(X)

        self.bval, self.cov_matrix = curve_fit(f,X,Y,p0=p0)
        self.sigma_bval = np.sqrt(np.diag(self.cov_matrix)) * (self.N/(self.N-len(self.bval))) # CORREZIONE DI BESSEL per interpolazioni

        self.x_best = np.linspace(min(X),max(X),100)
        self.y_best = f(self.x_best,*self.bval)

        # self.sigmaY = np.sqrt(self.__sigmaY()**2 + sigmaY_strumento**2) # propaga con sigmaY strumento

        if sigmaY_strumento is not None:
            self.sigmaY = sigmaY_strumento
            self.df = self.N - len(self.bval)
            self.rchisquare = self.__rchisquare()
            self.pvalue = np.round(sc.chi2.sf(self.rchisquare, self.df)*100,1)
        else:
            if weights is not None: # minimi quadrati pesati
                self.w = weights
                self.sigmaY = self.__w_sigmaY()
            else:                   # minimi quadrati
                self.sigmaY = self.__sigmaY()

        if names != None: # assegna i nomi alle variabili
            self.bval = {x : y for x,y in zip(names,self.bval)}
            self.sigma_bval = {x : y for x,y in zip(names,self.sigma_bval)}

    def __sigmaY(self): # deviazione standard
        return np.sqrt(
            np.sum( (self.Y - self.f(self.X,*self.bval))**2 ) 
            / (self.N - len(self.bval)))
    
    def __w_sigmaY(self): # minimi quadrati pesati
        return np.sqrt(
            np.sum(self.w*((self.Y - self.f(self.X,*self.bval))**2)) /
            ((np.sum(self.w)*(self.N-len(self.bval))) / self.N)
        )

    def __rchisquare(self):
        exp_val = self.f(self.X,*self.bval)
        return np.round(np.sum(((self.Y - exp_val) / self.sigmaY)**2) / self.df, 2)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        s1 = f"""   
Parameters: {self.bval} 
Sigma parameters: {self.sigma_bval}

sigmaY: {self.sigmaY}
"""
        s2 = f"""
chiquadro ridotto: {self.rchisquare}
df: {self.df}
pvalue: {self.pvalue}%""" if hasattr(self,'rchisquare') else ''
        
        s3 = f"""
covariance matrix: {self.cov_matrix}    
"""
        return s1 + s2 + s3

class RettaInterpolata(Interpolazione):
    def __init__(self,X,Y,sigmaY_strumento: ndarray[float64] | float64 = None,weights: ndarray[float64] = None):
        f = lambda x,A,B : A + B*x
        super().__init__(X,Y,f,sigmaY_strumento = sigmaY_strumento,names = ['A','B'],p0 = [Y[0], (Y[len(Y)-1] - Y[0]) / (X[len(X)-1] - X[0])],weights=weights)
        self.A = self.bval['A']
        self.B = self.bval['B']
        self.sigmaA = self.sigma_bval['A']
        self.sigmaB = self.sigma_bval['B']

    def __str__(self) -> str:
        s1 = f"""
linearità A + BX
    
A: {self.A} 
B: {self.B}
sigmaA: {self.sigmaA}
sigmaB: {self.sigmaB}

sigmaY: {self.sigmaY}    
"""
        s2 = f"""chiquadro ridotto: {self.rchisquare}
df: {self.df}
pvalue: {self.pvalue}%""" if hasattr(self,'rchisquare') else ''
        return s1 + s2

class RettaInterpolataB(Interpolazione):
    def __init__(self, X: ndarray[float64], Y: ndarray[float64], sigmaY_strumento: ndarray[float64] | float64 = None,weights: ndarray[float64] = None) -> None:
        f = lambda x,B : B*x
        super().__init__(X,Y,f,sigmaY_strumento = sigmaY_strumento,names = ['B'],p0 = [(Y[len(Y)-1] - Y[0]) / (X[len(X)-1] - X[0])],weights=weights)
        self.B = self.bval['B']
        self.sigmaB = self.sigma_bval['B']

    def __str__(self) -> str:
        s1 = f"""
linearità BX
    
B: {self.B}
sigmaB {self.sigmaB}

sigmaY: {self.sigmaY}
"""
        s2 = f"""chiquadro ridotto: {self.rchisquare}
df: {self.df}
pvalue: {self.pvalue}%""" if hasattr(self,'rchisquare') else ''
        return s1 + s2

def decimal_val(x,decimals = 2, exp=0, udm: str = '') -> str:
    x = np.round(x*np.power(10,-exp),decimals)
    return f'{x} {udm}' if exp == 0 else f'{x}e{exp} {udm}'


def final_val(x,sigma,decimals = 2,exp = 0, udm: str = '') -> str:
    x = np.round(x*np.power(10,-exp),decimals)
    sigma = np.round(sigma*np.power(10,-exp),decimals)
    return f'{x} ± {sigma} {udm}' if exp == 0 else f'({x} ± {sigma})e{exp} {udm}'

def b_std(x: ndarray): # deviazione standard con correzione di bessel
    return np.sqrt(np.sum((x - np.mean(x))**2)/(len(x)-1))

def probability_under_norm(mean,sigma,val):
    x = np.linspace(mean - 5*sigma,mean+5*sigma,1000)
    y = norm.pdf(x,loc = mean,scale = sigma)
    index_1,index_2 = 0,0
    i = 0
    t = np.abs(mean-val)/sigma
    for v in x:
        if v<mean-t*sigma:
            index_1 = i
        if v>mean+t*sigma:
            index_2 = i
            break
        i+=1
    A = np.trapz(y[index_1:index_2], x[index_1:index_2])
    return np.round(A,3)


if __name__ == '__main__':

    # print(probability_under_norm(0.2474,0.0035,0.250))

    # val = 0.51543e-12
    # sigma_val = 1.543e-15
    # print(final_val(val,sigma_val,exp=-14, decimals=3))


    import matplotlib.pyplot as plt

    X = np.linspace(0,10,10)
    Y = np.array([1,2,2,3,3,3,4,5,5,6]) #np.sin(X)*X**2#np.sin(X)# np.array([i for i in np.random],dtype=float64)
    r = RettaInterpolata(X,Y,1)
    print(r)
    def ret(x,A,B):
        return A + B*x
    
    r = Interpolazione(X,Y,ret,1)
    plt.errorbar(X,Y,fmt='o', yerr=r.sigmaY, capsize=7, color='red', ecolor='black')
    plt.plot(r.x_best,r.y_best)
    plt.show()
    print(r)

    # def f(x,a,b,c):
    #     return a*x**2 + b*x + c
    
    # X,Y = np.array([1,2,3,4,5,6,7,8], dtype=float64), np.array([1,3,10,15,28,33,45,64],dtype=float64)

    # r = Interpolazione(X,Y,f,names=['a','b','c'])
    # # print(r)
    # plt.errorbar(X,Y,fmt='o', yerr=r.sigmaY, capsize=7, color='red', ecolor='black')
    # plt.plot(r.x_best,r.y_best)
    # plt.show()