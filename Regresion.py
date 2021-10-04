# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:23:55 2020

@author: evule
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, max_error,explained_variance_score, mean_absolute_error, median_absolute_error, mean_squared_log_error

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

import statsmodels.api as sm
from scipy import stats

from math import pi

Base=pd.read_excel(r'C:\Users\evule\Desktop\Python Estadistica\Regresion.xlsx')



FechaInicio='2019-03-31'
FechaFin='2020-04-30'
Variable='A'
VariablesExplicativas=['B','C','E']

Base=Base.loc[Base[Base.Fecha==FechaInicio].index.values[0]:Base[Base.Fecha==FechaFin].index.values[0],:]
Base.reset_index(drop=True, inplace=True)

Y=pd.DataFrame(Base.loc[:,Variable])
X=pd.DataFrame(Base.loc[:,VariablesExplicativas])

Unos=pd.DataFrame([1,1,1,1,1,1,1,1,1,1,1,1,1,1], columns=['Intercepto'])
#X=pd.concat([Unos,X], axis=1)


YCentrada=Y-Y.mean()
XCentrada=X-X.mean()

n=len(YCentrada)
k=len(XCentrada.columns)

B=pd.DataFrame((np.linalg.inv(XCentrada.transpose()@XCentrada)) @ XCentrada.transpose() @ YCentrada).transpose()
Prediccion=pd.DataFrame(np.sum(XCentrada*B.values, axis=1), columns=Y.columns)+Y.mean().values
#Nota: yo hago la prediccion con los predictores centrados. Tambien se puede hacer sin centrar (como hace el summary mas abajo). El unico coeficiente que cambia es el intercepto o const. El resto de los coeficientes, el resultado de las predicciones y los tests son identicos.
#El error Medio Cuadratico es dividido n-k-1, pero en el paquete sklearn LinearRegression lo hacen con n.
#ErrorMedioCuadratico=((Y-Prediccion)**2).mean().values[0]
ErrorMedioCuadratico1=(((Y-Prediccion)**2).sum().values[0])/(n-k-1)
ErrorMedioCuadratico2=(((Y-Prediccion)**2).mean().values[0])
RCuadrado=1-((Y-Prediccion)**2).sum().values[0]/((Y-Y.mean())**2).sum().values[0]
RCuadradoAjustado=1-((len(YCentrada)-1)/(n-k-1))*(1-RCuadrado)
ErrorMaximo=abs((Y-Prediccion)).max().values[0]
VarExplicada=1-(Y-Prediccion).var().values[0]/Y.var().values[0]
ErrorMedioAbsoluto=abs((Y-Prediccion)).sum().values[0]/n
ErrorMedianoAbsoluto=(abs((Y-Prediccion))).median().values[0]
ErrorCuadMedioLog=((np.log(1+pd.DataFrame(Y[Variable]))-np.log(1+pd.DataFrame(Prediccion[Variable])))**2).mean().values[0]

print(B)
print(Prediccion)
#print('EMC es '+ str(ErrorMedioCuadratico))
print('R2 es ' + str(RCuadrado))
print('R2 ajustado es ' + str(RCuadradoAjustado))
#print(ErrorMaximo)
#print(VarExplicada)
#print(ErrorMedioAbsoluto)
#print(ErrorMedianoAbsoluto)
#print(ErrorCuadMedioLog)

 
LogVerosimilitud=n/2*np.log(1/(2*pi*ErrorMedioCuadratico2))-ErrorMedioCuadratico2*n/(2*ErrorMedioCuadratico2)
#print('La verosimilitud es ' + str(LogVerosimilitud))
AIC=-2*LogVerosimilitud+2*(k+1)
#print('AIC es ' + str(AIC))
BIC=-2*LogVerosimilitud+np.log(n)*(k+1)
#print('BIC es ' + str(BIC))
EstadSignifConjunta=(RCuadrado/k)/((1-RCuadrado)/(n-k-1))
TeoricoSignifConjunta=stats.f.ppf(0.05,dfn=3,dfd=10)
#print('El estadistico F es ' +str(EstadSignifConjunta))
#print(TeoricoSignifConjunta)
if EstadSignifConjunta>TeoricoSignifConjunta:
    print('Rechazo Ho, que establece que ningun coeficiente es significativo')
else:
    print('No rechazo Ho, que establece que ningun coeficiente es significativo')


MatrizVarCovarEstimadores=ErrorMedioCuadratico1 *np.linalg.inv((XCentrada.transpose()@XCentrada))
DesvioEstandarCoeficientes=[(MatrizVarCovarEstimadores[x,x])**0.5 for x in range(k)]
#print(DesvioEstandarCoeficientes)
EstadT=pd.DataFrame(np.divide(B,DesvioEstandarCoeficientes))
#print('EstadT es ' + str(EstadT))

ParaIntervalo=abs(stats.t.ppf(0.025,n-k-1))
IntMin=B-np.multiply(ParaIntervalo,DesvioEstandarCoeficientes)
IntMax=B+np.multiply(ParaIntervalo,DesvioEstandarCoeficientes)
#print('IntMin ' + str(IntMin))
#print('IntMax ' + str(IntMax))

PVaueT=(stats.t.cdf(-abs(EstadT),n-k-1))*2 #si es bajo, puedo rechazar Ho (o sea,rechazar que el coef es insignificativo)
#print('PVaueT es ' + str(PVaueT))


#Con paquetes Python:
regr=LinearRegression(fit_intercept=True, normalize=True)
regr.fit(X,Y)
print(regr.coef_)
print(regr.predict(X))
MSE=mean_squared_error(Y,Prediccion)
RSquare=r2_score(Y,Prediccion)
ErrorMax=max_error(Y,Prediccion)
ExplVar=explained_variance_score(Y,Prediccion)
AbsMeanError=mean_absolute_error(Y,Prediccion)
AbsMedianError=median_absolute_error(Y,Prediccion)
MSLE=mean_squared_log_error(Y,Prediccion)
#print(MSE)
#print(RSquare)
#print(ErrorMax)
#print(ExplVar)
#print(AbsMeanError)
#print(AbsMedianError)
#print(MSLE)

regr = OLS(Y, add_constant(X)).fit()
#print(regr.aic)

X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
#print(est.fit().f_pvalue)
#print(est.fit().summary())

