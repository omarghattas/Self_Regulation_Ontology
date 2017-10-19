# prediction_utils

import pandas,numpy

import rpy2.robjects as robjects
#import pandas.rpy.common as com

from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
R = robjects.r
base = importr('base')
stats = importr('stats')
mpath=importr('mpath')

# create a class that implements prediction using functions from R

class RModel:
    def __init__(self,modeltype,verbose=True,ncores=2,nlambda=100):
        self.modeltype=modeltype
        assert self.modeltype in ['NB', 'ZINB', 'ZIpoisson', 'poisson']
        self.verbose=verbose
        self.model=None
        self.coef_=None
        self.ncores=ncores
        self.nlambda=nlambda

    def fit(self,X,Y):
        self._fit_glmreg(X,Y)

    def _fit_glmreg(self,X,Y):
        if not isinstance(X, pandas.DataFrame):
            X=pandas.DataFrame(X,columns=['V%d'%i for i in range(X.shape[1])])
        X=X-X.mean(0)
        #X['intercept']=numpy.zeros(X.shape[0])
        if not isinstance(Y, pandas.DataFrame):
            Y=pandas.DataFrame(Y,columns=['X0'])

        if self.verbose:
            print('fitting using %s regression via mpath'%self.modeltype)
        data=X.copy()
        #data['y']=Y
        if self.modeltype=='poisson':
            robjects.globalenv['df']=pandas2ri.py2ri(data)
            robjects.r('df=data.matrix(df)')
            robjects.globalenv['y']=pandas2ri.py2ri(Y)
            robjects.r('y=data.matrix(y)')
            self.model=mpath.cv_glmreg(base.as_symbol('df'),base.as_symbol('y'),
                                    family = 'poisson')
            fit=self.model[self.model.names.index('fit')]
            self.lambda_which=numpy.array(self.model[self.model.names.index('lambda.which')])[0]
            self.coef_=numpy.array(fit[fit.names.index('beta')])[:,self.lambda_which-1]
        elif self.modeltype=='NB':
            data['y']=Y.copy()
            robjects.globalenv['df']=pandas2ri.py2ri(data)
            self.model=mpath.cv_glmregNB('y~.',base.as_symbol('df'),
                                n_cores=self.ncores,plot_it=False)
            fit=self.model[self.model.names.index('fit')]
            self.lambda_which=numpy.array(self.model[self.model.names.index('lambda.which')])[0]
            self.coef_=numpy.array(fit[fit.names.index('beta')])[:,self.lambda_which-1]

        elif self.modeltype=='ZINB' or self.modeltype=='ZIpoisson' :
            #data['y']=Y.copy()
            robjects.globalenv['df']=pandas2ri.py2ri(data)
            robjects.globalenv['y']=pandas2ri.py2ri(Y)
            robjects.r('df$y=y$X0')
            if self.modeltype=='ZINB':
                family='negbin'
            else:
                family='poisson'
            # this is a kludge because I couldn't get it to work using the
            # standard interface to cv_zipath
            robjects.r('fit=cv.zipath(y~.|.,df,family="%s",penalty="enet",plot.it=FALSE,nlambda=%d,n.cores=%d)'%(family,self.nlambda,self.ncores))

            self.model=robjects.r('fit')
            fit=self.model[self.model.names.index('fit')]
            #self.lambdas=numpy.array(self.model[self.model.names.index('lambda')])
            #if self.verbose:
            #    print('model:',self.model)
            self.lambda_which=numpy.array(self.model[self.model.names.index('lambda.which')])[0]
            if self.verbose:
                print("lambda_which = ",self.lambda_which)
            # just get the count coefficients
            robjects.r('coef_=coef(fit$fit,which=fit$lambda.which,model="count")')
            # drop the intercept term
            self.coef_=numpy.array(robjects.r('coef_'))[1:]

        #self.model=stats.lm('y~.', data = base.as_symbol('df')) #, family = "poisson")
    def predict(self,newX):
        if self.model is None:
            print('model must first be fitted')
            return None
        if not isinstance(newX, pandas.DataFrame):
            newX=pandas.DataFrame(newX,columns=['V%d'%i for i in range(newX.shape[1])])

        if self.modeltype=='poisson':
            robjects.globalenv['newX']=pandas2ri.py2ri(newX)
            robjects.r('newX=data.matrix(newX)')

            pred=mpath.predict_glmreg(self.model[self.model.names.index('fit')],
                                base.as_symbol('newX'),
                                which=self.lambda_which)
        elif self.modeltype=='NB':
            robjects.globalenv['newX']=pandas2ri.py2ri(newX)
            #robjects.r('newX=data.matrix(newX)')
            pred=mpath.predict_glmreg(self.model[self.model.names.index('fit')],
                                base.as_symbol('newX'),
                                which=self.lambda_which)
        elif self.modeltype=='ZINB' or self.modeltype=='ZIpoisson' :
            robjects.globalenv['newX']=pandas2ri.py2ri(newX)
            #robjects.r('newX=data.matrix(newX)')
            robjects.r('pred=predict(fit$fit,newX,which=fit$lambda.which)')
            pred=robjects.r('pred')

        return numpy.array(pred)

if __name__=='__main__':
    # run some tests
    # generate some data, here we just use gaussian
    X=pandas.DataFrame(numpy.random.randn(100,4),columns=['V%d'%i for i in range(4)])
    Y=X.dot([1,-1,1,-1])+numpy.random.randn(100)
    Yz=numpy.floor(Y-numpy.median(Y))
    Yz[Yz<0]=0
    Y=pandas.DataFrame(numpy.floor(Yz))

    for modeltype in [ 'poisson' ,'NB','ZINB', 'ZIpoisson']:
        rm=RModel(modeltype)
        rm.fit(X,Y)
        if not rm.model is None:
            print(R.summary(rm.model))
            pred=rm.predict(X)
            print('corr(pred,actual):',numpy.corrcoef(numpy.array(pred).T,Y.T))
            print(rm.coef_)
