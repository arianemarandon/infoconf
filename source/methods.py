import numpy as np
import torch

from source.misc import BH, compute_emp_pvalue, train_test_split_wrapper, custom_subset
  

class StandardSplitConformal:
    def __init__(self, X, Y, black_box, alpha, random_state=2020, allow_empty=True, verbose=False,
                 null_class=None):
        self.allow_empty = allow_empty

        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split_wrapper(X, Y, test_size=0.5, random_state=random_state)
        n2 = len(X_calib) #X_calib.shape[0]

        self.black_box = black_box


        # Fit model
        self.black_box.fit(X_train, Y_train)

        # Form prediction sets on calibration data
        p_hat_calib = self.black_box.predict_proba(X_calib)
        scores_calib = 1-p_hat_calib[np.arange(n2), Y_calib]
        q_level = np.ceil((n2+1)*(1-alpha))/n2
        self.qhat = np.quantile(scores_calib, q_level, method='higher')
        self.K=np.max(Y)+1

        

    def predict(self, X):
        """
        X:observation (single data point)
        Returns: prediction set 
        """
        m=len(X)
        #compute score for X
        test_score = self.black_box.predict_proba(X)
        S_hat=[0.]*m
        for i in range(m):
            S_hat[i]=np.arange(self.K)[(test_score[i] >= 1-self.qhat)]

        return S_hat
    


class basicSel(object):
    '''
    apply BH without pre-processing 
    '''
    def __init__(self, X, Y, black_box, alpha, random_state, 
                allow_empty=True, verbose=False,
                null_class=None):

        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split_wrapper(X, Y, test_size=0.5, random_state=random_state)
        n2 = len(X_calib) 

        self.black_box = black_box

        # Fit model
        self.black_box.fit(X_train, Y_train)

        
        # Form prediction sets on calibration data
        p_hat_calib = self.black_box.predict_proba(X_calib)
        self.scores_calib = 1-p_hat_calib[np.arange(n2), Y_calib]

        self.K = np.max(Y_calib)+1 
        self.scores_calib_by_classes=[self.scores_calib[Y_calib==k] for k in range(self.K)]


        self.alpha=alpha


        self.pvalues=None


    def _compute_pvalues(self, X):
        m=len(X)
        pred_test=self.black_box.predict_proba(X)
        scores_test = 1-pred_test #m x K
        self.pvalues = np.array([ [compute_emp_pvalue(scores_test[i,k], self.scores_calib) 
                                for k in range(self.K)] for i in range(m) ]) #m x K
        
        
        
    def _compute_qvalues(self):
        pass #eg np.min(self.pvalues, axis=1)


    def predict(self, X):

        self._compute_pvalues(X)



        m=len(X)

        #apply BH to qvalues
        qvalues=self._compute_qvalues()
        R = BH(qvalues, level=self.alpha)
        threshold = self.alpha * len(R) / m 
        
        S_hat=[0.]*m
        for i, pval_row in enumerate(self.pvalues):
            if np.nonzero(pval_row > threshold)[0].size:
                S_hat[i]=np.nonzero(pval_row > threshold)[0]
            else: S_hat[i] = np.array([np.argmax(pval_row)]) 


        return S_hat
    

class minSel(basicSel):

    def __init__(self, X, Y, black_box, alpha, random_state=2020, 
                allow_empty=True, verbose=False, null_class=None):
        basicSel.__init__(self, X, Y, black_box, alpha, random_state, allow_empty, verbose,None)

    def _compute_qvalues(self):
        return np.min(self.pvalues, axis=1)

class minSelCond(minSel):
    '''
    '''
    def __init__(self, X, Y, black_box, alpha, random_state=2020, 
                allow_empty=True, verbose=False,
                null_class=None):
        minSel.__init__(self, X, Y, black_box, alpha, random_state, allow_empty, verbose)


    def _compute_pvalues(self, X):
        m=len(X)
        scores_test = 1-self.black_box.predict_proba(X) #m x K
        
        pvalues = np.array([ [compute_emp_pvalue(scores_test[i,k], self.scores_calib_by_classes[k]) 
                                for k in range(self.K)] for i in range(m) ]) #m x K
        
        self.pvalues = pvalues





class nonullSelbasic(basicSel):
    def __init__(self, X, Y, black_box, alpha, random_state=2020, 
                allow_empty=True, verbose=False, 
                null_class=None):
        basicSel.__init__(self, X, Y, black_box, alpha, random_state, allow_empty, verbose,
                          null_class)
        
        self.null_class=null_class
    def _compute_qvalues(self):
        return self.pvalues[:, self.null_class]   



class nonullSelcond(nonullSelbasic):
    def __init__(self, X, Y, black_box, alpha, random_state=2020, 
                allow_empty=True, verbose=False, 
                null_class=None):
        nonullSelbasic.__init__(self, X, Y, black_box, alpha, random_state, allow_empty, verbose,
                          null_class)
        
        self.null_class=null_class

    def _compute_pvalues(self, X):
        m=len(X)
        scores_test = 1-self.black_box.predict_proba(X) #m x K
        
        pvalues = np.array([ [compute_emp_pvalue(scores_test[i,k], self.scores_calib_by_classes[k]) 
                                for k in range(self.K)] for i in range(m) ]) #m x K
        
        self.pvalues = pvalues   
    


class preprocSel(object):
    '''
    with pre-processing 
    '''
    def __init__(self, X, Y, black_box, alpha, random_state, 
                allow_empty=True, verbose=False,
                null_class=None):
        self.allow_empty = allow_empty
        self.alpha=alpha
        self.null_class = null_class

        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split_wrapper(X, Y, test_size=0.5, random_state=random_state)
        self.black_box = black_box
        self.black_box.fit(X_train, Y_train)        # Fit model

        #Pre-processing: Further split calibration sample 
        self.X_calib1, self.X_calib2, self.Y_calib1, self.Y_calib2 = train_test_split_wrapper(X_calib, Y_calib, test_size=0.5, random_state=random_state)
        self.K = np.max(Y)+1

        self.pvalues=None

    def _compute_first_stage_pvalues(self, X):
        pass

    def _compute_qvalues(self):
        pass 
        
    def predict(self, X):
        m=len(X)
        
        # compute first stage p-values
        first_stage_pvalues=self._compute_first_stage_pvalues(X) #m+r

        # apply BH to first stage p-values 
        R = BH(first_stage_pvalues, level=self.alpha) #selection set S0 as in paper 

        R_cal=R[R>=len(X)]
        
        R_test = R[R <len(X)]

        ind_cal = R_cal- len(X) #indexes wrt X_calib2
        ind_test = R_test #indexes wrt to X are the same (but not ordered)

        # compute "conditional on selection" pvalues as in paper
        S_hat=[np.arange(self.K)]*m
        
        if len(ind_test)>0:  

            if isinstance(X, np.ndarray):
                new_X_test=X[ind_test] 
                new_X_calib= self.X_calib2[ind_cal]

            elif isinstance(X, torch.utils.data.Dataset):
                new_X_test= custom_subset(X, ind_test)
                new_X_calib= custom_subset(self.X_calib2, ind_cal)

            new_n_test= len(new_X_test)
            new_scores_test = 1-self.black_box.predict_proba(new_X_test)

            new_Y_calib = self.Y_calib2[ind_cal]
            n_cal=len(new_X_calib)
            new_scores_calib = 1-self.black_box.predict_proba(new_X_calib)[np.arange(n_cal),new_Y_calib]
            

            #conformal iid p-values 
            self.pvalues = np.array([ [compute_emp_pvalue(new_scores_test[i,k], new_scores_calib) 
                                for k in range(self.K)] for i in range(new_n_test) ]) #new_n_test x K
        
            # apply BH and compute threshold
            qvalues=self._compute_qvalues()
            
            R_new = BH(qvalues, level=self.alpha)#indexes wrt to new_X_test (reordered X or subset of it)

            threshold = self.alpha * len(R_new) / len(R_test) # below alpha

            # get prediction sets (idem usual)
            #pval_row[0] corresponds to X[ind_test[0]]
            for i,pval_row in enumerate(self.pvalues): #ordered according to ind_test
                if np.nonzero(pval_row> threshold)[0].size:
                    S_hat[ind_test[i]]=np.nonzero(pval_row > threshold)[0]
                else: S_hat[ind_test[i]] = np.array([np.argmax(pval_row)]) 


        return S_hat
    

class nonullSelpreproc(preprocSel):
    '''
    '''
    def __init__(self, X, Y, black_box, alpha, random_state=2020, 
                allow_empty=True, verbose=False,
                null_class=None):
        preprocSel.__init__(self, X, Y, black_box, alpha, random_state, allow_empty, verbose,
                          null_class)
        

    def _compute_qvalues(self):
        return self.pvalues[:, self.null_class]    
    
    def _compute_first_stage_pvalues(self, X):

        X_calib, Y_calib = self.X_calib1, self.Y_calib1
        n=len(Y_calib)
        ind_null = (Y_calib==self.null_class)
        
        if isinstance(X, np.ndarray):
            X_test = np.concatenate([X, self.X_calib2]) 
            X_calib = X_calib[ind_null]
        elif isinstance(X, torch.utils.data.Dataset):
            X_test = torch.utils.data.ConcatDataset([X, self.X_calib2]) 
            X_calib = custom_subset(X_calib, np.arange(n)[ind_null]) 

        n_test=len(X_test)
        scores_test=1-self.black_box.predict_proba(X_test)[:,self.null_class]

        scores_calib= 1-self.black_box.predict_proba(X_calib)[:,self.null_class]

        pvalues =np.array([compute_emp_pvalue(scores_test[i], scores_calib) for i in range(n_test)])   

        null_prop = (sum(ind_null)+1)/(n+1)
        pvalues *= null_prop 


        return pvalues

class minSelpreproc(preprocSel):
    def __init__(self, X, Y, black_box, alpha, random_state=2020, 
                allow_empty=True, verbose=False,
                null_class=None):
        preprocSel.__init__(self, X, Y, black_box, alpha, random_state, allow_empty, verbose,
                          null_class)
        

    def _compute_qvalues(self):
        return np.min(self.pvalues, axis=1)

    def _compute_first_stage_pvalues(self, X):

        X_calib, Y_calib = self.X_calib1, self.Y_calib1

        if isinstance(X, np.ndarray):
            X_test = np.concatenate([X, self.X_calib2]) 
        elif isinstance(X, torch.utils.data.Dataset):
            X_test = torch.utils.data.ConcatDataset([X, self.X_calib2]) 

        n=len(X_calib)
        n_test=len(X_test)
        scores_test=1-self.black_box.predict_proba(X_test)
        pred_calib=self.black_box.predict_proba(X_calib)
        scores_calib= 1-pred_calib[np.arange(n),Y_calib]

        pvalues =np.array([np.min([compute_emp_pvalue(scores_test[i,k], scores_calib) 
                           for k in range(self.K)]) for i in range(n_test)])   

        return pvalues 
