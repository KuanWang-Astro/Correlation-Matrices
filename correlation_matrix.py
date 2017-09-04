import numpy as np
import matplotlib.pyplot as plt

class correlation(object):
    def __init__(self,table_w,table_wo):
        self.wp_w = table_w['wprp']
        self.wp_wo = table_wo['wprp']
        self.ngals_w = table_w['ngals']
        self.ngals_wo = table_wo['ngals']
        self.Pcic_w = table_w['Pcic'][:,:30]
        self.Pcic_wo = table_wo['Pcic'][:,:30]
        self.vpf_w = table_w['vpf'][:,:-1]
        self.vpf_wo = table_wo['vpf'][:,:-1]
        
    def compute_correlation_matrix(self):    
        self.all_comparison = np.corrcoef(np.concatenate((self.wp_w.T,self.Pcic_w.T,self.vpf_w.T)))
        self.half_wo = np.corrcoef(np.concatenate((self.wp_wo.T,self.Pcic_wo.T,self.vpf_wo.T)))
        self.dim = self.all_comparison.shape[0]
        for i in range(self.dim):
            for j in range(self.dim):
                if i>j:
                    self.all_comparison[i,j] = self.half_wo[i,j]
        return self.all_comparison
    
    def plot_correlation_matrix(self):
        self.compute_correlation_matrix()
        plt.imshow(self.all_comparison,cmap='bwr',vmax=1,vmin=-1,interpolation='None')
        #plt.colorbar()
        plt.xticks((10,35,57),('wp','P(Ncic)','vpf'))
        plt.yticks((10,35,57),('wp','P(Ncic)','vpf'))
        plt.axhline(18.5,color='k')
        plt.axhline(48.5,color='k')
        plt.axvline(18.5,color='k')
        plt.axvline(48.5,color='k')
