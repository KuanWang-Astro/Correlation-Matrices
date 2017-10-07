import numpy as np
import matplotlib.pyplot as plt

class correlation(object):
    def __init__(self,table_w,table_wo):
        self.wp_w = table_w['wprp']
        self.wp_wo = table_wo['wprp']
        self.ngals_w = table_w['ngals']
        self.ngals_wo = table_wo['ngals']
        self.param_w = table_w['param']
        self.param_wo = table_wo['param']
        self.wpcov_w = table_w['wp_cov']
        self.wpcov_wo = table_wo['wp_cov']
        
    def compute_correlation_matrix(self): 
        self.wpcov_mean_w = np.mean(self.wpcov_w,axis=0)
        self.wpcov_mean_wo = np.mean(self.wpcov_wo,axis=0)
        self.sigma_wp_w = np.sqrt(self.wpcov_mean_w.diagonal())
        self.sigma_wp_wo = np.sqrt(self.wpcov_mean_wo.diagonal())
        self.all_comparison = (self.wpcov_mean_w/self.sigma_wp_w).T/self.sigma_wp_w
        self.half_wo = (self.wpcov_mean_wo/self.sigma_wp_wo).T/self.sigma_wp_wo
        self.dim = self.all_comparison.shape[0]
        for i in range(self.dim):
            for j in range(self.dim):
                if i>j:
                    self.all_comparison[i,j] = self.half_wo[i,j]
        return self.all_comparison
    
    def plot_correlation_matrix(self,vmax,vmin):
        self.compute_correlation_matrix()
        plt.imshow(self.all_comparison,cmap='bwr',vmax=vmax,vmin=vmin,interpolation='None')
        plt.colorbar()
