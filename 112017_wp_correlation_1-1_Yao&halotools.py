
# coding: utf-8

# In[1]:

import gc
import numpy as np

import matplotlib.pyplot as plt

from halotools.sim_manager import CachedHaloCatalog

from datetime import datetime
import sys
sys.path.append('../Compute statistics new/')

from HOD_models import decorated_hod_model
from HOD_models import standard_hod_model

from halotools.empirical_models import MockFactory

from halotools.mock_observables import return_xyz_formatted_array
from halotools.utils import randomly_downsample_data
from halotools.mock_observables import wp_jackknife

from helpers.CorrelationFunction import projected_correlation

from concurrent.futures import ProcessPoolExecutor as Pool
import collections


# In[2]:

simname_list = ['chinchilla','bolshoi','chinchilla','bolplanck','chinchilla',                'diemerL0250','consuelo','diemerL0500','chinchilla']
version_list = ['250-2560','halotools_v0p4','250-2048','halotools_v0p4','250-1024',               'antonio','halotools_v0p4','antonio','250-512']
redshift_list = [0 for i in range(9)]
halofinder_list = ['rockstar' for i in range(9)]
Lbox_list = [250,250,250,250,250,250,420,500,250]
Nsidejk_list = [10,10,10,10,10,10,10,17,20,10]


# In[3]:

sim_list = zip(simname_list,version_list,redshift_list,halofinder_list,Lbox_list,Nsidejk_list)


# In[4]:

param_names = ('alpha','logM1','sigma_logM','logM0','logMmin',               'mean_occupation_centrals_assembias_param1','mean_occupation_satellites_assembias_param1')


# In[5]:

model_w = decorated_hod_model()
model_wo = standard_hod_model()


# In[6]:

##median values
p_w_20p0 = np.array((1.02654941214,13.1879106954,0.8781897069,12.1031391855,12.2692942798,0.91596941012,0.0258608345476))
p_wo_20p0 = np.array((1.14385007593,13.2858403826,0.348464903173,11.3075027005,11.9718570282))
p_w_20p5 = np.array((1.1385115,13.57578547,0.48679456,11.57262142,12.35790058,0.82680138,0.07528584))
p_wo_20p5 = np.array((1.19652223,13.59168639,0.18536064,11.20133648,12.25470422))
p_w_21p0 = np.array((1.17616081,13.95019057,0.49658948,12.65226286,12.78575124,0.26511832,0.08899419))
p_wo_21p0 = np.array((1.33738418,13.98811392,0.55950074,11.95796534,12.82356125))


# In[7]:

rbins1 = np.logspace(-1, 1.5, 13)
rbins2 = np.logspace(-1, 1.5, 20)
pi_max = 60


# In[8]:

def comb(mw,mwo):
    m = np.copy(mw)
    dim = m.shape[0]
    for i in range(dim):
        for j in range(dim):
            if i>j:
                m[i,j] = mwo[i,j]
    return m


# In[11]:
output_names = ('wp_w_yao', 'wpcov_w_yao', 'wp_wo_yao', 'wpcov_wo_yao', 'wp_w_halotools', 'wpcov_w_halotools', 'wp_wo_halotools', 'wpcov_wo_halotools')

def wp_wpcov(simidx,pw,pwo,rbins):
    
    output = []
    halocat = CachedHaloCatalog(simname = sim_list[simidx][0], version_name = sim_list[simidx][1],redshift = sim_list[simidx][2], halo_finder = sim_list[simidx][3])
    model_w.param_dict.update(dict(zip(param_names, pw)))
    model_w.populate_mock(halocat)
    model_wo.param_dict.update(dict(zip(param_names, pwo)))
    model_wo.populate_mock(halocat)
    gc.collect()
    
    pos_gals_w = return_xyz_formatted_array(*(model_w.mock.galaxy_table[ax] for ax in 'xyz'),velocity=model_w.mock.galaxy_table['vz'],\
velocity_distortion_dimension='z',period=sim_list[simidx][4])
    pos_gals_w = np.array(pos_gals_w,dtype=float)
    pos_gals_wo = return_xyz_formatted_array(*(model_wo.mock.galaxy_table[ax] for ax in 'xyz'),velocity=model_wo.mock.galaxy_table['vz'],\
velocity_distortion_dimension='z',period=sim_list[simidx][4])
    pos_gals_wo = np.array(pos_gals_wo,dtype=float)
    
    Nran = int(1e5*(sim_list[simidx][4]/250.)**3)
    xran = np.random.uniform(0, sim_list[simidx][4], Nran)
    yran = np.random.uniform(0, sim_list[simidx][4], Nran)
    zran = np.random.uniform(0, sim_list[simidx][4], Nran)
    randoms = np.vstack((xran,yran,zran)).T
    
    wp_w_halotools,wpcov_w_halotools = wp_jackknife(pos_gals_w,randoms,rbins,pi_max, Nsub=[sim_list[simidx][5],sim_list[simidx][5],1])
    output.append(wp_w_halotools)
    output.append(wpcov_w_halotools)
    wp_wo_halotools,wpcov_wo_halotools = wp_jackknife(pos_gals_wo,randoms,rbins,pi_max, Nsub=[sim_list[simidx][5],sim_list[simidx][5],1])
    output.append(wp_wo_halotools)
    output.append(wpcov_wo_halotools)
    
    wp_w_yao,wpcov_w_yao = projected_correlation(pos_gals_w, rbins, pi_max, sim_list[simidx][4], jackknife_nside=sim_list[simidx][5])
    output.append(wp_w_yao)
    output.append(wpcov_w_yao)
    wp_wo_yao,wpcov_wo_yao = projected_correlation(pos_gals_wo, rbins, pi_max, sim_list[simidx][4], jackknife_nside=sim_list[simidx][5])
    output.append(wp_wo_yao)
    output.append(wpcov_wo_yao)
    
    return output


def main(output_fname):
    nproc = 55

    i_list = [j for m in range(2) for l in [[i,i,i] for i in range(9)] for j in l]
    pw_list = [j for i in range(18) for j in [p_w_20p0,p_w_20p5,p_w_21p0]]
    pwo_list = [j for i in range(18) for j in [p_wo_20p0,p_wo_20p5,p_wo_21p0]]
    rbins_list = [j for j in [rbins1,rbins2] for i in range(27)]

    output_dict = collections.defaultdict(list)
    with Pool(nproc) as pool:
        print 'starting'
        for i, output_data in enumerate(pool.map(wp_wpcov, i_list,pw_list,pwo_list,rbins_list)):
            if i%nproc == nproc-1:
                sys.stdout.write("\r{} {}".format(i, str(datetime.now())))
                sys.stdout.flush() 
            for name, data in zip(output_names, output_data):
                output_dict[name].append(data)
    
    for name in output_names:
        output_dict[name] = np.array(output_dict[name])

    np.savez(output_fname, **output_dict)



if __name__ == '__main__':
    main('112017_wp_corr_yao&halotools')

# In[ ]:



