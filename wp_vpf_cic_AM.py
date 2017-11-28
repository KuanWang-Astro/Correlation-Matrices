import argparse

parser = argparse.ArgumentParser(description='Compute wp(rp),P(Ncic),vpf')
parser.add_argument('--Lbox',type=int,required=True,dest='Lbox')
parser.add_argument('--simname',required=True,dest='simname')
parser.add_argument('--Nsidejk',type=int,default=7,dest='Nsidejk')
parser.add_argument('--version',default='halotools_v0p4',dest='version')
parser.add_argument('--redshift',type=float,default=0.,dest='redshift')
parser.add_argument('--halofinder',default='rockstar',dest='halofinder')
parser.add_argument('--haloprop',default='halo_vpeak',dest='haloprop')
parser.add_argument('--outfile',required=True,dest='outfile')
parser.add_argument('--parallel',type=int,default=5,dest='nproc')
args = parser.parse_args()


from abundance_matching import AM

import collections
import gc
import numpy as np
from concurrent.futures import ProcessPoolExecutor as Pool

from datetime import datetime
import sys

from halotools.sim_manager import CachedHaloCatalog

from halotools.mock_observables import return_xyz_formatted_array
from halotools.mock_observables import counts_in_cylinders
from halotools.mock_observables import void_prob_func
from halotools.mock_observables import wp
from halotools.utils import randomly_downsample_data
from helpers.CorrelationFunction import projected_correlation

##########################################################

output_names = ('ngals','Pcic','vpf','wprp','wp_y','wpcov','param')

##########################################################

Lbox = args.Lbox

proj_search_radius = 2.0         ##a cylinder of radius 2 Mpc/h
cylinder_half_length = 10.0      ##half-length 10 Mpc/h

##cic

r_vpf = np.logspace(0, 1.25, 20)
num_spheres = int(1e5)
##vpf

pi_max = 60
r_wp = np.logspace(-1,1.4,20)

##wp

##########################################################

def calc_all_observables(threshold):
    
    am = AM(args.haloprop,halocat.halo_table,Lbox,threshold)
    
    gc.collect()
    
    output = []


    pos_gals_d = am.match_gal()
    pos_gals_d = np.array(pos_gals_d,dtype=float)
    # ngals
    output.append(pos_gals_d.shape[0])
    
    # Pcic
    output.append(np.bincount(counts_in_cylinders(pos_gals_d, pos_gals_d, proj_search_radius, \
            cylinder_half_length), minlength=100)[1:100]/float(pos_gals_d.shape[0]))

    
    # vpf
    output.append(void_prob_func(pos_gals_d, r_vpf, num_spheres, period=Lbox))
    
    # wprp
    output.append(wp(pos_gals_d, r_wp, pi_max, period=Lbox))
    
    # wprp and cov
    wp_wpcov = projected_correlation(pos_gals_d, r_wp, pi_max, Lbox, jackknife_nside=args.Nsidejk)
    output.append(wp_wpcov[0])
    output.append(wp_wpcov[1])
    
    # luminosity threshold
    output.append(threshold)
    
    return output


############################################################
consuelo20_box_list = ['0_4001','0_4002','0_4003','0_4004','0_4020','0_4026','0_4027','0_4028','0_4029','0_4030',\
            '0_4032','0_4033','0_4034','0_4035','0_4036','0_4037','0_4038','0_4039','0_4040']


threshold_list = np.array((-22.,-21.5,-21.,-20.5,-20))

def main(output_fname):
    
    output_dict = collections.defaultdict(list)
    nproc = args.nproc
    
    global halocat
    
    with Pool(nproc) as pool:
        if args.simname=='consuelo20' and args.version=='all':
            for box in consuelo20_box_list:
                halocat = CachedHaloCatalog(simname = args.simname, version_name = box,redshift = args.redshift, \
                                halo_finder = args.halofinder)
                for i, output_data in enumerate(pool.map(calc_all_observables, threshold_list)):
                    sys.stdout.write("\r{} {}".format(i, str(datetime.now())))
                    sys.stdout.flush()
                    for name, data in zip(output_names, output_data):
                        output_dict[name].append(data)
                print box
        else:
            halocat = CachedHaloCatalog(simname = args.simname, version_name = args.version,redshift = args.redshift, \
                                halo_finder = args.halofinder)
            for i, output_data in enumerate(pool.map(calc_all_observables, threshold_list)):
                sys.stdout.write("\r{} {}".format(i, str(datetime.now())))
                sys.stdout.flush()
                for name, data in zip(output_names, output_data):
                    output_dict[name].append(data)
    
    for name in output_names:
        output_dict[name] = np.array(output_dict[name])

    np.savez(output_fname, **output_dict)


if __name__ == '__main__':
    main(args.outfile)
    with open(args.outfile+'_log','w') as f:
        for arg in vars(args):
            f.write(str(arg)+':'+str(getattr(args, arg))+'\n')
        f.write(sys.argv[0])


