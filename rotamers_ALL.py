import os 
import mdtraj as md
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, product
import pickle
import glob
import matplotlib as mpl
import warnings
import sys

warnings.filterwarnings( "ignore" )

########### FUNCTIONS ###########

def extract_methyls( angles, table ):
    
    methyls = ['ALA', 'MET', 'VAL', 'ILE', 'THR', 'LEU']
    
    idx_methyls = []
    chi_methyls = []
    residues    = []
    
    for k,ndx in enumerate(angles[0]):
        res = list(table[table.index == ndx[0]]['resName'])[0]
        if res in methyls:
            idx_methyls.append( k )
            residues.append( str(res) + str(list(table[table.index == ndx[0]]['resSeq'])[0]) )    
            
    for chi in angles[1]:
        chi_methyls.append( chi[idx_methyls] )
            
    return chi_methyls, residues

def save_hist( results, out ):
    with open( out + ".pkl", "wb" ) as fp: # save all the results in a pickle
        pickle.dump( results, fp )
        

trajfile   = sys.argv[1] #"/storage1/kummerer/T4L/a15ipq/4-md/sim1/Protein_sim1.xtc" 
top        = sys.argv[2] #"/storage1/kummerer/T4L/a15ipq/4-md/Protein_initial_nopbc.pdb"
outdir     = sys.argv[3] #"/storage1/kummerer/T4L/a15ipq/4-md/sim1/all/" 
os.system("mkdir " + outdir )

########### MAIN ########### 

stride     = 10
traj       = md.load( trajfile, top = top, stride = stride )
topology   = traj.topology
table, _   = topology.to_dataframe()

funcs = [ md.compute_chi1, md.compute_chi2, md.compute_chi3, md.compute_phi, md.compute_psi ]
outs  = [ "chi1", "chi2", "chi3", "phi", "psi"]

for k,f in enumerate(funcs):
    angle = f( traj, periodic = False )
    angm, residues  = extract_methyls( angle, table )
#     hist  = get_hist(  nblocks, block_size, angm )
    save_hist( residues, outdir + "residues_" + outs[k] )
    save_hist( angm, outdir + outs[k] )