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

def parse():
    parser = argparse.ArgumentParser(description='')

    # Mandatory Inputs
    parser.add_argument('--traj', type = str, required = True,
                        help = 'Path to the trajectory file.')
    parser.add_argument('--top', type = str, required = True,
                        help = 'Path to a reference PDB file.')
    parser.add_argument('--out', type = str, required = True,
                        help = 'Path to the output directory.')
    parser.add_argument('--stride', type = int, required = False, default = 10,
                        help = 'Stride to read the trajectory (default: 10 frames).')
    args = parser.parse_args()

    return args.traj, args.top, args.out, args.stride

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
        
def main():
    trajfile, top, outdir, stride = parse()
    os.system( "mkdir " + outdir )

    traj       = md.load( trajfile, top = top, stride = stride )
    topology   = traj.topology
    table, _   = topology.to_dataframe()

    funcs = [ md.compute_chi1, md.compute_chi2, md.compute_chi3, md.compute_phi, md.compute_psi ]
    outs  = [ "chi1", "chi2", "chi3", "phi", "psi"]

    for k,f in enumerate(funcs):
        angle = f( traj, periodic = False )
        angm, residues  = extract_methyls( angle, table )
        save_hist( residues, outdir + "residues_" + outs[k] )
        save_hist( angm, outdir + outs[k] )

if __name__ = "__main__":
    main()