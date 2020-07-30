import subprocess
import glob
import os
import mdtraj as md
import numpy as np
import sys
import pickle as pkl
import _pickle as cPickle
import bz2
import argparse

########## Functions ##########
def mkdir( directory ):
    if not os.path.isdir(directory):
        os.mkdir( directory )

def save( outfile, results ):
    with open( outfile + ".pkl", "wb" ) as fp:
        pkl.dump( results, fp, protocol = 4 )
        
def save_bz2( outfile, results ):
    with bz2.BZ2File(outfile + '.pbz2', 'w') as f: 
        cPickle.dump(results, f)

def parse():
    
    parser = argparse.ArgumentParser( description = '' )

    # Mandatory Inputs
    parser.add_argument( '--xtc', type = str, required = True,
                        help = 'Path to the xtc trajectory.')
    parser.add_argument( '--tpr', type = str, required = True,
                        help = 'Path to the tpr file.' )
    parser.add_argument( '--gmx', '-i', type = str, required = True,
                        help = 'Path to the GMX executable.' )
    parser.add_argument( '--out', type = str, required = False, default = 'analysis',
                         help = 'Path to the output directory.')
    parser.add_argument( '--b', type = int, required = False, default = 0,
                        help = 'Initial time (in ps) to be used in the trajectory (default: 0 ps).' )
    parser.add_argument( '--e', type = int, required = False, default = 1000000,
                        help = 'Final time (in ps) to be used in the trajectory (default: 1000000 ps, corresponding to 1us at 1ps stride). In case the trajectories have different lengths, this has to be the maximum length of the shortest one.' )
    parser.add_argument( '--lblocks_bb', type = int, required = True,
                        help = 'Length of the blocks (in ps) employed to estimate the backbone tumbling time. Rule of thumb: ~50x the expected tumbling time.' )
    parser.add_argument( '--lblocks_m', type = int, required = False, default = 10000,
                        help = 'Length of the blocks (in ps) employed to compute the methyl time correlation functions (default: 10000 ps). These will provide the blocks for final reweighting.' )

    args = parser.parse_args()
    
    return args.xtc, args.tpr, args.gmx, args.out, args.b, args.e, args.lblocks_bb, args.lblocks_m
    

########## User input ##########

xtc, tpr, gmx, out, b_frame, e_frame, l_blocks_bb, l_blocks_methyl = parse()

##########################
########## MAIN ##########
##########################

mkdir(f'{out}')
mkdir(f'{out}/xtc/')

""" Prepare trajectory (BB): rm solvent & pbs & center """

print("# Preparing trajectory")

traj_nopbc = f'{out}/xtc/' + xtc.split('/')[-1].rstrip('.xtc') + '_prot_nopbc.xtc'
process    = f'{gmx} trjconv -s {tpr} -f {xtc} -o {traj_nopbc} -center -pbc mol -ur compact -b {b_frame} -e {e_frame}'

p          = subprocess.Popen( process, shell = True, stdin = subprocess.PIPE, stdout = subprocess.PIPE, universal_newlines = True )
newline    = os.linesep
commands   = ['Protein', 'Protein']
p.communicate( newline.join( commands) )

""" Prepare trajectory (Methyls): rm overall tumbling """

print("# Remove overall tumbling")
rot_trans = f'{out}/xtc/' + xtc.split('/')[-1].rstrip('.xtc') + '_rot_trans.xtc'
process   = f'{gmx} trjconv -s {tpr} -f {traj_nopbc} -o {rot_trans} -fit rot+trans'

p         = subprocess.Popen( process, shell = True, stdin = subprocess.PIPE, stdout = subprocess.PIPE, universal_newlines = True )
newline   = os.linesep
commands  = ['Backbone', 'Protein']
p.communicate( newline.join( commands) )

""" Prepare and load pdb with initial coordinates """

initial = f'{out}/initial.pdb'

if os.path.isfile(initial):
    print("# initial.pdb already exists. It won't be re-written.")
else:
    print("# Prepare and load initial coordinates")

    process  = f'{gmx} trjconv -s {tpr} -f {traj_nopbc} -o {initial} -b 0 -e 0'
    print(process)

    p        = subprocess.Popen(process, shell = True, stdin=subprocess.PIPE, stdout = subprocess.PIPE, universal_newlines = True)
    newline  = os.linesep
    commands = ['Protein']
    p.communicate( newline.join( commands ) )

struct       = md.load(initial)
topology     = struct.topology
table, bonds = topology.to_dataframe()

""" Prepare index file: bb N-H bonds """

print("# Write N-H bonds index file")
nh_res   = []
nh_count = 0
with open(f"{out}/nh.ndx", 'w') as f:
    f.write('[ NH ]\n')
    for n,i in enumerate(table['name']):
        if not table['resSeq'][n] == 1: # Exclude 1st residue
            if (i == 'N' or i == 'H') and table['resName'][n] != 'PRO': # get bb N-H residues, exclude PRO
                tmp = table['resSeq'][n]
                if tmp not in nh_res:
                    nh_res.append( tmp )
                nh_count += 1
                f.write( f'{table["serial"][n]} ')
    f.write('\n')
nh_count = int(nh_count/2)
pkl.dump( nh_res, open(f'{out}/nh_residues.pkl', "wb") )
                      
""" Prepare index file: methyl C-H bonds """

print("# Write methyl C-H bonds index file")
methyls_carbons = { 'ALA': ['CB'], 'VAL': ['CG1', 'CG2'], 'THR': ['CG2'], 'ILE': ['CG2', 'CD1'], 'LEU': ['CD1', 'CD2'], 'MET': ['CE'] }

methyl_count = 0                   
with open(f'{out}/methyls.ndx', 'w') as f:
    f.write('[ Methyls ]\n')
    for res in methyls_carbons.keys():
        mtable = table[table['resName'] == res]

        for mc in methyls_carbons[res]:
            tmp = list( mtable.loc[mtable['name'] == mc, 'serial'] )
            methyl_count += len(tmp)
            for c in tmp:
                f.write(f'{c} {c+1} {c} {c+2} {c} {c+3} ')
    f.write('\n')
print(methyl_count)
                        
""" Prepare index file: methyl C-C(H) bonds """
print("# Write methyl C-C bonds index file")
carbons = { 'ALA': ['CA'], 'VAL': ['CB', 'CB'],   'THR': ['CB'],  'ILE': ['CB', 'CG1'],  'LEU': ['CG', 'CG'], 'MET': ['SD'] }

with open(f'{out}/cc.ndx', 'w') as f:
    f.write('[ CarbonCarbon ]\n')
    for res in carbons.keys():
            mtable   = table[table['resName'] == res]

            for x,c in enumerate( carbons[res] ):
                mc   = methyls_carbons[res][x]
                tmp  = list( mtable.loc[mtable['name'] == c, 'serial'] )
                tmp2 = list( mtable.loc[mtable['name'] == mc, 'serial'] )
                for n,i in enumerate(tmp):
                    f.write( f'{i} {tmp2[n]} ')
    f.write('\n')
    
""" Calculate and parse bb N-H TCFs """
len_traj    = e_frame - b_frame
n_blocks_bb = int( len_traj / l_blocks_bb )
diff        = len_traj - n_blocks_bb * l_blocks_bb
                            
if diff > 0:
    print( f'WARNING: The selected block length is not an integer factor of the trajectory length. {diff} frames will be skipped at the end of the trajectory.' )

mkdir(f'{out}/tcf_bb')
traj_nodir_bb = traj_nopbc.split("/")[-1][:-4]

for bl in range( n_blocks_bb ):
    b       = int( bl * l_blocks_bb )
    e       = int( (bl+1) * l_blocks_bb )
    bb_tcf  = f'{out}/tcf_bb/{bl+1}_{traj_nodir_bb}.xvg'
    process = f'{gmx} rotacf -s {tpr} -f {traj_nopbc} -o {bb_tcf} -n nh.ndx -P 2 -d -noaver -b {b} -e {e} -xvg none'

    p        = subprocess.Popen( process, shell = True, stdin = subprocess.PIPE, stdout = subprocess.PIPE, universal_newlines = True )
    newline  = os.linesep
    commands = ['NH']
    p.communicate( newline.join( commands) )

tcfs_bb = []
print( "# Parse backbone NH time correlation functions")
for bl in range( n_blocks_bb ):
    bb_tcf = f'{out}/tcf_bb/{bl+1}_{traj_nodir_bb}.xvg'
    tcf_bb = np.empty( ( int(l_blocks_bb/2 + 1), nh_count+1 ))

    # Converts GROMACS .xvg file into a np.array:
    with open( bb_tcf ) as f:
        time  = []
        for l in f:
            l = l.split()
            if l[0] != '&':
                time.append( float(l[0]) )
            else:
                break

    tcf_bb[:,0] = np.array( time )

    with open( bb_tcf ) as f:

        line = []
        c    = 0
        for l in f:
            l = l.split()
            if l[0] != '&':
                line.append( float(l[1]) )
            else:
                tcf_bb[:,c+1] = np.array( line )
                line = []
                c += 1
    tcfs_bb.append( tcf_bb )
save_bz2( f'{out}/tcf_bb/{traj_nodir_bb}', tcfs_bb )
                            
""" Calculate and parse methyl C-H TCFs """

n_blocks_methyl = int( len_traj / l_blocks_methyl )
diff            = len_traj - n_blocks_methyl * l_blocks_methyl
                            
if diff > 0:
    print( f'WARNING: The selected block length is not an integer factor of the trajectory length. {diff} frames will be skipped at the end of the trajectory.' )

mkdir(f'{out}/tcf_methyl')
traj_nodir_met = rot_trans.split("/")[-1][:-4]

for bl in range( n_blocks_methyl ):
    b          = int( bl * l_blocks_methyl )
    e          = int( (bl+1) * l_blocks_methyl )
    methyl_tcf = f'{out}/tcf_methyl/{bl+1}_{traj_nodir_met}.xvg'
    process = f'{gmx} rotacf -s {tpr} -f {rot_trans} -o {methyl_tcf} -n methyls.ndx -P 2 -d -noaver -b {b} -e {e} -xvg none'
    
    p          = subprocess.Popen( process, shell = True, stdin = subprocess.PIPE, stdout = subprocess.PIPE, universal_newlines = True )
    newline    = os.linesep
    commands   = ['Methyls']
    p.communicate( newline.join( commands) )
                        
print( "   # Parse methyl time correlation functions" )
tcfs_methyl_l  = []
for bl in range( n_blocks_methyl ):
    methyl_tcf = f'{out}/tcf_methyl/{bl+1}_{traj_nodir_met}.xvg'
    tcf_methyl = np.empty( ( int(l_blocks_methyl/2 +1), methyl_count*3 +1 ))

    with open( methyl_tcf ) as f:
        time   = []
        for l in f:
            l  = l.split()
            if l[0] != '&':
                time.append( float(l[0]) )
            else:
                break
    tcf_methyl[:,0] = np.array( time )
    with open( methyl_tcf ) as f:

        line  = []
        c     = 0
        for l in f:
            l = l.split()
            if l[0] != '&':
                line.append( float(l[1]) )
            else:
                tcf_methyl[:,c+1] = np.array( line )
                line = []
                c += 1
    tcfs_methyl_l.append( tcf_methyl )
tcfs_methyl = np.array(tcfs_methyl_l)
save_bz2( f'{out}/tcf_methyl/{traj_nodir_met}', tcfs_methyl )  # save one array containing all methyl TCFs from one trajectory

""" Delete gromacs xvg files to save space and lower file count """       

print( "   # Delete .xvg files" )
if os.path.exists( f'{out}/tcf_bb/{traj_nodir_bb}.pbz2'  ):
    for f in glob.glob(f'{out}/tcf_bb/*_{traj_nodir_bb}.xvg'):
        os.remove( f )
                        
elif os.path.exists( f'{out}/tcf_methyl/{traj_nodir_met}.pbz2'  ):
    for f in glob.glob(f'{out}/tcf_methyl/*_{traj_nodir_met}.xvg'):
        os.remove( f )
                        
else:
    print( "   # No files deleted." )

print("# DONE!")
