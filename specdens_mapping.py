import subprocess
import glob
import os
import mdtraj as md
import numpy as np
import sys
import pickle as pkl
import _pickle as cPickle
import bz2
import re
import argparse
from math import log
from lmfit import minimize, Parameters
from scipy import optimize


########## Functions ##########

def mkdir( directory ):
    if not os.path.isdir(directory):
        os.mkdir( directory )

def load( inp ):
    pin = open( inp, "rb" )
    return pkl.load( pin )

def load_bz2_pkl(inp):
    data = bz2.BZ2File(inp, 'rb')
    data = cPickle.load(data)
    return data

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
    
def LS_simple( params, x, data ):
    a = params[ 'a' ].value
    b = params[ 'b' ].value
    model = np.exp( -x / a ) * b
    return model - data

def LS_extended( params, x, data ):
    a = params[ 'a' ].value
    b = params[ 'b' ].value
    c = params[ 'c' ].value
    d = params[ 'd' ].value
    model = np.exp( -x / a ) * ( b + ( d - b) * np.exp( -x / c ) )
    return model - data

def save( outfile, results ):
    with open( outfile + ".pkl", "wb" ) as fp:
        pkl.dump( results, fp )

def fit_tcf_int( p0, S2, bounds, times, methyl ):
#     print(p0, S2, times, methyl, bounds)
    fitfunc = lambda p, x: S2 + p[0] * np.exp( -x / p[5] ) + p[1] * np.exp( -x / p[6] ) + p[2] * np.exp( -x / p[7] ) + p[3] * np.exp( -x / p[8] ) + p[4] * np.exp( -x / p[9] ) + ( 1 - S2 - p[0] - p[1] - p[2] - p[3] - p[4] ) * np.exp( -x / p[10] ) # Target function
    errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function
    p1 = optimize.least_squares(errfunc, p0, bounds=bounds, args=(times, methyl))
    
    return p1

def get_tau( tauR, ct_f, S2 ):
    pico = 1.e-12 #picoseconds
    
    tauR = np.round( tauR * pico * 1000, decimals = 15 ) #tauR is in nanoseconds, so we multiply it by pico*1000 to get it in seconds
    taus = ct_f.x[5:11] * pico # time constants from fit are in picoseconds
    taureds = 1. / ( 1./tauR + 1./taus ) # effective time constants
    w6 = 1 - S2 - ct_f.x[0] - ct_f.x[1] - ct_f.x[2] - ct_f.x[3] - ct_f.x[4] # last prefactor of target function (Attention: Can be negative!!!)
    w = np.append( ct_f.x[0:5], w6 ) # all prefactors of target function
    return tauR, taureds, w

def J( freq, S2, tauR, taureds, w ): # Eq. XXXX
    tmp = S2 * tauR / (1. + (freq * tauR) ** 2)
    
    for i in range(6):
        tmp += w[i] * taureds[i] / (1 + ( freq * taureds[i] ) ** 2 )
    return  2./5. * tmp

def R1( J_0, J_wD, J_2wD, pre ): # Eq. XXXX
    return 3./16. * pre ** 2 * ( 0 * J_0 +  1 * J_wD + 4 * J_2wD )

def R2( J_0, J_wD, J_2wD, pre ): # Eq. XXXX
    return 1./32. * pre ** 2 * ( 9 * J_0 + 15 * J_wD + 6 * J_2wD )

def R3( J_0, J_wD, J_2wD, pre ): # Eq. XXXX                        
    return 3./16. * pre ** 2 * ( 0 * J_0 +  3 * J_wD + 0 * J_2wD )

def R4( J_0, J_wD, J_2wD, pre ): # Eq. XXXX
    return 3./16. * pre ** 2 * ( 0 * J_0 +  1 * J_wD + 2 * J_2wD )

def calc_Relax( JomegaD, pre ):
    r1 = R1( JomegaD[0], JomegaD[1], JomegaD[2], pre ) 
    r2 = R2( JomegaD[0], JomegaD[1], JomegaD[2], pre ) 
    r3 = R3( JomegaD[0], JomegaD[1], JomegaD[2], pre )
    r4 = R4( JomegaD[0], JomegaD[1], JomegaD[2], pre )
    rates = [r1, r2, r3, r4]
    return rates

# using window average to estimate S2??
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def load_and_concat( infile ):
    
    def load( inp ):
        pin = open( inp, "rb" )
        return pkl.load( pin )
    
    arr = []
    for f in glob.glob( infile ):
        arr.append( load(f) )
    arr = np.concatenate( arr, axis = -1 )
    
    return arr
# For sorting:
def tryint(s):
        try:
            return int(s)
        except ValueError:
            return s
        
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
        l.sort(key = alphanum_key)
        return l
    
def sort_methyls( array, label_list ): # Sorts a list (methyl names) in natural order
    
    def tryint(s):
        try:
            return int(s)
        except ValueError:
            return s
     
    def alphanum_key(s):
         return [ tryint(c) for c in re.split('([0-9]+)', s) ]
    
    label_list_sort = label_list[:]
    sort_nicely( label_list_sort )
    array_dict = dict( zip(label_list, array) )
    temp_list  = []
    for l in label_list_sort:
        temp_list.append( array_dict[l] )
        
    return np.array(temp_list), label_list_sort

def sort_arr( array, label_list ): # Sorts an array based the sorting of another list
    arr_new = []
    for r in range( len( array ) ):
        arr_temp, all_sorted = sort_methyls( array[r], label_list )
        arr_new.append( arr_temp )
    
    return np.array( arr_new )

def parse():
    
    parser = argparse.ArgumentParser( description = '' )

    # Mandatory Inputs
    parser.add_argument( '--quadric', type = str, required = True,
                        help = 'Path to the QUADRIC Diffusion executable.' )
    parser.add_argument( '--pdbinertia', type = str, required = True,
                        help = 'Path to the PDBInertia executable.' )
    parser.add_argument( '--in', type = str, required = True,
                        help = 'Path to input directory.' )
    parser.add_argument( '--b', type = int, required = False, default = 0,
                        help = 'Initial time (in ps) to be used in the trajectory (default: 0 ps).' )
    parser.add_argument( '--e', type = int, required = False, default = 1000000,
                        help = 'Final time (in ps) to be used in the trajectory (default: 1000000 ps, corresponding to 1us at 1ps stride). In case the trajectories have different lengths, this has to be the maximum length of the shortest one.' )
    parser.add_argument( '--lblocks_bb', type = int, required = True,
                        help = 'Length of the blocks (in ps) employed to estimate the backbone tumbling time. Rule of thumb: ~50x the expected tumbling time (if you ran compute_tcfs.py use the same value you specified there).' )
    parser.add_argument( '--lblocks_m', type = int, required = True,
                        help = 'Length of the blocks (in ps) employed to compute the methyl time correlation functions (suggested: 10000 ps. If you ran compute_tcfs.py use the same value you specified there). These will provide the blocks for final reweighting.' )
    parser.add_argument( '--stau', type = int, required = True,
                        help = 'Initial guess for the backbone tumbling time (in ps).' )
    parser.add_argument( '--trajname', type = str, required = True,
                        help = 'Common prefix of the trajectories (Ex. sim for sim1.xtc, sim2.xtc etc.).' )
    parser.add_argument( '--ct_lim', type = float, required = True,
                        help = 'Portion of the methyl time correlation function to be kept to estimate long time limit S2 (Ex. 1.5 means the last third is used, 2 means the last half is used).' )
    parser.add_argument( '--wD', type = float, required = True,
                        help = 'Larmor frequency of 2H at the used magnetic field strength in MHz (Ex. 145.858415 for 2H at 950 MHz magnetic field strength).' )
    parser.add_argument('--tumbling', type = str, required = False, default = '', 
                       help = 'Path to the file with user defined backbone tumbling times. If specified, the calculation of the backbone tumbling times will be skipped. Use --gen_mlist to get the ordered list of methyls.')
    parser.add_argument('--gen_mlist', action = 'store_true', required = False, default = False, help = 'Get the ordered list of methyls. The rest of the code will not be executed.')

    args = parser.parse_args()
    
    return args.quadric, args.pdbinertia, args.b, args.e, args.lblocks_bb, args.lblocks_m, args.stau, args.trajname, args.ct_lim, args.wD, args.tumbling, args.gen_mlist


##########################
########## MAIN ##########
##########################

quadric, pdbinertia, in_dir, b_frame, e_frame, l_blocks_bb, l_blocks_methyl, start_taum, trajname, ct_lim, wD, tumbling, gen_mlist = parse()

print( f"# The following parameters were chosen:\n quadric: {quadric}\n pdbinertia: {pdbinertia}\n b: {b_frame}, e: {e_frame}\n BB block length: {l_blocks_bb} ps, Methyl block length: {l_blocks_methyl} ps\n starting tauM: {start_taum} ps\n Common trajectory name: {trajname}\n C(t) limit: {ct_lim}\n OmegaD: {wD} MHz\n" )

if gen_mlist:
    ''' Get list with methyl names '''
    
    print("# Generating methyl list.")
    mkdir( f'{in_dir}/tau' )
    methyls_carbons = { 'ALA': ['CB'], 'VAL': ['CG1', 'CG2'], 'THR': ['CG2'], 'ILE': ['CG2', 'CD1'], 'LEU': ['CD1', 'CD2'], 'MET': ['CE'] }
    
    struct          = md.load(f'{in_dir}/initial.pdb')
    topology        = struct.topology
    table, bonds    = topology.to_dataframe()
    
    with open( f'{in_dir}/tau/tauR_methyl_specific', 'w' ) as f: # Creates a file listing the methyl names
        for res in methyls_carbons.keys():
            mtable = table[table['resName'] == res]

            for mc in methyls_carbons[res]:
                tmp = list( mtable.loc[mtable['name'] == mc, 'serial'] )

                for c in tmp:

                    a1 = list(mtable.loc[mtable['serial'] == c, 'resName'])
                    a2 = list(mtable.loc[mtable['serial'] == c, 'resSeq'])
                    a3 = list(mtable.loc[mtable['serial'] == c, 'name'])
                    a4 = list(mtable.loc[mtable['serial'] == c+1, 'name'])
                    f.write( f'{a1[0]}{a2[0]}-{a3[0]}{a4[0][:-1]}\n' ) # Ex. format: ALA41-CBHB
    print("# DONE!")
    sys.exit()

len_traj        = e_frame - b_frame
n_blocks_bb     = int( len_traj / l_blocks_bb )
n_blocks_methyl = int( len_traj / l_blocks_methyl )
nh_res          = load(f'{in_dir}/nh_residues.pkl')
nh_count        = len( nh_res )
tmp             = glob.glob(f'{in_dir}/tcf_methyl/*.pbz2')
ntraj           = int( len( tmp ) )
# Larmor frequency of deuterium
wD *= 2. * np.pi * 1000000

''' Parse bb NH TCFs into np.array '''

if tumbling == '':
#     tmp = glob.glob('tcf_bb/*.xvg')
    
    bb_pkls = sort_nicely( glob.glob(f'{in_dir}/tcf_bb/{trajname}*_prot_nopbc.pbz2') )
    for trj in range( ntraj ): 
        tcfs_bb = []
        
        print(f"# Analysing trajectory {trj+1}/{ntraj}")
        print( "   # Load backbone NH time correlation functions")
        tcfs_bb = load_bz2_pkl( bb_pkls[trj] )        

        ''' Fit bb NH TCFS to LS models '''

        # Starting parameters
        simlength  = l_blocks_bb
        fit_length = int(l_blocks_bb / 2)
        accuracy   = 100

        delta_taum = start_taum
        min_taum   = start_taum - delta_taum
        max_taum   = start_taum + delta_taum

        start_S    = 0.8 
        min_S      = 0
        max_S      = 1

        start_taue = 50
        min_taue   = 5
        max_taue   = 100

        start_Sf   = 1.0
        min_Sf     = 0
        max_Sf     = 1

        t          = np.arange(0, fit_length+1, 1)
        maxlogtime = int( accuracy * np.round( np.log( simlength ) ) )
        tmp        = [ np.exp( i / accuracy ) for i in np.linspace( 1, maxlogtime, maxlogtime ) ] # exp(maxlogtime/accuracy) ~ simlength; exponentially distributed time points
        exp_t      = np.unique( [ int (np.round( tmp[j]) ) for j in range( len(tmp) ) if [ i < fit_length for i in tmp ][j] ] ) # exponentially distributed time points < fit_length

        print("   # Fit backbone time correlation function to LS")
        tau_Ms = []
        for bl in range( n_blocks_bb ):

            tau_tmp = []
            for tcf in range(1, nh_count+1 ):

                params = Parameters()
                params.add( 'a', value = start_taum, min = min_taum, max = max_taum )
                params.add( 'b', value = start_S,    min = min_S,    max = max_S )
                params.add( 'c', value = start_taue, min = min_taue, max = max_taue )
                params.add( 'd', value = start_Sf,   min = min_Sf,   max = max_Sf )

                ct = [ tcfs_bb[bl][ i,tcf ] for i in exp_t ]
                try:
                    result   = minimize( LS_simple, params, args = ( exp_t,ct ) )
                    popt1    = np.zeros(2)
                    popt1[0] = result.params['a'].value
                    popt1[1] = result.params['b'].value
                except:
                    popt1    = [0,0]
                try:
                    result   = minimize( LS_extended, params, args = ( exp_t,ct ) )
                    popt2    = np.zeros(4)
                    popt2[0] = result.params['a'].value
                    popt2[1] = result.params['b'].value
                    popt2[2] = result.params['c'].value
                    popt2[3] = result.params['d'].value
                except:
                    popt2    = [ 0, 0, 0, 0 ]

                ct_fit1 = np.exp( -t / popt1[0] ) *   popt1[1]
                ct_fit2 = np.exp( -t / popt2[0] ) * ( popt2[1] + ( popt2[3] - popt2[1] ) * np.exp( -t / popt2[2] ) )
                
                # LS Model selection:
                if (( popt2[2] < 200 ) and ( popt2[2] > 0 ) and ( popt2[0] < max_taum - 100 ) and ( popt2[0] > min_taum + 100 ) and (( abs( popt2[0] - start_taum ) > 10 ) or ( abs(popt2[1] - start_S) > 0.1 )) and ( popt2[1] < 1.5 ) and ( popt2[1] > 0 ) and ( popt2[3] < 1.5 ) and ( popt2[3] > 0 )):
                    tau_tmp.append( popt2[0] ) # Selected: Extended LS
                elif (( popt1[0] < max_taum - 100 ) and ( popt1[0] > min_taum + 100 ) and (( abs( popt1[0] - start_taum) > 10 ) or ( abs(popt1[1] - start_S) > 0.1 )) and ( popt1[1] < 1.5 ) and ( popt1[1] > 0 )): 
                    tau_tmp.append( popt1[0] ) # Selected: Simple LS
                else:
    #                 tau_tmp.append( -1 )
                    raise ValueError( f'Invalid fit for NH vector {tcf}.' )
            tau_Ms.append( tau_tmp )

        mkdir( f'{in_dir}/tau' )
        with open( f'{in_dir}/tau/{trajname}{trj}_tauM.pkl', 'wb' ) as fp:
                    pkl.dump( tau_Ms, fp )

    ''' Load tauMs and residue numbers & save tm.avg.input '''

    print("# Averaging tauMs over trajectories and blocks")
    tauMs = []
    for f in glob.glob( f'{in_dir}/tau/*_tauM.pkl' ):
        pin      = open( f, "rb" )
        tauMs.append( pkl.load( pin ) )

    tauMs         = np.concatenate( tauMs, axis = 0 )
    tauM_av       = np.average( tauMs, axis = 0 )
    tauM_std      = np.std( tauMs, axis = 0 )

    tm_input      = np.empty( (len(nh_res), 3) )
    tm_input[:,0] = nh_res
    tm_input[:,1] = tauM_av/1000
    tm_input[:,2] = tauM_std/1000

    np.savetxt(f'{in_dir}/tau/tm.avg.input', tm_input, fmt = '%g')

    ''' Prepare pdb with initial coordinates '''

    print("# Running PDBInertia")
    process = f'{pdbinertia} -r initial.pdb {in_dir}/tau/initial.prot.inertia.pdb > {in_dir}/tau/initial.inertia.output'
    p       = subprocess.Popen( process, shell = True, stdin = subprocess.PIPE, stdout = subprocess.PIPE, universal_newlines = True )
    p.communicate()

    ''' Write QUADRIC input file and execute QUADRIC '''

    print("# Running QUADRIC Diffusion")
    with open( f'{in_dir}/tau/quadric.in', 'w' ) as f:
        f.write( f"# sample control file\n0.8 1.2 10\n1 'N'\n{in_dir}/tau/tm.avg.input\n{in_dir}/tau/initial.prot.inertia.pdb\navg.axial.pdb\navg.anis.pdb\n" )

    process = f'{quadric} {in_dir}/tau/quadric.in > {in_dir}/tau/quadric_log.out'

    p       = subprocess.Popen( process, shell = True, stdin = subprocess.PIPE, stdout = subprocess.PIPE, universal_newlines = True )
    p.communicate()
    os.system(f'mv avg.axial.pdb avg.anis.pdb {in_dir}/tau/') # Quadric has problems saving the output in the correct directory, thus they are moved afterwards

    ''' Calculate distances between C-CH atoms '''

    print("# Calculating distances between C-CH atoms")
    struct          = md.load(f'{in_dir}/initial.pdb')
    axial           = md.load(f'{in_dir}/tau/avg.axial.pdb')
    topology        = struct.topology
    table, bonds    = topology.to_dataframe()

    methyls_carbons = { 'ALA': ['CB'], 'VAL': ['CG1', 'CG2'], 'THR': ['CG2'], 'ILE': ['CG2', 'CD1'], 'LEU': ['CD1', 'CD2'], 'MET': ['CE'] }
    carbons         = { 'ALA': ['CA'], 'VAL': ['CB', 'CB'],   'THR': ['CB'],  'ILE': ['CB', 'CG1'],  'LEU': ['CG', 'CG'], 'MET': ['SD'] }
    xyz             = axial.xyz[0]
    distances_mod   = []
    distances_z     = []

    for res in carbons.keys():
        mtable = table[table['resName'] == res]

        for x,c in enumerate( carbons[res] ):
            mc   = methyls_carbons[res][x]
            tmp  = list( mtable.loc[mtable['name'] == c, 'serial'] ) # preceeding carbon of methyl carbon
            tmp2 = list( mtable.loc[mtable['name'] == mc, 'serial'] ) # methyl carbons

            carbon_indices = []
            for n,i in enumerate(tmp):
                r = - xyz[i-1] + xyz[tmp2[n]-1]
                distances_z.append( r[2] ) # save z component of C-C distance vector
                carbon_indices.append( [i-1, tmp2[n]-1] )

            carbon_indices = np.array(carbon_indices)
            distances_mod.append( md.compute_distances( axial, carbon_indices, periodic = False, opt = False )[0] ) # Computes modules of the distances

    distances_mod = np.concatenate( distances_mod )

    ''' Get list with methyl names '''
    methyls_carbons = { 'ALA': ['CB'], 'VAL': ['CG1', 'CG2'], 'THR': ['CG2'], 'ILE': ['CG2', 'CD1'], 'LEU': ['CD1', 'CD2'], 'MET': ['CE'] }

    methyl_names   = [] # !!! NOT IN THE SAME ORDER AS THE FINAL OUTPUT !!!

    for res in methyls_carbons.keys():
        mtable     = table[table['resName'] == res]

        for mc in methyls_carbons[res]:
            tmp    = list( mtable.loc[mtable['name'] == mc, 'serial'] )

            for c in tmp:
                a1 = list(mtable.loc[mtable['serial'] == c, 'resName'])
                a2 = list(mtable.loc[mtable['serial'] == c, 'resSeq'])
                a3 = list(mtable.loc[mtable['serial'] == c, 'name'])
                a4 = list(mtable.loc[mtable['serial'] == c+1, 'name'])
                methyl_names.append(f'{a1[0]}{a2[0]}-{a3[0]}{a4[0][:-1]}')

    ''' Parsing Diso and Dpar from quadric output '''

    print("# Computing methyl specific tauR")
    with open(f'{in_dir}/tau/quadric_log.out', 'r') as f:
        lines = f.readlines()
        
    # Parses the QUADRIC output file to extract Diso and Dpar/Dper:
    c = 0
    for l in lines:
        if 'Axial' in l:
            axc = c 
            c  += 1
            break
        c += 1

    for l in lines[axc+1:]:
        if 'Jacknife' in l:
            axj = c
            break
        c += 1

    Diso     = float( lines[axj+1].split()[2] )
    DparDper = float( lines[axj+2].split()[1] )

    Dzz = 3 * Diso /( 1+2 / DparDper )
    Dyy = 3 * Diso /( 2+ DparDper )

    # Calculate methyl specific tumbling time tauR according to an axially symmetric tumbling model:
    tauRs     = []
    for i in range( len(distances_z) ): 
        ratio = distances_z[i] / distances_mod[i]
        Y20   = ( 3*ratio**2 - 1 )/2
        Di    = Diso - Y20 * ( Dzz-Dyy )/3
        tauRs.append( ( 1 / ( 6*Di ) )*100 )

    pkl_tauRs = []

    # Write to human readible file:
    with open( f'{in_dir}/tau/tauR_methyl_specific', 'w' ) as f:
        f.write( '#resID tauR[ns]\n' )
        for i,tauR in enumerate( tauRs ):
            pkl_tauRs.append( [methyl_names[i], tauR] )
            f.write( f'{methyl_names[i]} {tauR}\n' )
else:
    print( "# A tumbling file has been provided. The calculation of the backbone tumbling time will be skipped." )
    print( "# Parsing tumbling file." )
    with open( tumbling, 'r' ) as f:
        lines = f.readlines()

    tauRs = []
    for l in lines:
        l = l.split()
        if '#' in l[0]:
            continue
        else:
            tauRs.append( float(l[1]) ) 
        
''' Parse methyl TCFs into np.array '''
methyl_count = len(tauRs)

tcfs_methyl_l  = []

methyl_pkl = sort_nicely( glob.glob(f'{in_dir}/tcf_methyl/{trajname}*_rot_trans.pbz2') )

for trj in range( ntraj ):
    print( f"# Analysing trajectory {trj+1}/{ntraj}" )
    print( "   # Load methyl time correlation functions" )
    tcfs_methyl = load_bz2_pkl( methyl_pkl[trj] )
    
    print("   # Calculating relaxation rates")
    
    shape           = np.shape( tcfs_methyl )
    tcfs_methyl_avg = np.empty( (shape[0], shape[1], int((shape[2]-1)/3+1)) )
    
    # Average the TCFs over the 3 C-H methyl bonds in one methyl group:
    for b in range( shape[0] ):
        k = 1
        tcfs_methyl_avg[b,:,0] = tcfs_methyl[b,:,0]
        for i in range( 1, shape[2], 3 ):
            tmp = np.array([tcfs_methyl[b,:,i], tcfs_methyl[b,:,i+1], tcfs_methyl[b,:,i+2]])
            tcfs_methyl_avg[b,:,k] = np.average( tmp, axis = 0 )
            k  += 1
            
    # Starting parameters:
    simlength  = l_blocks_methyl
    fit_length = int(l_blocks_methyl / 2)
    accuracy   = 80

    t          = np.arange(0, fit_length+1, 1)
    maxlogtime = int( accuracy * np.round( np.log( simlength ) ) )
    tmp        = [ np.exp( i / accuracy ) for i in np.linspace( 1, maxlogtime, maxlogtime ) ] # exp(maxlogtime/accuracy) ~ simlength; exponentially distributed time points
    exp_t      = np.unique( [ int (np.round( tmp[j]) ) for j in range( len(tmp) ) if [ i < fit_length for i in tmp ][j] ] ) # exponentially distributed time points < fit_length
#     exp_t = np.array(exp_titty)
    
    # Initial guess for the parameters
    p0         = [0.5, 0.5, 0.5, 0.5, 0.5, 1., 1., 1., 1., 1., 1.] # A1, A2, A3, A4, A5, tau1. tau2, tau3, tau4, tau5, tau6
    bounds_min = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.] # A1, A2, A3, A4, A5, tau1. tau2, tau3, tau4, tau5, tau6
    bounds_max = [1., 1., 1., 1., 1., np.inf, np.inf, np.inf, np.inf, np.inf, np.inf] # A1, A2, A3, A4, A5, tau1. tau2, tau3, tau4, tau5, tau6
    bounds     = [bounds_min, bounds_max]

    pre = 2 * np.pi * 167000 # Prefector (here: quadrupolar coupling constant CH3 CD3)
    exp_freq = np.array( [i*wD for i in [0, 1, 2]] ) # Frequencies of spectral density used for the fit / usually frequencies measured in NMR
    freq     = np.linspace(0, 2*10**9, 100) 

    # Compute relaxation rates:
    Jws        = []
    JNMRs      = []
    traj_rates = []
    for bl in range( n_blocks_methyl ):
        print(f"   # Block {bl+1}/{n_blocks_methyl}")

        tmp_rates = []
        tmp_JNMRs = []
        tmp_Jws   = []
        for methyl in range(1, methyl_count+1 ):
            ct     = np.array( [ tcfs_methyl_avg[bl, int(i), methyl] for i in exp_t ] )
            ct = np.round(ct, decimals = 5)
            ct_idx = np.where(exp_t == find_nearest(exp_t, 5000/2))[0][0]
            S2 = np.round( np.mean( ct[ct_idx:] ), decimals = 5 )
            #wavg = running_mean(ct,100) # 100 ~ 25% of points in ct
            #S2 = wavg[-1]
            ct_fit          = fit_tcf_int( p0, S2, bounds, exp_t, ct )
            tauR            = tauRs[methyl-1]
            tauR,taureds, w = get_tau( tauR, ct_fit, S2 )
            Jw              = J( freq, S2, tauR, taureds, w )
            JNMR            = np.array([float(J( i, S2, tauR, taureds, w )) for i in exp_freq]) # values at measured frequencies
            rates           = calc_Relax( JNMR, pre )
            tmp_rates.append( rates )
            tmp_JNMRs.append( JNMR )
            tmp_Jws.append( Jw )
            
        traj_rates.append(tmp_rates)
        Jws.append( tmp_Jws )
        JNMRs.append( tmp_JNMRs )

    traj_rates = np.array(traj_rates).T  
    Jws        = np.array( Jws ).T
    JNMRs      = np.array( JNMRs ).T

    mkdir(f'{in_dir}/results')
    save(f'{in_dir}/results/{trj+1}_rates', traj_rates)
    save(f'{in_dir}/results/{trj+1}_J', Jws)
    save(f'{in_dir}/results/{trj+1}_JNMR', JNMRs)

''' Load final results, merge trajectories and order methyls in sequence order '''

print("# Merging final results")
# Merge the arrays (from different trajectories):
rates = load_and_concat( f'{in_dir}/results/*_rates.pkl' )
J     = load_and_concat( f'{in_dir}/results/*_J.pkl' )
JNMR  = load_and_concat( f'{in_dir}/results/*_JNMR.pkl' )

if tumbling == '':
    with open( f'{in_dir}/tau/tauR_methyl_specific', 'r' ) as f:
        lines = f.readlines()
else:
    with open( tumbling, 'r' ) as f:
        lines = f.readlines()
        
methyl_labels = []
for l in lines:
    l = l.split()
    if '#' in l[0]:
        continue
    else:
        methyl_labels.append( l[0] )  s
# Sort the arrays in methyl order:      
rates = sort_arr( rates, methyl_labels )
J     = sort_arr( J, methyl_labels )
JNMR  = sort_arr( JNMR, methyl_labels )

save( f'{in_dir}/results/rates', rates )
save( f'{in_dir}/results/J', J )
save( f'{in_dir}/results/JNMR', JNMR )
save( f'{in_dir}/results/methyls', sort_nicely( methyl_labels ) ) # Save sorted list of methyls

print("# DONE!")