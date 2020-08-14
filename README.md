# ABSURDer
**Authors**:  
Simone Orioli&nbsp; &nbsp; &nbsp;| email: simone.orioli@nbi.ku.dk      
Felix K&uuml;mmerer | email: felix.kummerer@bio.ku.dk  
Falk Hoffmann&nbsp; &nbsp;| email: falk.hoffmann@hzg.de   

Python code for calculating and reweighting NMR relaxation rates.

## Dependencies
**Core**
1. Python >= 3.7.x
2. Numpy >= 1.17.0
3. Scipy >= 3.1.1  

**specdens_mapping.py**
1. lmfit >= 1.0.0

**compute_rotamers.py**
1. MdTraj >= 1.9.3

## Required Software
**compute_tcfs.py**
1. GROMACS >= 2018.x (http://manual.gromacs.org/documentation/2018/index.html)

**specdens_mapping.py**
1. Quadric Diffusion >= 1.14 (http://comdnmr.nysbc.org/comd-nmr-dissem/comd-nmr-software/software/quadric-diffusion)
2. PDBInertia (http://comdnmr.nysbc.org/comd-nmr-dissem/comd-nmr-software/software/pdbinertia)

## Scripts
1. `compute_tcfs.py` processes Gromacs trajectories and calculates backbone and methyl time correlation functions.
For help, run `python compute_tcfs.py -h`.
2. `spcdens_mapping.py` computes methyl specific tumbling correlation times and calculates NMR methyl
relaxation rates. For help, run `python specdens_mapping.py -h`.
3. `ABSURDer.py` is a class for the reweighting of NMR relaxation data. For help, refer to the Jupyter Notebook
`absurder_example.ipynb`.
4. `compute_rotamers.py` computes the values of the dihedral angles for methyl bearing residues.
For help, run `python compute_methyl_rotamers.py -h`.

## Example
### compute_tcfs.py
The script requires as input files a trajectory in xtc format and a topology in tpr format. The trajectory is
processed in order to remove the PBCs and roto-translations. The flags `--b` and `--e` can be used to skip
parts of the trajectory and are mostly important only in the case where the trajectories have different lengths.
`--lblocks_bb` and `--lblocks_m` are used to define the block sizes used to compute the backbone and methyl
time correlation functions (TCFs). The script saves, in the output directory, backobone and methyls TCFs, index files
needed to compute such. An example run is provided by the following:
```
python compute_tcfs.py --xtc /path/to/xtc.xtc --tpr /path/to/tpr.tpr --gmx /path/to/gmx --out out_dir \
--b 0 --e 1000000 --lblocks_bb 1000000 --lblocks_m 10000
```

### specdens_mapping.py
This script requires as an input most of the outputs of `compute_tcfs.py`. We suggest therefore to use them
in combination. The script calculates methyl specific tumbling correlation times from the backbone TCFs and
uses them within the spectral density mapping approach to estimate the NMR methyl relaxation rates.
`--lblocks_m`, `--lblocks_m`, `--b` and `--e` should be the same employed
in `compute_tcfs.py`. `--stau` is used as a starting guess for the tumbling correlation time while `--wD`
represrnts the Larmor frequency of 2H at the used magnetic field strangth in MHz. An example run is provided
by the following:
```
python specdens_mapping.py --quadric /path/to/quadric_diffusion --pdbinertia /path/to/pdbinertia --b 0 \
--e 1000000 --lblocks_m 10000 --lblocks_bb 1000000 --stau 10000 --trajname md --wD 145.858415
```
Optionally, a tumbling times file can be provided with `--tumbling`. To use this option, we suggest to use
`--gen_mlist` to generate a template for the tumbling file. The order of the methyl groups in the tumbling
file has not to be modified.

### ABSURDer.py
`ABSURDer.py` is a class for the reweighting of NMR relaxation data. For help, refer to the Jupyter Notebook
`absurder_example.ipynb`.

### compute_rotamers.py
The script uses MDTraj to compute the values of the dihedral angles for methyl bearing residues. The use of
this script is optional and is not required for the calculation of the NMR relaxation rates. An example run
is provided by the following:
```
python compute_rotamers.py --traj /path/to/xtc.xtc --top /path/to/pdb.pdb --out /path/to/outdir \
--stride 10
```

## References
[1] F. K&uuml;mmerer, S. Orioli, D. Harding-Larsen, Y. Gravilov, F. Hoffman, K. Lindorff-Larsen,
 "Fitting side-chain NMR relaxation data using molecular simulations". In preparation.  
[2] F. Hoffmann, F. A. Mulder, L. V. Schäfer, "Predicting NMR relaxation of proteins from molecular dynamics
simula tions with accurate methyl rotation barriers". J. Chem. Phys 152(8), 2020.  
[3] F. Hoffmann, F. A. Mulder, L. V. Schäfer, "Accurate methyl group dynamics in protein simulations
with AMBER force fields". J. Phys. Chem. B, 122(19), 2018.  
[4] F. Hoffmann, M. Xue, L. V. Schäfer, F. A. Mulder, "Narrowing the gap between experimental and
computational determination of methyl group dynamics in proteins". Phys. Chem. Chem. Phys. 20(38), 2018.
