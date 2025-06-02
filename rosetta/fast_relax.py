from pyrosetta import *
from pyrosetta.toolbox import cleanATOM
init()

cleanATOM('/home/tc415/muPPIt/pdbs/fold_wt_model_0.pdb')
pose = pose_from_pdb('/home/tc415/muPPIt/pdbs/fold_wt_model_0.clean.pdb')

sf = create_score_function("ref2015")
sf.set_weight(rosetta.core.scoring.atom_pair_constraint, 5)
sf.set_weight(rosetta.core.scoring.dihedral_constraint, 1)
sf.set_weight(rosetta.core.scoring.angle_constraint, 1)

mmap = MoveMap()
mmap.set_bb(True)   
mmap.set_chi(True)   
mmap.set_jump(True) 

relax = rosetta.protocols.relax.FastRelax()
relax.set_scorefxn(sf)
relax.max_iter(200)
relax.dualspace(True)
relax.set_movemap(mmap)

relax.apply(pose)

pose.dump_pdb('/home/tc415/muPPIt/pdbs/fold_wt_model_0.relax.pdb')