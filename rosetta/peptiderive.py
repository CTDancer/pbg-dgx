import pyrosetta; pyrosetta.init()
from pyrosetta import *
init()

from pyrosetta.rosetta.protocols.peptide_deriver import *
xml_string = """
<ROSETTASCRIPTS>
    <FILTERS>
        <PeptideDeriver name="peptiderive"
        restrict_receptors_to_chains="{restrict_receptors_to_chains}"
        restrict_partners_to_chains="{restrict_partners_to_chains}"
        pep_lengths="{peptide_lengths}"
        dump_peptide_pose="true"
        dump_report_file="true"
        dump_prepared_pose="true"
        dump_cyclic_poses="true"
        skip_zero_isc="true"
        do_minimize="true"
        report_format="markdown" />
    </FILTERS>
    <PROTOCOLS>
        <Add filter_name="peptiderive"/>
    </PROTOCOLS>
</ROSETTASCRIPTS>
"""
peptide_length = "7,8,9,10,11,12,13,14,15"
receptor = "A"
partner = "B"
xml_protocol = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(xml_string.format(restrict_receptors_to_chains=receptor,restrict_partners_to_chains=partner,peptide_lengths=peptide_length))

peptiderive_from_xml_protocol = xml_protocol.get_mover("ParsedProtocol")

from pathlib import Path
import os
import glob

folder_path = Path("/home/tc415/localcolabfold/outputs/CCR9_4_pdbs/")
files = list(folder_path.glob("*_relaxed_rank_001_*.pdb"))
files = [os.path.join(folder_path, f.name) for f in files]
# import pdb
# pdb.set_trace()

for f in files:
    pose = pose_from_file(f)
    peptiderive_from_xml_protocol.apply(pose)

    directory = os.getcwd()
    files_to_remove = glob.glob(os.path.join(directory, '*.pdb'))
    for file in files_to_remove:
        os.remove(file)

    os.rename('/home/tc415/rosetta/peptiderive.txt', f"/home/tc415/rosetta/peptiderive/CCR9_4/{f.split('/')[-1].split('_')[0]}_peptiderive.txt")