from alphafold.relax import relax
from alphafold.relax import utils
from alphafold.common import protein as p_cf

from colabdesign.af.alphafold.common import protein
from colabdesign.shared.protein import renum_pdb_str
from colabdesign.af.alphafold.common import residue_constants

import pdbfixer
import numpy as np
import jax

def rank_array(input_array):
    # numpy.argsort returns the indices that would sort an array.
    # We convert it to a python list before returning
    return list(np.argsort(input_array))

def rank_and_write_pdb(af_model, name, write_all=False, renum_pdb = True):
   
    ranking = rank_array(af_model.aux['all']['loss'])
    if write_all != True:
        ranking = [ranking[0]]
    
    aux = af_model._tmp["best"]["aux"]
    aux = aux["all"]
    
    p = {k:aux[k] for k in ["aatype","residue_index","atom_positions","atom_mask"]}
    p["b_factors"] = 100 * p["atom_mask"] * aux["plddt"][...,None]
    
    def to_pdb_str(x, n=None):
        p_str = protein.to_pdb(protein.Protein(**x))
        p_str = "\n".join(p_str.splitlines()[1:-2])
        if renum_pdb: p_str = renum_pdb_str(p_str, af_model._lengths)
        if n is not None:
            p_str = f"MODEL{n:8}\n{p_str}\nENDMDL\n"
        return p_str

    m=1
    for n in ranking:
        p_str = ""
        p_str += to_pdb_str(jax.tree_map(lambda x:x[n],p), n+1)
        p_str += "END\n"
    
        with open(name + '_model_{n}_rank_{m}.pdb'.format(n=n, m=m), 'w') as f:
            f.write(p_str)
        m+=1
        
MODRES = {'MSE':'MET','MLY':'LYS','FME':'MET','HYP':'PRO',
          'TPO':'THR','CSO':'CYS','SEP':'SER','M3L':'LYS',
          'HSK':'HIS','SAC':'SER','PCA':'GLU','DAL':'ALA',
          'CME':'CYS','CSD':'CYS','OCS':'CYS','DPR':'PRO',
          'B3K':'LYS','ALY':'LYS','YCM':'CYS','MLZ':'LYS',
          '4BF':'TYR','KCX':'LYS','B3E':'GLU','B3D':'ASP',
          'HZP':'PRO','CSX':'CYS','BAL':'ALA','HIC':'HIS',
          'DBZ':'ALA','DCY':'CYS','DVA':'VAL','NLE':'LEU',
          'SMC':'CYS','AGM':'ARG','B3A':'ALA','DAS':'ASP',
          'DLY':'LYS','DSN':'SER','DTH':'THR','GL3':'GLY',
          'HY3':'PRO','LLP':'LYS','MGN':'GLN','MHS':'HIS',
          'TRQ':'TRP','B3Y':'TYR','PHI':'PHE','PTR':'TYR',
          'TYS':'TYR','IAS':'ASP','GPL':'LYS','KYN':'TRP',
          'CSD':'CYS','SEC':'CYS'}

def pdb_to_string(pdb_file, chains=None, models=[1]):
  '''read pdb file and return as string'''

  if chains is not None:
    if "," in chains: chains = chains.split(",")
    if not isinstance(chains,list): chains = [chains]
  if models is not None:
    if not isinstance(models,list): models = [models]

  modres = {**MODRES}
  lines = []
  seen = []
  model = 1
  for line in open(pdb_file,"rb"):
    line = line.decode("utf-8","ignore").rstrip()
    if line[:5] == "MODEL":
      model = int(line[5:])
    if models is None or model in models:
      if line[:6] == "MODRES":
        k = line[12:15]
        v = line[24:27]
        if k not in modres and v in residue_constants.restype_3to1:
          modres[k] = v
      if line[:6] == "HETATM":
        k = line[17:20]
        if k in modres:
          line = "ATOM  "+line[6:17]+modres[k]+line[20:]
      if line[:4] == "ATOM":
        chain = line[21:22]
        if chains is None or chain in chains:
          atom = line[12:12+4].strip()
          resi = line[17:17+3]
          resn = line[22:22+5].strip()
          if resn[-1].isalpha(): # alternative atom
            resn = resn[:-1]
            line = line[:26]+" "+line[27:]
          key = f"{model}_{chain}_{resn}_{resi}_{atom}"
          if key not in seen: # skip alternative placements
            lines.append(line)
            seen.append(key)
      if line[:5] == "MODEL" or line[:3] == "TER" or line[:6] == "ENDMDL":
        lines.append(line)
  return "\n".join(lines)

def relax_me(pdb_in, pdb_out):
  pdb_str = pdb_to_string(pdb_in)
  protein_obj = p_cf.from_pdb_string(pdb_str)
  amber_relaxer = relax.AmberRelaxation(
    max_iterations=0,
    tolerance=2.39,
    stiffness=10.0,
    exclude_residues=[],
    max_outer_iterations=3,
    use_gpu=False)
  relaxed_pdb_lines, _, _ = amber_relaxer.process(prot=protein_obj)
  with open(pdb_out, 'w') as f:
      f.write(relaxed_pdb_lines)
        
        
def rank_array_predict(input_array):
    # numpy.argsort returns the indices that would sort an array.
    # We convert it to a python list before returning
    return list(np.argsort(input_array))[::-1]

def rank_and_write_pdb_predict(af_model, name, write_all=False, renum_pdb = True):
    ranking = rank_array_predict(np.mean(af_model.aux['all']['plddt'],-1))
    if write_all != True:
        ranking = [ranking[0]]
    
    aux = af_model.aux
    aux = aux["all"]
    
    p = {k:aux[k] for k in ["aatype","residue_index","atom_positions","atom_mask"]}
    p["b_factors"] = 100 * p["atom_mask"] * aux["plddt"][...,None]
    
    def to_pdb_str(x, n=None):
        p_str = protein.to_pdb(protein.Protein(**x))
        p_str = "\n".join(p_str.splitlines()[1:-2])
        if renum_pdb: p_str = renum_pdb_str(p_str, af_model._lengths)
        #if n is not None:
        #    p_str = f"MODEL{n:8}\n{p_str}\nENDMDL\n"
        return p_str

    m=1
    
    pdbs_out = []
    
    for n in ranking:
        p_str = ""
        p_str += to_pdb_str(jax.tree_map(lambda x:x[n],p), n+1)
        p_str += "END\n"

        with open(name + '_model_{n}_rank_{m}.pdb'.format(n=n, m=m), 'w') as f:
            f.write(p_str)
        pdbs_out.append(name + '_model_{n}_rank_{m}.pdb'.format(n=n, m=m))
        m+=1
    return pdbs_out


aa_order = residue_constants.restype_order
order_aa = {b:a for a,b in aa_order.items()}

def get_best_seq(aux):
    x = aux["seq"]["hard"].argmax(-1)
    return ["".join([order_aa[a] for a in s]) for s in x]


def get_best_seq1(aux):
    x = aux["seq"]["hard"].argmax(-1)[-1]
    return ["".join([order_aa[a] for a in s]) for s in x]