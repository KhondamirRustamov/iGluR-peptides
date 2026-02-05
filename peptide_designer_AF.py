from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import MDAnalysis as mda
import re
from string import ascii_uppercase, ascii_lowercase
from math import sqrt
import time
import warnings
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from colabfold.download import download_alphafold_params, default_data_dir
from colabfold.utils import setup_logging
from colabfold.batch import get_queries, run, set_model_type

from colabfold.colabfold import plot_protein
from pathlib import Path
import argparse

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generator-based peptide design with AlphaFold feedback"
    )
    parser.add_argument("--rec_seq", type=str, required=True,
                        help="FASTA file containing receptor sequence")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_seqs", type=int, default=20)
    parser.add_argument("--peptide_length", type=int, default=20)
    parser.add_argument("--output", type=str, default="Khondamir_results.txt")  
    parser.add_argument("--hotspots", type=str, default="")  
    return parser.parse_args()


# Set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Number of channels in the training images.
nc = 1
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 1000
# Learning rate for optimizers
lr = 0.0001
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 5, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 5 x 5
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 5, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 10 x 10
            nn.ConvTranspose2d( ngf * 4, ngf * 2, (4,3), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 10 x 20
            nn.ConvTranspose2d( ngf * 2, ngf, (4,3), (1,2), (0,1), bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 20 x 40
            nn.ConvTranspose2d( ngf, nc, (4,3), (1,2), (0,1), bias=False),
            # state size. (nc) x 20 x 80
        )

    def forward(self, input):
        return self.main(input)


device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)

# Print the model
print(netG)


# Initialize BCELoss function
criterion = nn.MSELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))



def alphafold_predict(sequence, epoch, item):

  query_sequence = "".join(sequence.split())

  jobname = 'tmp'+'_'+str(epoch)+'_'+str(item)

  with open(f"{jobname}.csv", "w") as text_file:
      text_file.write(f"id,sequence\n{jobname},{query_sequence}")

  queries_path=f"{jobname}.csv"

  # number of models to use
  use_amber = False 
  template_mode = "custom" 
  
  custom_template_path = 'custom_template'
  use_templates = True

  msa_mode = "single_sequence" 
  pair_mode = "unpaired+paired" 
  

  # decide which a3m to use
  if msa_mode.startswith("MMseqs2"):
    a3m_file = f"{jobname}.a3m"
  model_type = "auto"
  num_recycles = 3 
  save_to_google_drive = False 

  dpi = 200
    
  def prediction_callback(unrelaxed_protein, length, prediction_result, input_features, type):
    fig = plot_protein(unrelaxed_protein, Ls=length, dpi=150)

  result_dir="."
  if 'logging_setup' not in globals():
      setup_logging(Path(".").joinpath("log.txt"))
      logging_setup = True

  queries, is_complex = get_queries(queries_path)
  model_type = set_model_type(is_complex, model_type)
  download_alphafold_params(model_type, Path("."))
  run(
      queries=queries,
      result_dir=result_dir,
      use_templates=use_templates,
      custom_template_path=custom_template_path,
      use_amber=use_amber,
      msa_mode=msa_mode,    
      model_type=model_type,
      num_models=1,
      num_recycles=num_recycles,
      model_order=[1],
      is_complex=is_complex,
      data_dir=Path("."),
      keep_existing_results=False,
      recompile_padding=1.0,
      rank_by="auto",
      pair_mode=pair_mode,
      stop_at_score=float(100),
      prediction_callback=prediction_callback,
      dpi=dpi
  )




aa = 'A R N D C E Q G H I L K M F P S T W Y V'
aminoacid_list = aa.split(' ')
a_dict={}
for i, aaa in enumerate(aminoacid_list):
  a_dict[aaa]=i

def read_fasta_sequence(fasta_path):
    """
    Read first sequence from FASTA file.
    """
    sequence = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                continue
            sequence.append(line)

    if not sequence:
        raise ValueError(f"No sequence found in FASTA file: {fasta_path}")

    seq = "".join(sequence).upper()
    return seq


def numpy_to_seq(array, seq_len, sequence, oligomer=1):
  sequence = []
  receptor = sequence
  array = array.cpu().detach().numpy()
  for i in array:
    seq=''
    array_n = i[0]
    for i in array_n.T:
      z = np.argmax(i)
      seq+=aminoacid_list[z]
    seq = seq[:seq_len]
    seq = seq + (':' + receptor)
    sequence.append(seq)
  return sequence


#target_a = np.load('hal.npy')


def take_loss(target_pdb_name, seq_len, epoch, hotspots=''):
  all_loss = [take_loss1(f'{target_pdb_name}{epoch}_{i}_unrelaxed_rank_1_model_1.pdb', hotspots) for i in range(seq_len)]
  true_tensor = [0 for i in range(seq_len)]
  return torch.tensor(true_tensor).float(), torch.tensor(all_loss).float()


def take_loss1(filename, hotspots=''):
    """
    Compute distance between the center of CA atoms of peptide (chain B)
    and selected residues of receptor (chain C) using MDAnalysis.
    """

    # Load structure
    u = mda.Universe(filename)

    # Select CA atoms for peptide (chain B) and receptor (chain C)
    peptide = u.select_atoms("chainID B and name CA")
    receptor_all = u.select_atoms("chainID C and name CA")

    # Select specific receptor residues (as in your original code)
    # Residue indices are 1-based in MDAnalysis; adjust if needed
    selected_indices = [int(i) for i in hotspots.split(',')]  # corresponds to CAS_B indices in original
    receptor = receptor_all[selected_indices]

    # Compute geometric centers
    peptide_center = peptide.positions.mean(axis=0)
    receptor_center = receptor_all.positions.mean(axis=0)

    # Euclidean distance
    loss = np.linalg.norm(peptide_center - receptor_center)
    return loss



def training(args):
    #Training loop
    num_epochs = args.epochs
    num_seqs = args.num_seqs
    peptide_length = args.peptide_length
    rec_seq = args.rec_seq
    hotspots = args.hotspots
    img_list = []
    G_losses = []
    predicted_losses_list = []
    # tables=[]

    print("Starting Training Loop...")
    # For each epoch
    netG.apply(weights_init)
    for epoch in range(num_epochs):
        start = time.time()
        # For each batch in the dataloader
        fake = netG(torch.randn(num_seqs, nz, 1, 1, device=device))
        fake = numpy_to_seq(fake, peptide_length, rec_seq)
        img_list.append(fake)
        for z, x in enumerate(fake):
          alphafold(x, epoch, z)
        print('time:', time.time() - start)
        optimizerG.zero_grad()

        # Calculate G's loss based on this output
        target, pred_loss = take_loss('tmp', num_seqs, epoch, hotspots)
        errG = criterion(target, pred_loss).requires_grad_(True)
        print(errG)
        # Calculate gradients for G
        errG.backward()
        # Update G
        optimizerG.step()

        # Output training stats
        print(f'{epoch + 1}/{num_epochs}, {errG.item()}')
        # tables.append(fake.detach().numpy()[0][0])

        # Save Losses for plotting later
        G_losses.append(errG.item())
        predicted_losses_list.append(pred_loss)

        file = open(args.output, 'w')
        for i, x in enumerate(img_list):
            file.write('\n'+str(G_losses[i])+'\n')

            [file.write(' '+str(c.item())+',') for c in predicted_losses_list[i]]
            file.write('\n')
            for z in x:
                file.write(str(z)+'\n')
        file.close()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()

