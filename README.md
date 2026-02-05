<h1>CODE FOR: "COMPUTATIONAL DESIGN OF HIGH AFFINITY PEPTIDE MODULATORS FOR AMPAR AND NMDAR"</h1>
<img width="1100" height="1125" alt="image" src="https://github.com/user-attachments/assets/f75a48c1-2511-4f1f-a869-f8bbbb996a82" />
This script generates peptide sequences targeting a receptor and evaluates them using AlphaFold-based structural predictions. It uses a GAN-like generator to propose peptides and computes a distance-based loss between peptide and receptor.

---

## Requirements
```
- PyTorch
- NumPy
- MDAnalysis
- ColabFold
```
Install dependencies using pip:

```
pip install torch numpy mdanalysis colabfold
```
## Usage

Run the script from the terminal as follows:
```
python train_gan_af.py --rec_seq <RECEPTOR_FASTA> [OPTIONS]
```
Required Arguments
```
--rec_seq : Path to a FASTA file containing the receptor sequence.
```
Optional Arguments
```
Argument	Default	Description
--epochs	        50	Number of training epochs
--num_seqs	        20	Number of peptide sequences generated per epoch
--peptide_length	20	Length of each peptide
--output	        "my_results.txt"	Output file to save loss and predicted sequences
--hotspots	        "21,33,44"	Optional hotspots selection
```
## Example
```
python train_gan_af.py \
    --rec_seq receptor.fasta \
    --epochs 100 \
    --num_seqs 30 \
    --peptide_length 15 \
    --output results.txt
```

This will:

Load the receptor sequence from receptor.fasta.

Generate 30 peptide sequences per epoch, each 15 amino acids long.

Evaluate peptide-receptor interactions via AlphaFold predictions.

Save results and losses to results.txt.

## Notes

Make sure your FASTA file contains only canonical amino acids.

The script may take significant time depending on the number of sequences and epochs, as AlphaFold predictions are computationally intensive.

Optional hotspots can be used to guide peptide design, if supported by the GAN model.
