# Genetic variation in *Anopheles* RDL genes

## What is this

The scripts in this repo can be used to reproduce all main analyses from ``this manuscript``.

## Contents

All analyses are organised in three scripts:

* **`kartyotype_2La_phase2.ipynb`** to karyotype 2La inversions in Ag1000G phase 2 data, using genotype frequencies and known karyotypes from Phase 1. In theory you can use it to karyotype any other inversion as long as you've got a training set (Figure 5, also used in admixture analyses). Output goes to `karyotype_2La_output`.
* **`haplotype_analysis_28jun19.ipynb`** can be used to calculate genotype frequencies, build haplotype networks, perfom positive selection scans along the gene & chromosome, and to obtain haplotype alignments (Figures 1-4, 6 and 7). Output goes to `haplotype_analysis_output`.
* **`admixture_h28jun19.ipynb`** perform Patterson's D tests of introgression (aka ABBA-BABA test) between various pairs of populations (Figure 8). Output goes to `admixture_analysis_output`.

All three scripts are available as ipython notebooks, executable python scripts, and as HTML-formatted documents (with visual outputs).

Other folders:

* **`alignments_Rdl_mosquitoes`**: alignments and a couple of R scripts to calculate pairwise identity and dN/dS ratios for *Rdl* orthologs across species.
* **`data`** folder with metadata for the scripts above (sample info, karyotypes, etc.).
* **`scripts_hapclust`** and **`scripts_printtranscripts`**: some helper functions.

## Data

Where is the input data?

* All metadata required is in the `data` folder
* Some accessory scripts are also available in the `scripts_hapclust` and `scripts_printtranscripts` folders
* Genomic variation data *has to be downloaded* from Ag1000G in MalariaGEN. Download links for Phase1-AR3 and Phase2-AR1:

```
ftp://ngs.sanger.ac.uk/production/ag1000g/phase1/AR3/
ftp://ngs.sanger.ac.uk/production/ag1000g/phase2/AR1/
```

Notes on data download:

* All genome genome variation files you need are **specified at the beginning of each script**. Once you've downloaded them, edit the scripts to point to the relevant files. Variables to be edited are marked with `#### EDIT THIS` tags.
* Data is available for download in various formats (VCFs, zarr, and HDF5). The scripts above use the zarr arrays and HDF5 files, which are highly compressed and very handy to use compared to VCFs. The python scripts require some special libraries to deal with these formats, mostly implemented in the `scikit-allel`, `zarr` and `h5py` libraries (see dependencies below).
* **phased variants** are available under the `haplotype/main` subfolder:

```
ftp://ngs.sanger.ac.uk/production/ag1000g/phase2/AR1/haplotypes/main/
ftp://ngs.sanger.ac.uk/production/ag1000g/phase1/AR3/haplotypes/main/
```

* nucleotide **accessibility arrays** in HDF5 format:
```
ftp://ngs.sanger.ac.uk/production/ag1000g/phase2/AR1/accessibility/
ftp://ngs.sanger.ac.uk/production/ag1000g/phase1/AR3/accessibility/
```

* other **metadata** files
```
ftp://ngs.sanger.ac.uk/production/ag1000g/phase2/AR1/samples/
ftp://ngs.sanger.ac.uk/production/ag1000g/phase1/AR3/samples/
```


## Dependencies

Everything works with `Python 3.5.5` and the following libraries, which can all be installed using conda (`conda-forge` channel):

* numpy 1.15.2
* zarr 2.2.0
* pandas 0.23.4
* scikit-allel, allel 1.1.10
* scikit-learn, sklearn 0.20.0
* h5py 2.8.0
* scipy 1.1.0
* bcolz 1.2.1
* matplotlib 2.2.2
* seaborn 0.9.0
* itertools
* mlxtend 0.13.0

