# Genetic variation in *Anopheles* RDL genes

## What is this

The scripts and data in this repository can be used to reproduce all analyses from the manuscript [***Resistance to dieldrin* evolution in African malaria vectors is driven by interspecific and interkaryotypic introgression**](https://academic.oup.com/mbe/advance-article/doi/10.1093/molbev/msaa128/5843798) (Grau-Bové et al., MBE 2020).

Genome variation data for this project has been generated as part of the [***Anopheles gambiae* 1000 Genomes Consortium**](https://www.malariagen.net/projects/ag1000g).

## Contents

All main analyses are organised in three ipython notebooks:

* **`karyotype_2La_phase2.ipynb`** to karyotype 2La inversions in Ag1000G phase 2 data, using genotype frequencies and known karyotypes from Phase 1. In theory you can use it to karyotype any other inversion as long as you've got a training set. Output goes to `results_karyotype`, and is used in Figure 5. Some files are also used in the admixture notebooks.
* **`haplotype_analysis_26nov19.ipynb`** can be used to calculate genotype frequencies, build haplotype networks, perfom positive selection scans along the gene & chromosome, and to obtain haplotype alignments. Output goes to `results_haplotype_analysis` and is used in Figures 1-4, 6 and 8).
* **`admixture_h22nov19_296G.ipynb`** and **`admixture_h22nov19_296S.ipynb`**: perform Patterson's D tests of introgression (aka ABBA-BABA test) between various pairs of populations. Output goes to `results_admixture` and is used in Figure 7.

These scripts are available as ipython notebooks (they can be visualised in github) and python scripts.

Downstream analyses of *Rdl* alignments:

* **`alignments_Rdl_haplotype_phylo`**: alignments of haplotypes from the *Ag1000G* dataset, log files from `iqtree` ML phylogenetic analyses, and a R script to create phylogenetic visualisations (`00_phylodist_04tip_17jun19.R`, as in Figure 6).
* **`alignments_Rdl_multisps`**: alignments of *Rdl* (CDS and peptides) from multiple mosquito species (*A. gambiae*, *A. arabiensis*, *A. melas*, *A. merus*, *A. christyi*, *A. epiroticus*, *A. minimus*, *A. culicifacies*, *A. funestus*, *A. stephensi*, *A. maculatus*, *A. farauti*, *A. dirus*, *A. atroparvus*, *A. sinensis*, *A. albimanus*, *A. darlingi*, *Aedes aegypti*, *Aedes albopictus*, and *Culex quinquefasciatus*) and a couple of R scripts to calculate pairwise identity (`pairwise_identity.R`) and dN/dS ratios (`pairwise_dNdS.R`).

Other files & folders:

* **`data`** folder with metadata for the scripts above (sample info, karyotypes, etc.).
* **`scripts_hapclust`** and **`scripts_printtranscripts`**: some helper functions.

## Data

Where is the input data?

* All metadata required is in the `data` folder
* Some accessory scripts are also available in the `scripts_hapclust` and `scripts_printtranscripts` folders
* Genomic variation data **has to be downloaded** from the [Ag1000G project archive](https://www.malariagen.net/projects/ag1000g). These are huge files that don't fit in this repository. Download links for Phase1-AR3 and Phase2-AR1:

```bash
ftp://ngs.sanger.ac.uk/production/ag1000g/phase1/AR3/
ftp://ngs.sanger.ac.uk/production/ag1000g/phase2/AR1/
```

Notes on data download:

* All genome genome variation files you need are **specified at the beginning of each python notebook**. Once you've downloaded them, edit the scripts to point to the relevant files. Variables to be edited are marked with `#### EDIT THIS` comments.
* Data is available for download in various formats (VCFs, zarr, and HDF5). The scripts above use the zarr arrays and HDF5 files, which are highly compressed and very handy to use compared to VCFs. The python scripts require some special libraries to deal with these formats, mostly implemented in the `scikit-allel`, `zarr` and `h5py` libraries (see dependencies below).
* **phased variants** are available under the `haplotype/main` subfolder:

```bash
ftp://ngs.sanger.ac.uk/production/ag1000g/phase2/AR1/haplotypes/main/
ftp://ngs.sanger.ac.uk/production/ag1000g/phase1/AR3/haplotypes/main/
```

* nucleotide **accessibility arrays** in HDF5 format:

```bash
ftp://ngs.sanger.ac.uk/production/ag1000g/phase2/AR1/accessibility/
ftp://ngs.sanger.ac.uk/production/ag1000g/phase1/AR3/accessibility/
```

* other **metadata** files:

```bash
ftp://ngs.sanger.ac.uk/production/ag1000g/phase2/AR1/samples/
ftp://ngs.sanger.ac.uk/production/ag1000g/phase1/AR3/samples/
```

## Dependencies

**Python** notebooks work with Python 3.7.4 and the following libraries, which can all be installed using `conda`:

* numpy 1.17.3
* zarr 2.3.2
* pandas 0.25.3
* scikit-allel, allel 1.2.1
* scikit-learn, sklearn 0.21.3
* h5py 2.10.0
* scipy 1.3.2
* bcolz 1.2.1
* matplotlib 3.1.2
* seaborn 0.9.0
* itertools 7.2.0

**R scripts** work with R 3.6.1 and require the following libraries:

* seqinr 3.4-5
* ape 5.3
* phytools 0.6-60
* pheatmap 1.0.12

If you use these scripts in your own work, please do not forget to cite the relevant packages as well. It's free and it makes everyone happy :)

For example, in R:

```R
> citation("ape")

To cite ape in a publication use:

  Paradis E. & Schliep K. 2018. ape 5.0: an environment for modern
  phylogenetics and evolutionary analyses in R. Bioinformatics 35:
  526-528.

A BibTeX entry for LaTeX users is

  @Article{,
    title = {ape 5.0: an environment for modern phylogenetics and evolutionary analyses in {R}},
    author = {E. Paradis and K. Schliep},
    journal = {Bioinformatics},
    year = {2018},
    volume = {35},
    pages = {526-528},
  }

As ape is evolving quickly, you may want to cite also its version
number (found with 'library(help = ape)' or
'packageVersion("ape")').

```
