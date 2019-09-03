
# coding: utf-8

# # Admixture in 2L
# 
# ## Input
# 
# Input files:

# In[1]:


# output
outdir   = "admixture_analysis_output"
outcode  = "rdl_admix"

# gene of interest
chrom     = "2L"
loc_start = 25363652 # start gene
loc_end   = 25434556 # end gene

# inversion 2La
inv_start = 20524058 # interval 20524058 - 20528089
inv_end   = 42165532 # interval 42165182 - 42165532

# input data phase2
popc          = "population" 
p2_metasam_fn = "data/samples_p2.meta.txt"
p2_callset_fn = "/home/xavi/Documents/VariationAg1k/data/phase2.AR1/haplotypes/zarr2/ag1000g.phase2.ar1.samples/"      #### EDIT THIS 
p2_accessi_fn = "/home/xavi/Documents/VariationAg1k/data/phase2.AR1/accessibility/accessibility.h5"                    #### EDIT THIS 
p2_popc       = popc
p2_popl       = ["AOcol","BFcol","BFgam","CIcol","CMgam","FRgam","GAgam","GHcol","GHgam","GM","GNcol","GNgam","GQgam","GW","KE","UGgam"]

# outgroup populations
ou_species    = ["arab","quad","meru","mela","epir","chri"]
ou_callset_fl = ["/home/xavi/dades/Variation/phase1.AR3_Fontaine_AltSps/haplotypes/arab_ref_hc_vqsr_cnvrt_sort.zarr/", #### EDIT THIS 
                 "/home/xavi/dades/Variation/phase1.AR3_Fontaine_AltSps/haplotypes/quad_ref_hc_vqsr_cnvrt_sort.zarr/", #### EDIT THIS 
                 "/home/xavi/dades/Variation/phase1.AR3_Fontaine_AltSps/haplotypes/meru_hc_vqsr_cnvrt_sort.zarr/",     #### EDIT THIS 
                 "/home/xavi/dades/Variation/phase1.AR3_Fontaine_AltSps/haplotypes/mela_ref_hc_vqsr_good_cnvrt_sort.zarr/", #### EDIT THIS 
                 "/home/xavi/dades/Variation/phase1.AR3_Fontaine_AltSps/genotypes/epir_fake_cnvrt_sort.zarr/",         #### EDIT THIS 
                 "/home/xavi/dades/Variation/phase1.AR3_Fontaine_AltSps/genotypes/chri_fake_cnvrt_sort.zarr/"]         #### EDIT THIS 
ou_metasam_fl = ["data/samples.metaara.txt",
                 "data/samples.metaqua.txt",
                 "data/samples.metamer.txt",
                 "data/samples.metamel.txt",
                 "data/samples.metaepi.txt",
                 "data/samples.metachr.txt"]
ou_popc       = popc


# accessibility phase2
accessi_fn = "/home/xavi/Documents/VariationAg1k/data/phase2.AR1/accessibility/accessibility.h5" #### EDIT THIS 
# gff
gffann_fn  = "data/Anopheles-gambiae-PEST_BASEFEATURES_AgamP4.9.gff3"


# Libraries:

# In[2]:


import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import bcolz
import scipy
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import zarr
import pandas as pd
import allel
import h5py
import itertools


# Functions:

# In[3]:


# load plot settings
sns.set(context="notebook",style="ticks",
        font_scale=1.3,rc={"lines.linewidth": 1},
        font="Arial",palette="bright")

def remove_indels(gt, ref, alt, pos):
    """This removes indels from the Fontaine data
    https://github.com/SeanTomlinson30/phd-ops/blob/master/218-arabiensis-aim/20181127-fontaine-data-munging.ipynb"""
    # Utility
    mylen = np.vectorize(len)
    # ALT
    count_alt = mylen(alt)
    alt_count = np.sum(count_alt, axis=1)
    # This array returns true at each position if the alt is fixed for the reference or has only one alternate allele.
    is_ref_or_snp = alt_count <= 1
    # REF Let's pull out the positions where it's a single base
    length_of_ref = mylen(ref) <= 1 
    # return the logical and of these 2 arrays this removes the second instance of a duplicated position
    is_not_dup = np.concatenate([[True], np.diff(pos) > 0])
    # this is the final bool array to filter the data with, and results in positions that are not multiallelic not containing indel and not containing duplicated positions
    bool_mask = length_of_ref & is_ref_or_snp & is_not_dup
    return(bool_mask)
    print('DONE')


# ## Load variants
# 
# ### Phase2
# 
# Population and sample structure:

# In[4]:


# load samples list with sample code, groupings, locations etc.
p2_samples_df   = pd.read_csv(p2_metasam_fn, sep='\t')
p2_samples_bool = (p2_samples_df[p2_popc].isin(p2_popl).values)
p2_samples      = p2_samples_df[p2_samples_bool]
p2_samples.reset_index(drop=True, inplace=True)

# indexed dictionary of populations
p2_popdict = dict()
for popi in p2_popl: 
    p2_popdict[popi]  = p2_samples[p2_samples[p2_popc] == popi].index.tolist()

# add an extra population composed of all other locations
p2_popdict["all"] = []
for popi in p2_popl:
    p2_popdict["all"] = p2_popdict["all"] + p2_popdict[popi]


# report
print("Data:")
print("* Samples     = ", p2_samples.shape[0])
print("* Populations = ", set(p2_samples[p2_popc]))
print(p2_samples.groupby(("population")).size())


# Variants and genotypes:

# In[5]:


# declare objects with variant data
p2_callset   = zarr.open(p2_callset_fn)
# variants of genotypes
print("Variants...")
p2_callset_var = p2_callset[chrom]["variants"]
p2_genvars = allel.VariantChunkedTable(p2_callset_var,names=["POS","REF","ALT"],index="POS") 
print(p2_genvars.shape)
# genotype data
print("Genotypes...")
p2_callset_gen = p2_callset[chrom]["calldata"]["genotype"]
p2_genotyp     = allel.GenotypeChunkedArray(p2_callset_gen) 
p2_genotyp     = p2_genotyp.subset(sel1=p2_samples_bool)
print(p2_genotyp.shape)


# #### Outgroups
# 
# Loads one outgroup, removes indels (duplicated variant positions) and subsets phase2 to include variants present in this outgroup. Then, loads outgroup genotypes and subsets them to remove indels and fit phase2. Then, loads the second outgroup and performs the same task. Thus, at each iteration, less and less variants remain (hopefully not too many are lost; worst offenders are `chri` and `epir`).

# In[6]:


oc_genotyp = p2_genotyp
oc_genvars = p2_genvars

for outn,outi in enumerate(ou_species):
    
    print("# p2 genotypes remaining: %i" % oc_genotyp.shape[0])
    
    # Variants
    print("Variants %s..." % outi)
    ou_callset = zarr.open(ou_callset_fl[outn])
    ou_genvars = allel.VariantChunkedTable(ou_callset[chrom]["variants"],names=["POS","REF","ALT"],index="POS")

    # retain positions in inversion & remove indels
    print("Remove indel %s..." % outi)
    pos_noindel = remove_indels(gt = ou_callset[chrom]["calldata"]["GT"], ref = ou_callset[chrom]["variants"]["REF"], alt = ou_callset[chrom]["variants"]["ALT"], pos = ou_genvars["POS"])
    ou_genvars  = ou_genvars[:][pos_noindel]
    
    # subset phase2 to fit phase1
    print("Subset phase2 to %s..." % outi)
    is_p2_in_ou = np.isin(oc_genvars["POS"],test_elements=ou_genvars["POS"])
    oc_genotyp  = oc_genotyp.compress((is_p2_in_ou))
    oc_genvars  = oc_genvars.compress((is_p2_in_ou))
    print(oc_genotyp.shape)
    
    # genotype data
    print("Genotypes %s..." % outi)
    ou_genotyp     = allel.GenotypeChunkedArray(ou_callset[chrom]["calldata"]["GT"]).compress(pos_noindel)
    
    # retain positions available in phase2
    print("Subset %s to phase2..." % outi)
    is_ou_in_p2 = np.isin(ou_genvars["POS"],test_elements=oc_genvars["POS"])
    ou_genotyp  = ou_genotyp.compress(is_ou_in_p2)
    print(ou_genotyp.shape)

    # add new genotypes to phase2
    print("Merge %s into phase2..." % outi)
    oc_genotyp = oc_genotyp.concatenate(ou_genotyp,axis=1)
    print(oc_genotyp.shape)


# ### Merge sample data
# 
# Merge metadata files, with sample codes, species and populations:

# In[7]:


print("Cast sample metadata...")
oc_samples = pd.DataFrame(data={
    "ox_code"    :  p2_samples["ox_code"].values.tolist() + 
                    list(itertools.chain.from_iterable([ pd.read_csv(ou_metasam_fl[n], sep='\t')["ox_code"].values.tolist()  for n,i in enumerate(ou_species) ])),
    "species"    :  p2_samples["m_s"].values.astype(str).tolist() + 
                    list(itertools.chain.from_iterable([ pd.read_csv(ou_metasam_fl[n], sep='\t')["species"].values.tolist()  for n,i in enumerate(ou_species) ])),
    "population" :  p2_samples[p2_popc].values.tolist()   + 
                    list(itertools.chain.from_iterable([ pd.read_csv(ou_metasam_fl[n], sep='\t')[ou_popc].values.tolist()  for n,i in enumerate(ou_species) ]))
})
print(oc_samples.shape)

# rename species...
oc_samples["species"].values[oc_samples["species"].values == "M"]   = "col"
oc_samples["species"].values[oc_samples["species"].values == "S"]   = "gam"
oc_samples["species"].values[oc_samples["species"].values == "M/S"] = "gamcol"
oc_samples["species"].values[oc_samples["species"].values == "M-S"] = "gamcol"
oc_samples["species"].values[oc_samples["species"].values == "nan"] = "gamcol"

# obtain full population & species list
oc_popl = np.unique(oc_samples["population"].values)
oc_spsl = np.unique(oc_samples["species"].values)


# Dictionaries of populations and species:

# In[8]:


print("Population dict...")
oc_popdict = dict()
for popi in oc_popl: 
    oc_popdict[popi]  = oc_samples[oc_samples["population"] == popi].index.tolist()
    
oc_popdict["all"] = []
for popi in oc_popl:
    oc_popdict["all"] = oc_popdict["all"] + oc_popdict[popi]

print("Species dict...")
oc_spsdict = dict()
for spsi in oc_spsl: 
    oc_spsdict[spsi]  = oc_samples[oc_samples["species"] == spsi].index.tolist()


# ### Allele counts
# 
# Using both dictionaries:

# In[9]:


print("Genotypes to allele counts (population)...")
oc_genalco_pop = oc_genotyp.count_alleles_subpops(subpops=oc_popdict)
print(oc_genalco_pop.shape)

print("Genotypes to allele counts (species)...")
oc_genalco_sps = oc_genotyp.count_alleles_subpops(subpops=oc_spsdict)
print(oc_genalco_sps.shape)


# Define which variants to retain from phase2 (all other datasets will be recast to fit this):

# In[10]:


# Filters
# subset data: segregating alleles, biallelic and no singletons
print("Filters p2+ou...")
oc_is_seg      = oc_genalco_pop["all"].is_segregating()[:] # segregating
oc_is_nosing   = oc_genalco_pop["all"][:,:2].min(axis=1)>2 # no singletons

# subset phase2 to seg, nosing, biallelic & outgroup size
oc_genotyp_seg     = oc_genotyp.compress((oc_is_seg & oc_is_nosing))
oc_genvars_seg     = oc_genvars.compress((oc_is_seg & oc_is_nosing))
oc_genalco_pop_seg = oc_genalco_pop.compress((oc_is_seg & oc_is_nosing))
oc_genalco_sps_seg = oc_genalco_sps.compress((oc_is_seg & oc_is_nosing))

# report
print(oc_genotyp_seg.shape,"/", oc_genotyp.shape)


# ### Other input
# 
# Accessibility

# In[11]:


accessi_df  = h5py.File(accessi_fn,mode="r")
accessi_arr = accessi_df[chrom]["is_accessible"][:]    


# Import GFF geneset:

# In[12]:


# import gff
def geneset_gff(geneset):
    items = []
    for n in geneset.dtype.names:
        v = geneset[n]
        if v.dtype.kind == 'S':
            v = v.astype('U')
        items.append((n, v))
    return pd.DataFrame.from_items(items)
get_ipython().run_line_magic('run', '/home/xavi/Documents/VariationAg1k/rdl_notebooks/scripts_printtranscripts/allel_printtranscripts_24gen18.py')

geneset = allel.FeatureTable.from_gff3(gffann_fn,attributes=['ID', 'Parent'])
geneset = geneset_gff(geneset)


# ## Admixture analyses
# 
# ### Patterson $D$
# 
# $D$ statistic or $ABBA-BABA$ test for introgression.
# 
# * C is the donor/acceptor population that might have undergone gene flow with either A or B.
# * A and B are two pops of the same species and we want to see if there was introgression between C and either A or B
# * D is an unadmixed outgroup
# 
# ```
# A   B   C   D
# `---´   ¦   ¦
#   `-----´   ¦
#      `------´
#          ¦ 
# ```
# 
# Interpretation:
# 
# * If **`D~0`**, tree concordance
# * If **`D>0`**, flow between **A and C**
# * If **`D<0`**, flow between **B and C**

# Define parameters for windowed estimation of $D$:

# In[13]:


# window lengths for SNP blocks
block_len_snp = 5000
step_frac_snp = 0.2
step_perc_snp = int(step_frac_snp * 100)
step_len_snp  = int(block_len_snp * step_frac_snp)


# Function to loop through population combinations:

# In[14]:


def loop_D_statistic3(name, popA_list, popB_list, popC_list, popD_list, 
                     popA_ac, popB_ac, popC_ac, popD_ac, 
                     pos,cycle = "C", block_len_snp=block_len_snp, step_len_snp=step_len_snp,color=["blue","darkorange","turquoise","crimson","magenta","limegreen",
                                "forestgreen","slategray","orchid","darkblue"]):
    
    windows_pos = allel.stats.moving_statistic(pos, statistic=lambda v: v[0], size=block_len_snp,step=step_len_snp)

    # calculate pvalues and focus in this region: duplicated region proper
    is_locus = np.logical_and(pos > loc_start,pos < loc_end) # gene region
    is_inv   = np.logical_and(pos > inv_start,pos < inv_end) # inversion region
    
    # loop
    pdf = PdfPages("%s/%s.Dstat_%s.pdf" % (outdir,outcode,name))
        
    colors = cm.rainbow(np.linspace(0, 1, len(popC_list)))

    for dn,popD in enumerate(popD_list):

        for bn,popB in enumerate(popB_list):

            for an,popA in enumerate(popA_list):

                print("(((%s,%s),X),%s) chr" % (popA,popB,popD))

                fig = plt.figure(figsize=(15,3))

                # whole chromosome: frame
                ax1 = plt.subplot(1, 2, 1)
                sns.despine(ax=ax1,offset=10)
                ax1.set_title("Chr %s (((%s,%s),X),%s)" % (chrom,popA,popB,popD))
                ax1.set_xlim(0,50)
                ax1.set_ylim(-1,1)
                ax1.set_xlabel("Mb")
                ax1.set_ylabel("D")
                plt.axhline(0, color='k',linestyle="--",label="")
                plt.axvline(loc_start/1e6, color='red',linestyle=":",label="locus")
                plt.axvline(loc_end/1e6, color='red',linestyle=":",label="")
                plt.axvline(inv_start/1e6, color='orange',linestyle=":",label="inversion")
                plt.axvline(inv_end/1e6, color='orange',linestyle=":",label="")

                ax2 = plt.subplot(1, 4, 3)
                sns.despine(ax=ax2,offset=10)
                ax2.set_xlim(loc_start/1e6-0.5,loc_end/1e6+0.5)
                ax2.set_ylim(-1,1)
                ax2.set_xlabel("Mb")
                ax2.set_ylabel("D")
                plt.axhline(0, color='k',linestyle="--",label="")
                plt.axvline(loc_start/1e6, color='red',linestyle=":",label="locus")
                plt.axvline(loc_end/1e6, color='red',linestyle=":",label="")
                plt.axvline(inv_start/1e6, color='orange',linestyle=":",label="inversion")
                plt.axvline(inv_end/1e6, color='orange',linestyle=":",label="")

                for cn,popC in enumerate(popC_list):

                    if popA != popB:

                        # block-wise patterson D (normalised)
                        admix_pd_n_win = allel.moving_patterson_d( 
                            aca=popA_ac[popA][:,0:2],
                            acb=popB_ac[popB][:,0:2],
                            acc=popC_ac[popC][:,0:2],
                            acd=popD_ac[popD][:,0:2],
                            size=block_len_snp,step=step_len_snp)

                        # whole chromosome: plot
                        plt.subplot(1, 2, 1)
                        plt.step(windows_pos/1e6, admix_pd_n_win, color=colors[cn])

                        # estimated D in locus with pval
                        admix_pd_av_indup = allel.average_patterson_d(
                            aca=popA_ac[popA][:,0:2][is_locus],
                            acb=popB_ac[popB][:,0:2][is_locus],
                            acc=popC_ac[popC][:,0:2][is_locus],
                            acd=popD_ac[popD][:,0:2][is_locus],
                            blen=100)
                        # convert Z-score (num of SD from 0) to pval (two-sided)
                        admix_pd_av_indup_pval = scipy.stats.norm.sf(abs(admix_pd_av_indup[2]))*2 
                        # add results in legend
                        plt.axvline(loc_end/1e6, color='red',linestyle=":",label="")
                        plt.axvline(loc_start/1e6, color='red',linestyle=":",)

                        # zoomed region: plot
                        plt.subplot(1, 4, 3)
                        plt.step(windows_pos/1e6, admix_pd_n_win, color=colors[cn], label="%s\nD = %.3f +/- %.3f | Z = %.3f | p = %.3E" % 
                                     (popC,admix_pd_av_indup[0],admix_pd_av_indup[1],admix_pd_av_indup[2],admix_pd_av_indup_pval))

                plt.axhline(0, color='k',linestyle="--",label="")
                ax2.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

                # save pdf
                pdf.savefig(fig,bbox_inches='tight')

    pdf.close()


# ### Results admixture Patterson's $D$
# 
# #### Introgression & inversion karyotypes
# 
# The 2La inversion blocks recombination between non-concordant karyotypes, and makes genotype frequencies within the inversion to be more similar between specimens sharing the inversion than between specimens from the same species (col, gam, ara...). Thus, introgression must be analysed separately for sub-populations with (2La) and without (2L+a) the inversion.
# 
# First, let's look at **karyotypes of the inversion per population** (produced in notebook `karyotype_2La_phase2.ipynb`)

# In[15]:


kary_fn = "/home/xavi/Documents/VariationAg1k/rdl_notebooks/data/kt_2la.karyotype_with_outgroups.csv"
kary_df = pd.read_csv(kary_fn, sep='\t')
kary_df = kary_df.loc[kary_df['population'].isin(oc_popl)]
print("karyotypes 2La phase2:",kary_df.shape)
kary_df.tail()


# In[16]:


oc_popl_inv = [j+"_"+str(i) for j in oc_popl for i in [0,1,2]]
oc_samples["inv"]    = kary_df["estimated_kt"].values

# indexed dictionary of populations
print("New dict pop...")
oc_samples["pop_inv"] = oc_samples[popc].astype(str) +"_"+ oc_samples["inv"].astype(str)
pop_inv_list = np.unique(oc_samples["pop_inv"].values).tolist()
popdict_inv = dict()
for popi in pop_inv_list: 
    popdict_inv[popi]  = oc_samples[oc_samples["pop_inv"] == popi].index.tolist()
popdict_inv["all"] = []
for popi in pop_inv_list:
    popdict_inv["all"] = popdict_inv["all"] + popdict_inv[popi]
print("New allele counts pop...")
oc_genalco_pop_seg_inv = oc_genotyp_seg.count_alleles_subpops(subpops=popdict_inv)
    
# indexed dictionary of species
print("New dict sps...")
oc_samples["sps_inv"] = oc_samples["species"].astype(str) +"_"+ oc_samples["inv"].astype(str)
sps_inv_list = np.unique(oc_samples["sps_inv"].values).tolist()
spsdict_inv = dict()
for spsi in sps_inv_list: 
    spsdict_inv[spsi]  = oc_samples[oc_samples["sps_inv"] == spsi].index.tolist()
spsdict_inv["all"] = []
for spsi in sps_inv_list:
    spsdict_inv["all"] = spsdict_inv["all"] + spsdict_inv[spsi]
print("New allele counts sps...")
oc_genalco_sps_seg_inv = oc_genotyp_seg.count_alleles_subpops(subpops=spsdict_inv)

# report
print(oc_genalco_pop_seg_inv.shape,oc_genalco_sps_seg_inv.shape)


# In[17]:


oc_samples.head()


# Plot these counts:

# In[18]:


def annotate_barplot(ax,color="k", labformat = "{:.2f}"):
    rects = ax.patches
    for rect in rects:
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2
        space   = 5
        ha      = 'left'
        label   = labformat.format(x_value) ## annotates bars with height labels, with 2 decimal points
        plt.annotate(label, (x_value, y_value), xytext=(space, 0), textcoords="offset points", va='center', ha=ha, color=color)


# In[19]:


fig = plt.figure(figsize=(12,10))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=None)

pdf = PdfPages("%s/%s.inversion_karyotype_freq.pdf" % (outdir,outcode))
for ki in [0,1,2]:
    
    pie_labels = oc_popl
    # 2La counts per pop
    pie_counts = [len(np.where(np.logical_and(oc_samples["inv"].values==ki,oc_samples["population"].values==i))[0])  for i in oc_popl]
    ax = plt.subplot(2, 3, ki+1)
    plt.barh(width=pie_counts[::-1],y=pie_labels[::-1],color="b")
    annotate_barplot(ax=ax, labformat="{:.0f}")
    ax.set_xlim(0,200)
    ax.set_title("counts 2La karyotype %i" % ki)

    # 2La freqs per pop
    pie_counts = [100 * len(np.where(np.logical_and(oc_samples["inv"].values==ki, oc_samples["population"].values==i))[0])/len(np.where(np.logical_and(oc_samples["inv"].values>=0,oc_samples["population"].values==i))[0])  for i in oc_popl]
    ax = plt.subplot(2, 3, 3+ki+1)
    plt.barh(width=pie_counts[::-1],y=pie_labels[::-1],color="b")
    annotate_barplot(ax=ax, labformat="{:.2f}")
    ax.set_xlim(0,100)
    ax.set_title("frq 2La karyotype %i" % ki)

pdf.savefig(fig,bbox_inches='tight')
pdf.close()


# We can now use this information to check our introgression hypotheses:
# 
# * **Introgression of 296S between BFcol <> arab**, which should be specific to the 2 (2La) karyotype, because arab and BFcol are fixed for 2La inversion.
# * **Introgression of 296G between col <> gam**. Is it specific to the 0 (2L+a) karyotype? Does it happen on the 2 (2La) too?

# ### A296S between BFcol_2 <> arab
# 
# #### Inverted background.
# 
# First, we look at **introgression of the 296S cluster between BFcol_2 and *arabiensis***, using inverted and non-inverted gambiae and coluzzii as control B. 
# 
# * A is BFcol_2
# * B are contrasts without 296S but with 2La inversions: GHgam, CMgam, UGgam, CIcol (2) ; I exclude hybrids (GW...)
# * C are putative donors/acceptors with 2La inversions: arab, meru
# * D are inverted outgroups (epir, chri)
# 
# Result: **BFcol Rdl has an introgression peak with arab** when compared against an inverted background. 

# In[20]:


get_ipython().run_cell_magic('capture', '--no-stdout --no-display', 'loop_D_statistic3(\n    name="araBFcol_back2_inv_B2", \n    popA_list=["BFcol_2"], \n    popB_list=["GHgam_2","CMgam_2","UGgam_2","CIcol_2"], \n    popC_list=["arab_2","meru_2"], \n    popD_list=["epir_2","chri_2"],\n    popA_ac=oc_genalco_pop_seg_inv, \n    popB_ac=oc_genalco_pop_seg_inv, \n    popC_ac=oc_genalco_sps_seg_inv, \n    popD_ac=oc_genalco_sps_seg_inv,\n    pos=oc_genvars_seg["POS"][:],\n    cycle="C"\n)')


# #### A296S between BFcol_2 <> arab populations
# 
# Repeat analysis above to see if there is differential introgression between BFcol and any of the *arabiensis* populations. 
# 
# Result: all arab populations them have clear introgression signals in *Rdl*.

# In[21]:


get_ipython().run_cell_magic('capture', '--no-stdout --no-display', 'loop_D_statistic3(\n    name="araBFcol_back2_inv_B2_arapop", \n    popA_list=["BFcol_2"], \n    popB_list=["GHgam_2","CMgam_2","UGgam_2","CIcol_2"], \n    popC_list=["BFara_2","CMara_2","TZara_2","KEmer_2","SAmer_2"], \n    popD_list=["epir_2"],\n    popA_ac=oc_genalco_pop_seg_inv, \n    popB_ac=oc_genalco_pop_seg_inv, \n    popC_ac=oc_genalco_pop_seg_inv, \n    popD_ac=oc_genalco_sps_seg_inv,\n    pos=oc_genvars_seg["POS"][:],\n    cycle="C"\n)')


# ### 296G between gam<>col
# 
# Another possible case of admixture is suggested by the co-occurrence of col and gam in the main component of the 296G haplotype cluster. 
# 
# #### 296G introgression in the non-inverted background 
# 
# First, let's see if this is the case in the non-inverted background (0 or 2L+a haplotypes). This is the most abundant background for 296G haplotypes, so it could be the ancestral one.
# 
# * A are non-inverted gam pops in the 296G cluster (GAgam_0,GHgam_0,GNgam_0,CMgam_0,GQgam_0)
# * B is a gam not in cluster and non-inverted (UGgam_0)
# * C are various cols, some in cluster ("AOcol_0","CIcol_0","GHcol_0") and some not ("GNcol_0"). We expect introgression with cols in cluster, and no introgression with the wt-only col pop.
# * D are non-inverted outgroups: mela and quad
# 
# This demonstrates that there was **introgression of 296G between gambiae and coluzzi, which happened on a non-inverted background**. This occurred somewhere in West Africa, as signal is stronger in WA populations but weaker in CMgam.

# In[22]:


get_ipython().run_cell_magic('capture', '--no-stdout --no-display', 'loop_D_statistic3(\n    name="colgam_clu296G_back0_ABgam_Ccol",\n    popA_list=["CMgam_0","GAgam_0","GHgam_0","GNgam_0","GQgam_0"], \n    popB_list=["UGgam_0"],\n    popC_list=["AOcol_0","CIcol_0","GHcol_0","GNcol_0"], \n    popD_list=["quad_0","mela_0"],\n    popA_ac=oc_genalco_pop_seg_inv, \n    popB_ac=oc_genalco_pop_seg_inv, \n    popC_ac=oc_genalco_pop_seg_inv, \n    popD_ac=oc_genalco_sps_seg_inv,\n    pos=oc_genvars_seg["POS"][:],\n    cycle="C"\n)')


# #### Introgression in the inverted background (`2`)
# 
# Now, let's see if the minority 296G-2La haplotype also has signals of introgression:
# 
# * A are inverted gams in 296G cluster ("GHgam_2","GNgam_2","CMgam_2")
# * B is a gam not in cluster and inverted (UGgam_2)
# * C are various cols, some in cluster ("CIcol_2","GHcol_2") and some not ("BFcol_2")
# * D are non-inverted outgroups: meru, epir, chri; arab is not good because of independent introgression with BFcol_2
# 
# This demonstrates that there was **no introgression of 296G between gambiae and coluzzi in the inverted background**.

# In[23]:


get_ipython().run_cell_magic('capture', '--no-stdout --no-display', 'loop_D_statistic3(\n    name="colgam_clu296G_back2_ABgam_Ccol",\n    popA_list=["BFgam_2","CMgam_2","GHgam_2","GNgam_2"],\n    popB_list=["UGgam_2"],\n    popC_list=["CIcol_2","GHcol_2","BFcol_2"], \n    popD_list=["meru_2","epir_2","chri_2","arab_2"],\n    popA_ac=oc_genalco_pop_seg_inv, \n    popB_ac=oc_genalco_pop_seg_inv, \n    popC_ac=oc_genalco_pop_seg_inv, \n    popD_ac=oc_genalco_sps_seg_inv,\n    pos=oc_genvars_seg["POS"][:],\n    cycle="C"\n)')


# #### Introgression of 296G between karyotypes (`0` and `2`)
# 
# Are there signals of introgression between the 2La and 2L+a haplotypes in populations that also have the 296G allele?
# 
# * A are non-inverted gam pops in the 296G cluster (GAgam_0,GHgam_0,GNgam_0,CMgam_0,GQgam_0)
# * B is non-inverted gam not in cluster (UGgam_0)
# * C are inverted gamcols with 296G cluster ("GHgam_2","GNgam_2","CMgam_2")
# * D are non-inverted OR inverted outgroups (analysed separately)
# 
# This analysis is problematic because `(((A,B),C),D)` is not valid along the entire chromosome, but in the Rdl locus specifically it does hint at 0<>2 introgression specific to 296G-carrying samples.

# In[24]:


get_ipython().run_cell_magic('capture', '--no-stdout --no-display', 'loop_D_statistic3(\n    name="colgam_clu296G_interkaryotype_2back",\n    popA_list=["CMgam_0","GAgam_0","GHgam_0","GNgam_0","GQgam_0","AOcol_0","CIcol_0","GHcol_0"], \n    popB_list=["UGgam_0"],\n    popC_list=["BFgam_2","CMgam_2","GHgam_2","GNgam_2"], \n    popD_list=["meru_2"],\n    popA_ac=oc_genalco_pop_seg_inv, \n    popB_ac=oc_genalco_pop_seg_inv, \n    popC_ac=oc_genalco_pop_seg_inv, \n    popD_ac=oc_genalco_sps_seg_inv,\n    pos=oc_genvars_seg["POS"][:],\n    cycle="C"\n)')


# In[25]:


get_ipython().run_cell_magic('capture', '--no-stdout --no-display', 'loop_D_statistic3(\n    name="colgam_clu296G_interkaryotype_0back",\n    popA_list=["CMgam_0","GAgam_0","GHgam_0","GNgam_0","GQgam_0","AOcol_0","CIcol_0","GHcol_0"], \n    popB_list=["UGgam_0"],\n    popC_list=["BFgam_2","CMgam_2","GHgam_2","GNgam_2"], \n    popD_list=["quad_0","mela_0"],\n    popA_ac=oc_genalco_pop_seg_inv, \n    popB_ac=oc_genalco_pop_seg_inv, \n    popC_ac=oc_genalco_pop_seg_inv, \n    popD_ac=oc_genalco_sps_seg_inv,\n    pos=oc_genvars_seg["POS"][:],\n    cycle="C"\n)')


# #### Plot transcripts in region of interest
# 
# Finally, plot the coordinates of transcripts in the region of interest, so that you can create nice-looking plots:

# In[26]:


pdf = PdfPages("%s/%s.genes.pdf" % (outdir,outcode))
# plot transcripts
fig = plt.figure(figsize=(15,2))
ax4 = plt.subplot(1, 4, 3)
sns.despine(ax=ax4,offset=10)
locus_genel = plot_transcripts(
    geneset=geneset,chrom=chrom,
    start=loc_start-5e5,stop=loc_end+5e5,
    height=0.1,label_transcripts=False,ax=ax4,label_axis=False,
    color="slategray")

pdf.savefig(fig,bbox_inches='tight')
pdf.close()


# ## Differentiation
# 
# Calculate Hudson's $F_{ST}$ along the entire 2L chromosomal arm, to see if there is a drop in differentiation between non-concordant karyotypes at the center of the 2La inversion.
# 
# First, a loop function.

# In[27]:


def loop_Fst(name, popA_list, popC_list, 
             popA_ac, popC_ac,
             pos,cycle = "C", block_len_snp=block_len_snp, step_len_snp=step_len_snp,
             color=["blue","darkorange","turquoise","crimson","magenta","limegreen",
                    "forestgreen","slategray","orchid","darkblue"]):
    
    windows_pos = allel.stats.moving_statistic(pos, statistic=lambda v: v[0], size=block_len_snp,step=step_len_snp)

    # calculate pvalues and focus in this region: duplicated region proper
    is_locus = np.logical_and(pos > loc_start,pos < loc_end) # gene region
    is_inv   = np.logical_and(pos > inv_start,pos < inv_end) # inversion region
    
    # loop
    pdf = PdfPages("%s/%s.Fst_%s.pdf" % (outdir,outcode,name))
        
    colors = cm.rainbow(np.linspace(0, 1, len(popC_list)))

    for an,popA in enumerate(popA_list):

        fig = plt.figure(figsize=(15,3))

        # whole chromosome: frame
        ax1 = plt.subplot(1, 2, 1)
        sns.despine(ax=ax1,offset=10)
        ax1.set_title("Chr %s Fst %s~X" % (chrom,popA))
        ax1.set_xlim(0,50)
        ax1.set_ylim(0,1)
        ax1.set_xlabel("Mb")
        ax1.set_ylabel("Fst")
        plt.axhline(0, color='k',linestyle="--",label="")
        plt.axvline(loc_start/1e6, color='red',linestyle=":",label="locus")
        plt.axvline(loc_end/1e6, color='red',linestyle=":",label="")
        plt.axvline(inv_start/1e6, color='orange',linestyle=":",label="inversion")
        plt.axvline(inv_end/1e6, color='orange',linestyle=":",label="")

        ax2 = plt.subplot(1, 4, 3)
        sns.despine(ax=ax2,offset=10)
        ax2.set_xlim(loc_start/1e6-0.5,loc_end/1e6+0.5)
        ax2.set_ylim(0,1)
        ax2.set_xlabel("Mb")
        ax2.set_ylabel("Fst")
        plt.axhline(0, color='k',linestyle="--",label="")
        plt.axvline(loc_start/1e6, color='red',linestyle=":",label="locus")
        plt.axvline(loc_end/1e6, color='red',linestyle=":",label="")
        plt.axvline(inv_start/1e6, color='orange',linestyle=":",label="inversion")
        plt.axvline(inv_end/1e6, color='orange',linestyle=":",label="")

        for cn,popC in enumerate(popC_list):

            if popA != popC:

                # block-wise patterson D (normalised)
                admix_pd_n_win = allel.moving_hudson_fst( 
                    ac1=popA_ac[popA][:,0:2],
                    ac2=popC_ac[popC][:,0:2],
                    size=block_len_snp,step=step_len_snp)

                # whole chromosome: plot
                plt.subplot(1, 2, 1)
                plt.step(windows_pos/1e6, admix_pd_n_win, color=colors[cn])

                # estimated D in locus with pval
                admix_pd_av_indup = allel.average_hudson_fst(
                    ac1=popA_ac[popA][:,0:2][is_locus],
                    ac2=popC_ac[popC][:,0:2][is_locus],
                    blen=100)
                # zoomed region: plot
                plt.subplot(1, 4, 3)
                plt.step(windows_pos/1e6, admix_pd_n_win, color=colors[cn], 
                         label="%s\nFst = %.3f +/- %.3f" % 
                             (popC,admix_pd_av_indup[0],admix_pd_av_indup[1]))

        plt.axhline(0, color='k',linestyle="--",label="")
        ax2.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

        # save pdf
        pdf.savefig(fig,bbox_inches='tight')

    pdf.close()


# Plot differentiation between specimens grouped by species and karyotype:

# In[28]:


get_ipython().run_cell_magic('capture', '--no-stdout --no-display', 'loop_Fst(\n    name="colgaminv",\n    popA_list=["gam_0","gam_2"],\n    popC_list=["gam_0","col_0","gam_2","col_2"],\n    popA_ac=oc_genalco_sps_seg_inv, \n    popC_ac=oc_genalco_sps_seg_inv, \n    pos=oc_genvars_seg["POS"][:],\n    cycle="C"\n)')

