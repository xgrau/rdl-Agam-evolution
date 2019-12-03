import sys
import Bio
from Bio import Phylo
import csv
import random

# input: master tree
tree = Phylo.read("/home/xavi/Documents/VariationAg1k/rdl-Agam-evolution/alignments_Rdl_haplotype_phylo/phylogenies/iqr_corehap.treefile","newick")

# input: lists of specimens from populations A,B,C,D
fil_A = "/home/xavi/Documents/VariationAg1k/rdl-Agam-evolution/alignments_Rdl_haplotype_phylo/phylogenies_HibbinsD1/alignments/rdo_hapnet_rdl.alig_loc.rei.seqs_col_gt0_kt0.list"
fil_B = "/home/xavi/Documents/VariationAg1k/rdl-Agam-evolution/alignments_Rdl_haplotype_phylo/phylogenies_HibbinsD1/alignments/rdo_hapnet_rdl.alig_loc.rei.seqs_col_gt1_kt0.list"
fil_C = "/home/xavi/Documents/VariationAg1k/rdl-Agam-evolution/alignments_Rdl_haplotype_phylo/phylogenies_HibbinsD1/alignments/rdo_hapnet_rdl.alig_loc.rei.seqs_gam_gt1_kt0.list"
fil_D = "/home/xavi/Documents/VariationAg1k/rdl-Agam-evolution/alignments_Rdl_haplotype_phylo/phylogenies_HibbinsD1/alignments/rdo_hapnet_rdl.alig_loc.rei.seqs_qua_gt0_kt0.list"

# load lists
lis_A = [line.rstrip('\n') for line in open(fil_A)] # A = 4
lis_B = [line.rstrip('\n') for line in open(fil_B)] # B = 3
lis_C = [line.rstrip('\n') for line in open(fil_C)] # C = 2
lis_D = [line.rstrip('\n') for line in open(fil_D)] # D = 1

lis_A = random.sample(lis_A, 40)
lis_B = random.sample(lis_B, 40)
lis_C = random.sample(lis_C, 40)

len_tot = len(lis_A) * len(lis_B) * len(lis_C)

# empty lists
dis_ABAB = []
dis_BCBC = []
dis_ACAB = []
dis_ACBC = []

# loop and add to output lists
n = 0
for popa in lis_A:
	for popb in lis_B:
		for popc in lis_C:
			if tree.distance(popb,popa) < tree.distance(popc,popb) and tree.distance(popc,popa): # if topology is AB
				dis_ABAB_i = tree.distance(popb,popa) # AB distance
				dis_ACAB_i = tree.distance(popc,popa) # AC distance
				dis_ABAB.append(dis_ABAB_i)
				dis_ACAB.append(dis_ACAB_i)
			if tree.distance(popc,popb) < tree.distance(popb,popa) and tree.distance(popc,popa): # if topology is BC
				dis_BCBC_i = tree.distance(popc,popb) # BC distance
				dis_ACBC_i = tree.distance(popc,popa) # AC distance
				dis_BCBC.append(dis_BCBC_i)
				dis_ACBC.append(dis_ACBC_i)
			
			if 100*(n/len_tot) % 5 == 0 : print("%.2f/1, %i/%i" % ( (n/len_tot), n, len_tot ) )
			n = n+1


D1 = (sum(dis_ABAB)/len(dis_ABAB))-(sum(dis_BCBC)/len(dis_BCBC)) # estimate D1
D2 = (sum(dis_ACAB)/len(dis_ACAB))-(sum(dis_ACBC)/len(dis_ACBC)) # estimate D2

#Write to csv
with open(sys.argv[2],'a') as statsfile:
	wr = csv.writer(statsfile)
	wr.writerow([D1,D2])
