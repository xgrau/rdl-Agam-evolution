import sys
import Bio
from Bio import Phylo
import csv
import random
import numpy as np
from matplotlib import pyplot as plt

# input: master tree
tree = Phylo.read("../phylogenies/iqr_corehap.treefile","newick")
out_name = "out_296S_ara2col"

# input: lists of specimens from populations A,B,C,D
fil_A = "specimens_lists/rdo_hapnet_rdl.alig_loc.rei.seqs_ara_gt0_kt2.list"
fil_B = "specimens_lists/rdo_hapnet_rdl.alig_loc.rei.seqs_ara_gt2_kt2.list"
fil_C = "specimens_lists/rdo_hapnet_rdl.alig_loc.rei.seqs_col_gt2_kt2.list"
# fil_D = "specimens_lists/rdo_hapnet_rdl.alig_loc.rei.seqs_mer_gt0_kt2.list"

# load lists
lis_A = [line.rstrip('\n') for line in open(fil_A)] # A = 4; non-admixed ingroup
lis_B = [line.rstrip('\n') for line in open(fil_B)] # B = 3; suspected ingroup donor
lis_C = [line.rstrip('\n') for line in open(fil_C)] # C = 2; suspected receptor
# lis_D = [line.rstrip('\n') for line in open(fil_D)] # D = 1; outgroup
# if B->C: D2>0
# if C->B: D2=0

lis_A = random.sample(lis_A, min(100,len(lis_A)))
lis_B = random.sample(lis_B, min(100,len(lis_B)))
lis_C = random.sample(lis_C, min(100,len(lis_C)))

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

D2_boot = []
for i in range(1000):
	dis_ACAB_b = np.random.choice(dis_ACAB, len(dis_ACAB))
	dis_ACBC_b = np.random.choice(dis_ACBC, len(dis_ACBC))
	D2_boot_b  = (sum(dis_ACAB_b)/len(dis_ACAB_b))-(sum(dis_ACBC_b)/len(dis_ACBC_b)) # estimate D2
	D2_boot.append(D2_boot_b)

plt.hist(D2_boot, bins=50)
plt.savefig("%s.pdf" % out_name)
print("D2 =",D2)
print("D1 =",D1)
