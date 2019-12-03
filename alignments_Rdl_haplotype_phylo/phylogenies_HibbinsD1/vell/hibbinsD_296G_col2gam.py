import sys
import Bio
from Bio import Phylo
import csv
import random
import numpy as np
from matplotlib import pyplot as plt

# input: master tree
tree = Phylo.read("../phylogenies/iqr_corehap.treefile","newick")
out_name = "out_296G_col2gam"
max_size = 20
num_bootstrap = 10000

# input: lists of specimens from populations A,B,C,D
fil_A = "specimens_lists/rdo_hapnet_rdl.alig_loc.rei.seqs_col_gt0_kt0.list"
fil_B = "specimens_lists/rdo_hapnet_rdl.alig_loc.rei.seqs_col_gt1_kt0.list"
fil_C = "specimens_lists/rdo_hapnet_rdl.alig_loc.rei.seqs_gam_gt1_kt0.list"
fil_D = "specimens_lists/rdo_hapnet_rdl.alig_loc.rei.seqs_qua_gt0_kt0.list"



# load lists
lis_A = [line.rstrip('\n') for line in open(fil_A)] # A = 4; non-admixed ingroup
lis_B = [line.rstrip('\n') for line in open(fil_B)] # B = 3; suspected ingroup donor
lis_C = [line.rstrip('\n') for line in open(fil_C)] # C = 2; suspected receptor
# lis_D = [line.rstrip('\n') for line in open(fil_D)] # D = 1; outgroup
# if B->C: D2>0
# if C->B: D2=0

lis_A = random.sample(lis_A, min(max_size,len(lis_A)))
lis_B = random.sample(lis_B, min(max_size,len(lis_B)))
lis_C = random.sample(lis_C, min(max_size,len(lis_C)))

len_tot = len(lis_A) * len(lis_B) * len(lis_C)
print(len_tot)


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
				dis_ABAB_i = tree.distance(popb,popa) # AB dist
				dis_ACAB_i = tree.distance(popc,popa) # AC dist
				dis_ABAB.append(dis_ABAB_i)
				dis_ACAB.append(dis_ACAB_i)
			if tree.distance(popc,popb) < tree.distance(popb,popa) and tree.distance(popc,popa): # if topology is BC
				dis_BCBC_i = tree.distance(popc,popb) # BC dist
				dis_ACBC_i = tree.distance(popc,popa) # AC dist
				dis_BCBC.append(dis_BCBC_i)
				dis_ACBC.append(dis_ACBC_i)
			
			n = n+1
			if n % int(len_tot*0.05) == 0 : print("%i/100 %i/%i | len AB: %i | len BC: %i" % ( 100*(n/len_tot), n, len_tot, len(dis_ACAB), len(dis_ACBC) ) )

D1 = (sum(dis_ABAB)/len(dis_ABAB))-(sum(dis_BCBC)/len(dis_BCBC)) # estimate D1
D2 = (sum(dis_ACAB)/len(dis_ACAB))-(sum(dis_ACBC)/len(dis_ACBC)) # estimate D2


# bootstrap result
D2_boot = []
for i in range(num_bootstrap):
	dis_ACAB_b = np.random.choice(dis_ACAB, len(dis_ACAB))
	dis_ACBC_b = np.random.choice(dis_ACBC, len(dis_ACBC))
	D2_boot_b  = (sum(dis_ACAB_b)/len(dis_ACAB_b))-(sum(dis_ACBC_b)/len(dis_ACBC_b)) # estimate D2
	D2_boot.append(D2_boot_b)

D2_boot     = np.array(D2_boot)
D2_over_thr = np.sum(D2_boot > 0) / num_bootstrap


# ploto output
plt.hist(D2_boot, bins=50, color="blue")
plt.xlabel("D2")
plt.ylabel("Freq")
plt.title("%s\nD2 = %.3E | frac>0 = %.3E (n boot = %i)" % (out_name, D2, D2_over_thr, num_bootstrap))
plt.savefig("%s.pdf" % out_name)
plt.close()

print("D2 =",D2)
print("D1 =",D1)
