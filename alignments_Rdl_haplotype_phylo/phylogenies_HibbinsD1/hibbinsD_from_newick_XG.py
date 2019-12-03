import sys
import Bio
from Bio import Phylo
import csv
import random
import numpy as np
from matplotlib import pyplot as plt

# INPUT:
# arg1: newick tree
# arg2: name of comparison (e.g. 296S_ara2col)
# arg3: list of specimens in A
# arg4: list of specimens in B
# arg5: list of specimens in C

# OUTPUT:
# histogram from 10000 bootstraps
# if B->C: D2>0
# if C->B: D2=0

# input: master tree
tree = Phylo.read(sys.argv[1],"newick")
out_name = sys.argv[2]
max_size = 100
num_bootstrap = 10000

# input: lists of specimens from populations A,B,C,D
fil_A = sys.argv[3]
fil_B = sys.argv[4]
fil_C = sys.argv[5]
# fil_D = "specimens_lists/rdo_hapnet_rdl.alig_loc.rei.seqs_qua_gt0_kt0.list"


# load lists
lis_A = [line.rstrip('\n') for line in open(fil_A)] # A = 4; non-admixed ingroup
lis_B = [line.rstrip('\n') for line in open(fil_B)] # B = 3; suspected ingroup donor
lis_C = [line.rstrip('\n') for line in open(fil_C)] # C = 2; suspected receptor

# subset lists to a max size
lis_A = random.sample(lis_A, min(max_size,len(lis_A)))
lis_B = random.sample(lis_B, min(max_size,len(lis_B)))
lis_C = random.sample(lis_C, min(max_size,len(lis_C)))

len_tot = len(lis_A) * len(lis_B) * len(lis_C)
print("Comparison:  ", out_name)
print("population A:", fil_A, "| size:", len(lis_A))
print("population B:", fil_B, "| size:", len(lis_B))
print("population C:", fil_C, "| size:", len(lis_C))
print("Total number of subtrees:", len_tot)

# init empty lists
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
			if n % int(len_tot*0.05) == 0 : 
				print("%i/100 %i/%i | len AB: %i | len BC: %i" % ( 100*(n/len_tot), n, len_tot, len(dis_ACAB), len(dis_ACBC) ) )


if len(dis_ACAB) > 0 and len(dis_ACBC) > 0:

	# mean statistics over entire dataset
	D1 = (sum(dis_ABAB)/len(dis_ABAB))-(sum(dis_BCBC)/len(dis_BCBC)) # estimate D1
	D2 = (sum(dis_ACAB)/len(dis_ACAB))-(sum(dis_ACBC)/len(dis_ACBC)) # estimate D2

	# bootstrap D2 result
	D2_boot = []
	for i in range(num_bootstrap):
		dis_ACAB_b = np.random.choice(dis_ACAB, len(dis_ACAB))
		dis_ACBC_b = np.random.choice(dis_ACBC, len(dis_ACBC))
		D2_boot_b  = (sum(dis_ACAB_b)/len(dis_ACAB_b))-(sum(dis_ACBC_b)/len(dis_ACBC_b)) # estimate D2
		D2_boot.append(D2_boot_b)

	# fraction over threshold
	D2_boot     = np.array(D2_boot)
	D2_over_thr = np.sum(D2_boot > 0) / num_bootstrap

	# ploto output
	plt.hist(D2_boot, bins=50, color="blue")
	plt.xlabel("D2")
	plt.ylabel("Freq")
	plt.title("%s\nD2 = %.5E | frac>0 = %.5f (n boot = %i)\np = %.5E" % (out_name, D2, D2_over_thr, num_bootstrap, 1-D2_over_thr))
	plt.savefig("out_%s.pdf" % out_name)
	plt.close()

	print("D2 =",D2)
	print("D1 =",D1)

else:
	print("D2 or D1 can't be calculated, at least one denominator is zero (e.g. there are no BC topologies)")

