import sys
import Bio
from Bio import Phylo
import csv

master_tree = Phylo.read("/home/xavi/Documents/VariationAg1k/rdl-Agam-evolution/alignments_Rdl_haplotype_phylo/phylogenies/iqr_corehap.treefile","newick")

master_tree.distance("AA0040-Ca_GHcol_gt0_kt0", "AJ0088-Cb_GW_gt0_kt0")

DistanceABAB=[]
DistanceBCBC=[]
DistanceACAB=[]
DistanceACBC=[]

for tree in trees: #for each genealogy
	if tree.distance('3','4') < tree.distance('2','3') and tree.distance('2','4'): #if the topology is AB
		DistanceABABi = tree.distance('3','4') #AB given AB
		DistanceACABi = tree.distance('2','4') #AC given AB
		DistanceABAB.append(DistanceABABi)
		DistanceACAB.append(DistanceACABi)
	if tree.distance('2','3') < tree.distance('3','4') and tree.distance('2','4'): #if the topology is BC
		DistanceBCBCi = tree.distance('2','3') #BC given BC
		DistanceACBCi = tree.distance('2','4') #AC given BC
		DistanceBCBC.append(DistanceBCBCi)
		DistanceACBC.append(DistanceACBCi)

D1 = (sum(DistanceABAB)/len(DistanceABAB))-(sum(DistanceBCBC)/len(DistanceBCBC)) #Estimate D1
D2 = (sum(DistanceACAB)/len(DistanceACAB))-(sum(DistanceACBC)/len(DistanceACBC)) #Estimate D2

#Write to csv
with open(sys.argv[2],'a') as statsfile:
	wr = csv.writer(statsfile)
	wr.writerow([D1,D2])
