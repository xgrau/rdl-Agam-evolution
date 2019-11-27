### Define input ####

# input files
library(ape)
library(phytools)
library(stringr)

prefix    = "out"

phy_list  = c("phylogenies/iqr_corehap.treefile","phylogenies/iqr_rdl5p.treefile","phylogenies/iqr_rdl3p.treefile","phylogenies/iqr_rdlup.treefile","phylogenies/iqr_rdldo.treefile")
phy_name  = c("Core Rdl haplotype","5' Rdl","3' Rdl","Upstream","Downstream")


#  plot phylogenies
pdf(file=paste(prefix,"dists_phylo.pdf",sep="."),height=6,width=6)
cou      = 0
for (phi in phy_list) {
  
  # tree to distance matrix
  cou    = cou+1
  phy    = read.tree(phi)
  dis    = cophenetic.phylo(phy)
  
  phy_p = midpoint.root(phy)
  phy_p$tip.label_sep  = gsub("_"," ",phy_p$tip.label)
  phy_p$tip.species    = gsub("^[A-Z][A-Z]","",word(phy_p$tip.label_sep,2))
  phy_p$tip.species[phy_p$tip.species == ""] = "gamcol"
  phy_p$tip.population = word(phy_p$tip.label_sep,2)
  phy_p$tip.genotype   = word(phy_p$tip.label_sep,3)
  phy_p$tip.karyotype  = as.factor(word(phy_p$tip.label_sep,4))
  phy_p$tip.ktgt       = as.factor(paste(phy_p$tip.karyotype,paste(phy_p$tip.genotype,phy_p$tip.species,sep="_"),sep="_"))
  phy_p$tip.colors     = c("orangered3","orangered3","orangered3","deeppink3","deeppink4",         # kt0_gt0_col, kt0_gt0_gam, kt0_gt0_gamcol, kt0_gt0_mel, kt0_gt0_qua
                           "orange","orange","deeppink",                                           # kt0_gt1_col, kt0_gt1_gam, kt0_gt2_qua
                           "springgreen4","dodgerblue3","dodgerblue3","dodgerblue3","dodgerblue4", # kt2_gt0_ara, kt2_gt0_col, kt2_gt0_gam, kt2_gt0_gamcol, kt2_gt0_mer
                           "purple","purple",                                                      # kt2_gt1_col, kt2_gt1_gam, 
                           "springgreen3","turquoise2"                                             # kt2_gt2_ara, kt2_gt2_col
                           )[phy_p$tip.ktgt]
  phy_p$tip.label      = rep("·", length(phy$tip.label))
  phy_p$edge.length[phy_p$edge.length >  0.005] = 0.005
  phy_p$edge.length[phy_p$edge.length == 0]    = 5e-5
  
  plot.phylo(phy_p, type="unr",
             use.edge.length=T, show.tip.label=T, show.node.label=F,
             tip.color = phy_p$tip.colors, lab4ut = "axial",
             edge.color = "slategray3",
             font = 0.5, edge.width = 0.5, node.depth=1,
             main=phy_name[cou])
  
  
  
}

dev.off()


pdf(file=paste(prefix,"dists_phylo_llarg.pdf",sep="."),height=150,width=30)
cou      = 0
for (phi in phy_list) {
  
  # tree to distance matrix
  cou    = cou+1
  phy    = read.tree(phi)
  dis    = cophenetic.phylo(phy)
  
  phy_p = midpoint.root(phy)
  phy_p$tip.label_sep  = gsub("_"," ",phy_p$tip.label)
  phy_p$tip.species    = gsub("^[A-Z][A-Z]","",word(phy_p$tip.label_sep,2))
  phy_p$tip.species[phy_p$tip.species == ""] = "gamcol"
  phy_p$tip.population = word(phy_p$tip.label_sep,2)
  phy_p$tip.genotype   = word(phy_p$tip.label_sep,3)
  phy_p$tip.karyotype  = as.factor(word(phy_p$tip.label_sep,4))
  phy_p$tip.ktgt       = as.factor(paste(phy_p$tip.karyotype,paste(phy_p$tip.genotype,phy_p$tip.species,sep="_"),sep="_"))
  phy_p$tip.colors     = c("orangered3","orangered3","orangered3","deeppink3","deeppink4",         # kt0_gt0_col, kt0_gt0_gam, kt0_gt0_gamcol, kt0_gt0_mel, kt0_gt0_qua
                           "orange","orange","deeppink",                                           # kt0_gt1_col, kt0_gt1_gam, kt0_gt2_qua
                           "springgreen4","dodgerblue3","dodgerblue3","dodgerblue3","dodgerblue4", # kt2_gt0_ara, kt2_gt0_col, kt2_gt0_gam, kt2_gt0_gamcol, kt2_gt0_mer
                           "purple","purple",                                                      # kt2_gt1_col, kt2_gt1_gam, 
                           "springgreen3","turquoise2"                                             # kt2_gt2_ara, kt2_gt2_col
  )[phy_p$tip.ktgt]
  
  plot.phylo(phy_p, type="phy",
             use.edge.length=T, show.tip.label=T, show.node.label=T,
             tip.color = phy_p$tip.colors, lab4ut = "axial",
             edge.color = "slategray3",
             font = 0.5, node.depth=1, underscore=F,
             main=phy_name[cou])
  
  
  
}

dev.off()


stop("ara")

