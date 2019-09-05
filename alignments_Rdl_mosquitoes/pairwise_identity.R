library(seqinr)
library(pheatmap)

# read alignment
setwd("/home/xavi/Documents/VariationAg1k/rdl-Agam-evolution/alignments_Rdl_mosquitoes/")
ali_file = "rdl_pep_complete_isoforms.ginsi.fasta"
ali      = read.alignment(ali_file, format = "fasta")
# ali = readAAMultipleAlignment(ali_file,format = "fasta")
# ali = read.FASTA(ali_file, type = "AA")

# identity matrix
ali_idm = dist.alignment(ali, matrix="id")
ali_idm = as.matrix(1-(ali_idm ^ 2))         # conversion: distance to identity. From dist.alignment vignette:
                                             # The resulting matrix contains the squared root of the pairwise distances. 
                                             # For example, if identity between 2 sequences is 80 the squared root of (1.0 - 0.8) i.e. 0.4472136

# print results for Anogam Rdl ortholog
print(ali_idm[c("Anogam_AGAP006028-RA","Anogam_AGAP006028-RB","Anogam_AGAP006028-RC"),"Aedaeg_AAEL008354-RJ"])

# print heatmap
col.fun = colorRampPalette(interpolate="l",c("aliceblue","deepskyblue","dodgerblue4"))
pdf(file=paste(ali_file,".identity.pdf",sep=""),height=12,width=12)
pheatmap(ali_idm, color = col.fun(20), breaks = seq(0.9,1,length.out = 20), 
         cellwidth = 18, cellheight = 18,
         border_color = "white", cluster_cols=F, cluster_rows=F,display_numbers = T,number_color = "white",
         main="Sequence identity")
dev.off()
