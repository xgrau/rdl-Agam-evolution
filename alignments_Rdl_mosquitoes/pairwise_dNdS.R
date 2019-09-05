library(ape)
library(pheatmap)

# read alignment
setwd("/home/xavi/Documents/VariationAg1k/rdl-Agam-evolution/alignments_Rdl_mosquitoes/")
ali_file = "rdl_cds_complete_isoforms_codon.pal2nal.fasta"
ali = read.FASTA(ali_file, type = "DNA")

# calculate dNdS
ali_dnds = dnds(ali)
ali_dnds = as.matrix(ali_dnds)

# print results for Anogam Rdl ortholog
# print(ali_dnds["Anogam_AGAP006028-RC",2:nrow(ali_dnds)])
# print(ali_dnds["Anogam_AGAP006028-RC","Aedaeg_AAEL008354-RJ"])
# max(ali_dnds["Anogam_AGAP006028-RC",2:nrow(ali_dnds)])
# min(ali_dnds["Anogam_AGAP006028-RC",2:nrow(ali_dnds)])


# print heatmap
col.fun = colorRampPalette(interpolate="l",c("aliceblue","deepskyblue","dodgerblue4"))
pdf(file=paste(ali_file,".dnds.pdf",sep=""),height=12,width=12)
pheatmap(ali_dnds, color = col.fun(20), breaks = seq(0,0.25,length.out = 20), 
         cellwidth = 18, cellheight = 18,
         border_color = "white", cluster_cols=F, cluster_rows=F,display_numbers = T,number_color = "white",
         main="dN/dS")
dev.off()
