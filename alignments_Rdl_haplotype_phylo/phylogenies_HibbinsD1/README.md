# Hibbins D2: direction of introgression

## Methods

From this paper

Formula:

```python
D2 = dist_AC_in_AB_topologies - dist_AC_in_BC_topologies
```

* If **D2=0**, it means that A-to-C distance in BC and AB topologies is more or less the same, which means that the divergence of A-C is more or less the same in any scenario. Therefore, **null**: no introgression from B to C.
* If **D2>0**, it means that A-to-C distance is higher in BC topologies than in AB topologies, which means that the divergence of A-C is younger in the BC scenario than in the AB scenario. Therefore, **introgression from B to C** specifically.

## Tests

We perform four tests:

### *296G* from *gambiae* to *coluzzii*

**296G from *gambiae* to *coluzzii*** on non-inverted backgrounds; where A=gam wt, B=gam *296G* (putative donor), C=col *296G* (putative acceptor), D=quad wt (outgroup):

```bash
python hibbinsD_from_newick_XG2.py ../phylogenies/iqr_loc.treefile 296G_gam2col specimens_lists/gam_gt0_kt0.list specimens_lists/gam_gt1_kt0.list specimens_lists/col_gt1_kt0.list
```

This test is positive (D2>0, p<1e-4 from 10,000 bootstrap replicates), proving that 296G went from gam to col.

Combinations of geographical populations that follow this *296S* presence pattern also show a positive signal, e.g.:

```bash
$ python hibbinsD_from_newick_XG2_max20.py ../phylogenies/iqr_loc.treefile 296G_pops_gam2col specimens_lists/pop.GM.list specimens_lists/pop.BFgam.list specimens_lists/pop.CIcol.list 
Comparison:   296G_pops_gam2col
population A: specimens_lists/pop.GM.list | size: 20
population B: specimens_lists/pop.BFgam.list | size: 20
population C: specimens_lists/pop.CIcol.list | size: 20
Total number of subtrees: 8000
[...]
D2 = 0.0036693675917181534
D1 = 0.010400313646903192
```

Whereas a combination where *col* population does not have *296S* will fail:

```bash
python hibbinsD_from_newick_XG2_max20.py ../phylogenies/iqr_loc.treefile 296G_pops_gam2col_neg specimens_lists/pop.GM.list specimens_lists/pop.BFgam.list specimens_lists/pop.GNcol.list
Comparison:   296G_pops_gam2col_neg
population A: specimens_lists/pop.GM.list | size: 20
population B: specimens_lists/pop.BFgam.list | size: 20
population C: specimens_lists/pop.GNcol.list | size: 8
Total number of subtrees: 3200
[...]
100/100 3200/3200 | len AB: 3040 | len BC: 160
D2 = 0.0
D1 = -0.0058808503107894705
```

### *296G* from *coluzzii* to *gambiae*

**296G from *coluzzii* to *gambiae*** on non-inverted backgrounds; where A=col wt, B=col *296G* (putative donor), C=gam *296G* (putative acceptor), D=quad wt (outgroup):

```bash
python hibbinsD_from_newick_XG2.py ../phylogenies/iqr_loc.treefile 296G_gam2col specimens_lists/gam_gt0_kt0.list specimens_lists/gam_gt1_kt0.list specimens_lists/col_gt1_kt0.list specimens_lists/qua_gt0_kt0.list
```

This test is non-significant (D2~0, p=1 from 10,000 bootstrap replicates).

### *296S* from *arabiensis* to *coluzzii*

**296S from *arabiensis* to *coluzzii*** on inverted backgrounds; where A=ara wt, B=ara 296S (putative donor), C=col 296S (putative acceptor), D=meru wt (outgroup):

```bash
python hibbinsD_from_newick_XG2.py ../phylogenies/iqr_loc.treefile 296G_gam2col specimens_lists/gam_gt0_kt0.list specimens_lists/gam_gt1_kt0.list specimens_lists/col_gt1_kt0.list specimens_lists/qua_gt0_kt0.list
```

This test is non-significant (D2~0, p=1 from 10,000 bootstrap replicates).

This is troubling, because we did know that **296S introgressed from *ara* to *col***, and in fact most of the topologies support the relationship between B & C:

```bash
AB: 397 (ara wt + ara 296S)
BC: 1983 (ara 296S + ara 296S)
total: 2380
```

But maybe this makes sense: since none of the *ara* haplotypes belong to the donor populations, the distances between *ara* (A/B) and *col* (C) is always going to be very high.

### *296S* from *coluzzii* to *gambiae*

**296S from *coluzzii* to *gambiae*** on inverted backgrounds; where A=ara wt, B=ara 296S (putative donor), C=col 296S (putative acceptor), D=meru wt (outgroup):

```bash
python hibbinsD_from_newick_XG2.py ../phylogenies/iqr_loc.treefile 296G_gam2col specimens_lists/gam_gt0_kt0.list specimens_lists/gam_gt1_kt0.list specimens_lists/col_gt1_kt0.list specimens_lists/qua_gt0_kt0.list
```

This test is non-significant (D2~0, p=1 from 10,000 bootstrap replicates).
