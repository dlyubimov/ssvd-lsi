### Stochastic SVD and LSI related patches for Mahout ###

## Purpose

This is repository for keeping working notes and usage etc. documentation 
for series of patches and various branches in *svd-lsi* repostiory. 

*svd-lsi* repository represents a series of patches for Apache Mahout project 
mainly growing out of [MAHOUT-376]( https://issues.apache.org/jira/browse/MAHOUT-376) work. I decided to sync my home git repo with 
github mainly 
out of my deep distrust towards my computer's hardware being 
a sole copy of this repository branches. 

Another reason is to leverage flexibility 
of Git branches etc. to work with various experimental optimizations without 
loosing the stable patches and variations, instead of operating with series of patches at Mahout Jira level.

## What is Stochastic SVD

Stochastic SVD is a stochastic technique for computing large dimensional approximate low rank SVDs 
(Singular Value Decomposition)  
with rank reaching potentially into hundreds of singular values with just very few passes
over data.

This particular project is focused on developing techniques allowing for parallelization of 
SSVD computation on top of MapReduce framework and Mahout's vectorization framework. 


Most of SSVD applications are expected to be revolving around PCA (Principal Component Analysis)
as well as LSI. Mahout also uses SVD computations for recommender work, although 
quality of such recomendations is often (IMO) questionable as in practice such 
recommendation input is often too sparse for SVD to make good predictions.

My interest in SVD is primarily LSI (nlp) work, context disambiguation and cross-language 
information retrieval applications. I may be looking into PCA and dimensionality reduction 
in certain cases.

I saw some ocasional use of SSVD acronym in literature to also denote "Standard SVD" as opposed to other methods. 
Stochastic SVD is not standard SVD, those are quite different meanings.

## Build & Install
This code uses new Hadoop API which is only partially supported in hadoop 0.20.x. 

Therefore, the patch updates dependency on CDH3 (I tested it with both beta2 and 3. ) I did not 
test it with hadoop 0.21, but i expect it to be compatible at the API level with that release 
as well. So appropriate Mahout jar should be installed in local maven repository for the build 
to go thru.

## Current branches 

### ssvd-givens in ssvd-lsi repo 
is current stable version with distributed QR step based on Givens rotations. 
This version by my estimates should scale at least to spec of 1 million by 1 billion dense data 
matrix in input for memory, assuming -Xmx1G for mapper child processes. 

It has a bunch of limitations for network I/O as discussed int the 
[working notes](https://github.com/dlyubimov/ssvd-lsi/raw/doc/SSVD%20working%20notes.pdf) but 
is optimized for most LSI work where average number of lemmas per document does not exceed 30k and 
contains no uncommitted Mahout hacks.

That said, even with perceived I/O deficiencies for wider matrices, this algorithm is expected 
to be CPU bound to such degree that seemingly far overshadows any I/O concerns.

tag *givens-verified* tags last point of successful distributed/scale test with verification of results. Use that.

### MAHOUT-593
Mahout patch, a variant of givens-ssvd branch, with dependencies on CDH removed and which (hopefully) 
works with regular 0.20.2 hadoop framework.

### branch ssvd-vw-hack in ssvd-lsi
This branch contains enhancements to VectorWritable in order to enable vector preprocessing capability 
(based on Mahout trunk). It introduces VectorPreprocessor interface. 'git diff trunk > ssvd-vw-hack.patch' 
would produce the patch.

### branch ssvd-preprocessing (alpha-ish)
This branch is based on merge of ssvd-givens and ssvd-vw-hack and contains further alterations 
to SSVD in order to captialize on vector preprocessing capability and thus making
vector prebuffering unnecessary. This opens up n-bound (width) of the source matrix in terms of RAM
and also allows for more efficient memory use in other parts of algorithms (i.e. higher blocks) thus 
reducing running time. Most importantly, it copes with the issues of occasional localized 
spikes in data density. It also reduces GC thrashing as default VectorWritable behavior is to allocate 
a new Vector-derived storage on each record read.

Status: all code intended for this patch is here and unit tests (including larger tests) are succeeding. 
Unit tests are using hadoop in local mode only though; i haven't yet tested it with distributed Hadoop 
setup (which is why i call it alhpa-ish quality at this point).

tag *preprocessing-verified* tags last point of successful distributed/scale test with verification of results. Use that.


### branch ssvd-wide (WIP)
This branch introduces more efficient support for wider matrices (>30k nonzero data elements per row) in terms of 
MapReduce network I/O. 

This branch is based on ssvd-preprocessing branch. 

When A blocks become too big to fit into an hdfs split, one has to choices: 
either to increase the FileInput's minSplitSize parameter, or aggregate Y rows and send them to reducer. 
This to a significant degree solves "supersplits" p
roblem defined in p. 6.1 of the working notes.

This should make I/O significantly more forgiving for wide matrices up to about 8 million non-zero elements in 
a row. 

After ~8 million in dense width, A matrix I/O would become an issue again but at this point i speculate 
that CPU will be far narrower bottleneck by then (as well as before this number).

### branch ssvd-tall (not started)
If billion rows doesn't sound like a sufficient scale, or if one wanted to work that scale under low RAM specs 
in child processes, or one would significantly increase number of singular values assessed (which increases 
quality of stochastic projection), then this branch is expected to address that. Each additional map-only 
pass over Q, R blocks will increase scale for number of rows approximately 1000 times.

It is possible to expect to have _ssvd-tall_ + _ssvd_wide_ combined branch which would really open up 
bounds for both width and height of A without incurring significant RAM or IO penalty. 
(decrease in RAM would translate into some CPU penalty though, so there's no quite free lunch here).

### branch ssvd-spliced-input (possible future work)
This branch will solve issue of input io beyond 8 million dense elements in a row per above. At this point 
it is purely speculative work since there is little insentive to go there: 

1) it'll be technology in search 
of a problem. i know almost of no problem that would exceed 8 million average dense elements per row. 

2) it requires spliced vectors or blockwise input format which is not supported by any prep utilities in Mahout as of 
now, so this works would depend on ways to produce such inputs. 

3) I have a strong suspicion that CPU will 
be far narrower bottleneck than network IO in such situation. I might be wrong, certain commercial applications 
actually might benefit from this approach, but i think it is very unlikely most of OSS crowd would have to deal
with that scale.



## How to use 

### branch *ssvd-givens* and *ssvd-preprocessing*

This branch contains more or less stable code optimized for tall but relatively thin (n=30,000...50,000 
non-zero elements). With 1G memory in child processes and right combination of block height, min split size 
and k,p parameters it is expected to be able to process 1 billion rows or more with 
1 million dense (i.e. non-zero) data elements. However, network IO deficiencies will start to occur much sooner 
as discussed in the p. 6.1 of the working notes (this is one of TODOs).

for usage see [cli and scaling notes](https://github.com/dlyubimov/ssvd-lsi/raw/doc/SSVD-scaling%20notes.pdf)
    


## Limitations 

### branch ssvd-givens
See Working Notes document, paragraph 6. I will be adding some experimental branches aimed at
doing easiest of optimizations providing satisfactory solution within existing vector framework.

### branch ssvd-preprocessing 
_SSVD-preprocessing_ branch addresses limitation as described in 6.2 but still has a "supersplits" IO issue 
as described in 6.1 of the working notes.

## License 

Mahout goes under Apache 2.0 license and so does all additional code in the ssvd-lsi repository. 
Some of the patches are part of Mahout contributions 
but all versions and branches bear ASF license header regardless whether they are currently 
contributed to Mahout or not. 


 