HMM formatted data:
====================
For when you need to calculate counts, then train

{split}_pair_alignments.npy: pair alignments, categorically encoded
  - matrix size: (num_pairs, 2, max_seq_len)
    > dim1=0: ancestor
    > dim2=1: descendant
  - categorically encoded by:
    > <pad>=0
    > amino acids: [A=3, C=4, D=5, E=6, F=7, G=8, H=9, I=10, 
                    K=11, L=12, M=13, N=14, P=15, Q=16, R=17, S=18, T=19, V=20, W=21, Y=22]
    > gap_char=63

    (mapping looks weird because I'm reusing tokens from neural model encoding, which had way more tokens)

    (NOTE THAT AMINO ACID ENCODING STARTS WITH 3!!! <bos> and <eos> would be the other special tokens, but I don't use them for GGI models)


{split}_AAcounts.npy: stationary distribution of amino acids
  - rows correspond to: [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]
  - vector size: (20, )
  - dtype: (I think default int type? not sure)


{split}_metadata.tsv: info about each pair in the split; index corresponds to dim1 in the pair alignments file
  - columns:
    > original_idx: index position in original numpy matrices (before creating data splits)
    > pairID: name for pair
    > ancestor: name of ancestor
    > descendant: name of descendant
    > TREEDIST_anc-to-root: tree distance from ancestor to root (from pfam trees)
    > TREEDIST_desc-to-root: tree distance from descendant to root (from pfam trees)
    > TREEDIST_stem-to-Root: tree distance from stem of cherry to root (from pfam trees)
    > TREEDIST_anc-to-desc: tree distance from ancestor to descendant (from pfam trees)
    > perc_seq_id: percent seq ID between ancestor and descendant (calculated according to Bileschi and Colwell et al 2022)
    > pfam: which pfam this pair came from
    > pfam_Nseqs: size of the pfam
    > type: type of the pfam
    > clan: clan label for the pfam (if given)
    > alignment_length: length of alignment (number of amino acids + gap characters)
    > KProt_Cluster_IDs: during kprototypes, which cluster was this pair a part of?
    > split_assignment: which data split was this assigned to?

 
Other files:
------------
LG08_exchangeability_r.npy: the LG08 exchangeability matrix (elements reordered to match my row/column ordering)
split_sizes.txt: how many samples per split; this is read by the lazy dataloader
