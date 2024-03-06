HMM formatted data:
===================
For when you've already precalculated counts for your data

{split}_subCounts.npy: categorical encoding of emissions seen at match states (i.e. matches and substitutions), summed over length of alignments
  - row and column indices correspond to counts for: [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]
    > example: mat[0, 1] is counts of A -> C transitions
  - matrix size: (num_seqs, 20, 20)
  - dtype: int16


{split}_insCounts.npy: count of emissions seen at insert states (i.e. insertions), summed over length of alignments
  - columns correspond to counts for: [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]
  - matrix size: (num_seqs, 20)
  - dtype: int16


{split}_delCounts.npy: count of deleted ancestor characters, summed over length of alignments
  - columns correspond to counts for: [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]
  - matrix size: (num_seqs, 20)
  - dtype: int16

{split}_transCounts.npy: count of transitions summed over length of alignments
  - rows and column indices correspond to counts for: [M, I, D]
    > example: mat[0, 1] is counts of M -> I
  - matrix size: (num_seqs, 3, 3)
  - dtype: int16


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
