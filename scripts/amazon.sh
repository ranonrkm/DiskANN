root="${HOME}/data/XC/LF-AmazonTitles-131K"
# data_path="${root}/data/DiskANN_data/lbl.bin"
# query_path="${root}/data/DiskANN_data/trn.bin"
# nnid_path="${root}/results/trn_vamana_10_idx_uint32.bin"
data_path="${root}/embeddings/xfc/classifier_embeddings_norm.bin"
R=70
Lbuild=500
alpha=1.2
# index_path="${root}/embeddings/xfc_shortlist/Vamana-${Lbuild}-${R}-${alpha}"
index_path="${root}/embeddings/xfc/diskann_R${R}_L${Lbuild}_A${alpha}"

# cmd="./tests/build_memory_index  --data_type float --dist_fn cosine --data_path $data_path --query_path $query_path --nnid_path $nnid_path \
#                                     --index_path_prefix $index_path \
#                                     -R $R -L $Lbuild --alpha $alpha"

cmd="./tests/build_memory_index  --data_type float --dist_fn cosine --data_path $data_path \
                                    --index_path_prefix $index_path \
                                    -R $R -L $Lbuild --alpha $alpha"

# echo $cmd
# eval $cmd

K=10
# query_path="${root}/data/DiskANN_data/tst.bin"
query_path="${root}/embeddings/xfc_shortlist/test_embeddings_norm.bin"
# gt_path="${root}/data/DiskANN_data/tst_gt.bin"
gt_path="${root}/embeddings/xfc_shortlist/test_gt_cosine.bin"
# gt_path="${root}/data/DiskANN_data/trn_gt.bin"
cmd="./tests/utils/compute_groundtruth  --data_type float --dist_fn cosine --base_file $data_path --query_file $query_path --gt_file $gt_path --K $K"

# echo $cmd
# eval $cmd

res_path="${root}/results/tst_vamana"
cmd="./tests/search_memory_index  --data_type float --dist_fn cosine --index_path_prefix $index_path --query_file $query_path  --gt_file $gt_path \
                                -K $K -L 500 --result_path $res_path"

echo $cmd
eval $cmd