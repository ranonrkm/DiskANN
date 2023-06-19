# root="/mnt/collections/data/XC/QQ-220M/results/Bert-XC/QQ-220M/BertDXML2-distributed16/version_1001/epoch_0"
# root="/mnt/collections/data/XC/EPM-20M/epoch_9"
root="/mnt/collections/data/XC/EPM-20M/Results/Bert-XC/EPM-20M/BertDXML2-distributed8/version_90003/epoch_24"
# root="/mnt/collections/data/XC/EPM-5M_ss20M/Results/Bert-XC/EPM-5M_ss20M/BertDXML2-distributed4/version_1004"
data_path="${root}/base.bin"
query_path="${root}/test_embeddings_sample_1_data.bin"
qids_path="${root}/tst_ivf_sample_1.bin"
R=50
Lbuild=128
alpha=1.2
index_path="${root}/diskann_R${R}_L${Lbuild}_A${alpha}_cosine" #_cosine_opq_16

# cmd="./tests/build_memory_index_with_ood_query  --data_type float --dist_fn cosine --data_path $data_path --query_path $query_path --qids_path $qids_path \
#                                     --index_path_prefix $index_path \
#                                     -R $R -L $Lbuild --alpha $alpha -T 64 --lambda 0.9 "

cmd="./tests/build_memory_index  --data_type float --dist_fn cosine --data_path $data_path \
                                    --index_path_prefix $index_path \
                                    -R $R -L $Lbuild --alpha $alpha -T 64 " # --build_PQ_bytes 16 --use_opq

echo $cmd
eval $cmd

K=200
query_path="${root}/test_embeddings_sample_data.bin"
gt_path="${root}/test_gt_sample_cosine.bin"
cmd="./tests/utils/compute_groundtruth  --data_type float --dist_fn cosine --base_file $data_path --query_file $query_path --gt_file $gt_path --K $K"

# echo $cmd
# eval $cmd

K=10
res_path="${root}/tst_vamana"
cmd="./tests/search_memory_index  --data_type float --dist_fn cosine --index_path_prefix $index_path --query_file $query_path  --gt_file $gt_path \
                                -K $K -L 100 200 300 --result_path $res_path -T 64"

echo $cmd
eval $cmd

query_path="${root}/test_embeddings.bin"
shortlist_path="${root}/anns_indices_500_idx_uint32.bin"
res_path="${root}/tst_vamana"
# ./tests/utils/prune_shortlist --data_type float --dist_fn cosine --base_file $data_path --query_file $query_path \
#                     --shortlist_file $shortlist_path --K $K --result_path_prefix $res_path \
#                     --alpha $alpha --num_threads 128

# ./tests/build_and_search_memory_index --data_type float --dist_fn cosine --data_path $data_path \
#             -R $R --Lbuild $Lbuild --alpha $alpha -T 64 \
#             --query_path $query_path --result_path $res_path --gt_path_prefix /mnt/collections/data/XC/LF-AmazonTitles-131K/data/trn_X_Y \
#             -K 100 --Lsearch 100 --num_bins 4 --adj_samp 0