root="${HOME}/data/TURINGT2I10M"
data_path="${root}/base.bin"
query_path="${root}/query_text_1M_data.bin"
nnid_path="${root}/results/train_1M_vamana_500_idx_uint32.bin"
R=64
Lbuild=128
alpha=1.2
index_path="${root}/vamana_R${R}_L${Lbuild}_A${alpha}_f10_qb"

cmd="./tests/build_memory_index  --data_type float --dist_fn l2 --data_path $data_path \
                                    --query_path $query_path --nnid_path $nnid_path --index_path_prefix $index_path \
                                    -R $R -L $Lbuild --alpha $alpha"
# cmd="./tests/build_memory_index  --data_type float --dist_fn l2 --data_path $data_path \
#                                     --index_path_prefix $index_path \
#                                     -R $R -L $Lbuild --alpha $alpha"
echo $cmd
eval $cmd

K=10
query_path="${root}/query_text.bin"
gt_path="${root}/l2_groundtruth.bin"
cmd="./tests/utils/compute_groundtruth  --data_type float --dist_fn l2 --base_file $data_path --query_file $query_path --gt_file $gt_path --K $K"

# echo $cmd
# eval $cmd

K=10
mkdir -p "${root}/results"
res_path="${root}/results/train_1M_vamana"
cmd="./tests/search_memory_index  --data_type float --dist_fn l2 --index_path_prefix $index_path --query_file $query_path  --gt_file $gt_path \
                                -K $K -L 20 40 60 80 100 --result_path $res_path"

echo $cmd
eval $cmd