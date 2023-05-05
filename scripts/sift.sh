root="${HOME}/data/SIFT1M"
# data_path="${root}/sift_base.bin"
# query_path="${root}/sift_learn.bin"
# nnid_path="${root}/results/train-query_vamana_200_idx_uint32.bin"
# R=64
# Lbuild=128
# alpha=1.2
# lambda=0.75
# index_path="${root}/vamana_R${R}_L${Lbuild}_A${alpha}_f10_lambda${lambda}"

# # cmd="./tests/build_memory_index  --data_type float --dist_fn l2 --data_path $data_path --query_path $query_path --nnid_path $nnid_path \
# #                                     --index_path_prefix $index_path \
# #                                     -R $R -L $Lbuild --alpha $alpha --lambda $lambda"

# cmd="./tests/build_memory_index  --data_type float --dist_fn l2 --data_path $data_path \
#                                     --index_path_prefix $index_path \
#                                     -R $R -L $Lbuild --alpha $alpha"

# # echo $cmd
# # eval $cmd

# K=100
# query_path="${root}/sift_query.bin"
# gt_path="${root}/sift_groundtruth.bin"
# cmd="./tests/utils/compute_groundtruth  --data_type float --dist_fn l2 --base_file $data_path --query_file $query_path --gt_file $gt_path --K $K"

# # echo $cmd
# # eval $cmd

# K=10
# # query_path="${root}/sift_learn.bin"
# mkdir -p "${root}/results"
# res_path="${root}/results/dev-query_vamana"
# cmd="./tests/search_memory_index  --data_type float --dist_fn l2 --index_path_prefix $index_path --query_file $query_path  --gt_file $gt_path \
#                                 -K $K -L 20 40 60 80 100 --result_path $res_path"

# echo $cmd
# eval $cmd

# ./tests/build_memory_index  --data_type float --dist_fn l2 --data_path ${root}/sift_learn.bin --index_path_prefix ${root}/index_sift_learn_R32_L50_A1.2 -R 32 -L 50 --alpha 1.2
./tests/utils/compute_groundtruth  --data_type float --dist_fn l2 --base_file ${root}/sift_learn.bin --query_file ${root}/sift_query.bin --gt_file ${root}/sift_query_learn_gt100 --K 100
# ./tests/search_memory_index  --data_type float --dist_fn l2 --index_path_prefix ${root}/index_sift_base_R32_L50_A1.2 --query_file ${root}/sift_learn.bin -K 10 -L 100 --result_path ${root}/train_id
# ./tests/build_memory_index  --data_type float --dist_fn l2 --data_path ${root}/sift_base.bin --query_path ${root}/sift_learn.bin --nnid_path ${root}/train_id_100_idx_uint32.bin --index_path_prefix ${root}/index_sift_base_R32_L50_A1.2_lamb0.5 -R 32 -L 50 --alpha 1.2
./tests/search_memory_index  --data_type float --dist_fn l2 --index_path_prefix ${root}/index_sift_learn_R32_L50_A1.2 --query_file ${root}/sift_query.bin  --gt_file ${root}/sift_query_learn_gt100 -K 10 -L 10 20 30 40 50 100 --result_path ${root}/res