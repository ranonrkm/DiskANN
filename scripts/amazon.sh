root="${HOME}/data/XC/LF-AmazonTitles-131K"
data_path="${root}/data/DiskANN_data/lbl.bin"
R=100
Lbuild=300
alpha=1.2
index_path="${root}/data/DiskANN_data/lbl_R${R}_L${Lbuild}_A${alpha}"

cmd="./tests/build_memory_index  --data_type float --dist_fn l2 --data_path $data_path --index_path_prefix $index_path \
                                    -R $R -L $Lbuild --alpha $alpha "
echo $cmd
eval $cmd

K=10
query_path="${root}/data/DiskANN_data/tst.bin"
gt_path="${root}/data/DiskANN_data/tst_gt.bin"
cmd="./tests/utils/compute_groundtruth  --data_type float --dist_fn l2 --base_file $data_path --query_file $query_path --gt_file $gt_path --K $K"

echo $cmd
eval $cmd

res_path="${root}/results/DiskANN/dev_vamana"
cmd="./tests/search_memory_index  --data_type float --dist_fn l2 --index_path_prefix $index_path --query_file $query_path  --gt_file $gt_path \
                                -K $K -L 10 20 30 40 50 100 --result_path $res_path"

echo $cmd
eval $cmd