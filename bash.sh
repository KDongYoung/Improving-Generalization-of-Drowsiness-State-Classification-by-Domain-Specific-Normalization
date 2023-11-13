
echo "B dataset"
for idx in 0 1 2 3 
do
    gpu_idx=$(($idx%2))                                                                                                                                                                                                                                                                              
    CUDA_VISIBLE_DEVICES=$gpu_idx python Code/TotalMain.py --subject_group=$idx --cuda_num=$gpu_idx --dataset_name=B --seed=2032&
done
wait
echo "B dataset End"
