# The name of experiment
export CUDA_VISIBLE_DEVICES=0
name=vlt5_ofa_dialog_sp_training_one_model_with_new_badsents_plus_feature
dataset=$2
split=$3

output=snap/$dataset/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/multitask_reg.py \
        --distributed --multiGPU \
        --train train \
        --valid val \
        --test test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-6 \
        --epochs 20 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output \
        --load /sharefs/baai-mrnd/yfl/codebase/Dialog/snap/refcoco+/vlt5_reg_new/0.0003/LAST \
        --num_beams 5 \
        --batch_size 256 \
        --valid_batch_size 1 \
        --dataset $dataset \
        --dataset_split $split\
        --experiment_name $name\
        --hyperparameter_search\
        --zero_shot_test\
        --use_rec\
        --dialog_round 2 \
        --last_round \
        --bad_res_path /sharefs/baai-mrnd/yfl/codebase/Dialog/src/new_generate_sent_set/vlt5_reg_new/refcoco+/vlt5_reg_new_refcoco+_bad_sent_threshold_0.5_with_bbox_new.json  \
        --test_threshold 0.5 \
        --mode 'train' \
        --dialog_sp_training\
        --use_detector
        # --no_evaluate \
        # --refine \
        # --combine_with_celoss\
        # --use_combine\
        # --rl_training\
        # --debug\
        # --use_mmi \

