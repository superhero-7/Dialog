# The name of experiment
export CUDA_VISIBLE_DEVICES=0
name=vlt5_reg_new
dataset=$2
split=$3

output=snap/$dataset/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/reg.py \
        --distributed --multiGPU \
        --train train \
        --valid val \
        --test test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 3 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output \
        --load /raid_sda/yfl/codebase/VL-T5-REG/VL-T5/snap/pretrain/VLT5/Epoch30 \
        --num_beams 5 \
        --batch_size 512 \
        --valid_batch_size 32 \
        --dataset $dataset\
        --dataset_split $split\
        --experiment_name $name\
        --hyperparameter_search\
        --mode 'train' \
        # --no_evaluate \
        # --use_mmi \
        # --use_detector \
        # --debug\
        # --dialog_sp_training\
        # --zero_shot_test\
        # --use_rec\
        # --dialog_round 2 \
        # --last_round \
        # --bad_res_path src/generate_sent_set/REG_mmi/$dataset/REG_mmi_refcoco+_vlt5_bad_sent_threshold_0.5_with_bbox.json \
        # --refine \
        # --test_threshold 0.5 \
        # --dialog_training\
        # --combine_with_celoss\
        # --use_combine\
        # --rl_training\
        # --debug\
        # --use_mmi \

