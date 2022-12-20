# The name of experiment
gpu0=$4
gpu1=$5
export CUDA_VISIBLE_DEVICES=$gpu0,$gpu1
name=ddl_reg_negative_text_training_from_pretrained
dataset=$2
split=$3
# mmi_margin=$4

output=snap/$dataset/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port=$6 \
    src/reg.py \
        --distributed --multiGPU \
        --train train \
        --valid val \
        --test test \
        --optim adamw \
        --warmup_ratio 0.15 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 30 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output \
        --load /sharefs/baai-mrnd/yfl/codebase/Dialog/snap/pretrain/VLT5/Epoch30 \
        --num_beams 5 \
        --batch_size 160 \
        --valid_batch_size 32 \
        --dataset $dataset\
        --dataset_split $split\
        --experiment_name $name\
        --hyperparameter_search\
        --mode 'train' \
        --use_rec \
        --use_negative_text_training \
        --negative_text_training_data /sharefs/baai-mrnd/yfl/codebase/Dialog/src/new_generate_sent_set/ddl_vlt5_reg_baseline/refcoco+/ddl_vlt5_reg_baseline_refcoco+_bad_sent_threshold_0.5_with_bbox.json \
        # --use_mmi \
        # --mmi_margin $mmi_margin \
        # --no_evaluate \
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
