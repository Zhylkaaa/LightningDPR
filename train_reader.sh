python reader_model.py \
--model_name_or_path allegro/herbert-base-cased \
--seed 42 \
--fp16 \
--do_train \
--do_predict \
--wandb_project train_dpr_reader \
--monitor_metric val_acc \
--output_dir dpr_reader_herbert_v1 \
--train_batch_size 4 \
--accumulate_grad_batches 1 \
--eval_batch_size 4 \
--val_check_interval 0.5 \
--warmup_steps 100 \
--num_train_epochs 20 \
--learning_rate 1e-5 \
--log_every_n_steps 50 \
--train_data $TRAIN_DATA_FILE \
--dev_data $DEV_DATA_FILE \
--test_data $TEST_DATA_FILE \
--accelerator ddp \
--gpus 2 \
--num_nodes 1 \
--num_negative_ctx 0 \
--num_hard_negative_ctx 24 \
--max_seq_len 386 \
--gradient_clip_val 2.0 \
--progress_bar_refresh_rate 1