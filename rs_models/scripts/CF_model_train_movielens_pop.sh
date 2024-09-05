#!/bin/bash
learning_rates=(0.0005)

# start_reasons_ratio=(0.01 0.01 0.1 0.01)
# end_reasons_ratio=(100 100 10 100)
# # start_epoch=(10 10 10 5 5)
# # end_epoch=(40 30 20 15 25)
# start_epoch=(10 0 0 0)
# end_epoch=(30 10 10 1)

ui_ratios=(0)
reason_ratio=1


neg_user_loss=(true)
ur_i_loss=(true)
ir_u_loss=(true)
u_ir_loss=(false)
reg_loss=(true)

pop_bucket=10
debias_coeffiecient=(5)
bucket_overlap=(3)
pop_ratio=0.01

# Loop through all arrays by index
for lr in "${learning_rates[@]}"; do
    for ui_r in "${ui_ratios[@]}"; do
        for neg in "${neg_user_loss[@]}"; do
            for ur in "${ur_i_loss[@]}"; do
                for ir in "${ir_u_loss[@]}"; do
                    for uir in "${u_ir_loss[@]}"; do
                        for rgl in "${reg_loss[@]}"; do
                            for dec in "${debias_coeffiecient[@]}"; do
                                for bo in "${bucket_overlap[@]}"; do
                                    python CF_model_train_debias.py --gpu_id=0 --dataset=cluster_results_MiniLM_UMAP20_openai_movielens --learning_rate=$lr --is_pairwise=False --ui_ratio=$ui_r --reason_ratio=$reason_ratio --neg_user_loss=$neg --ur_i_loss=$ur --ir_u_loss=$ir --u_ir_loss=$uir --reg_loss=$rgl --pop_bucket=$pop_bucket --debias_coeffiecient=$dec --bucket_overlap=$bo --pop_ratio=$pop_ratio --wandb_project=new_reason_CF
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done