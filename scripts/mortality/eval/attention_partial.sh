python main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision-backbone resnet34 \
--mode eval \
--epochs 50 --batch_size 16 \
--loss uncertainty_loss \
--vision_num_classes 1 --num_classes 1 \
--data_pairs partial_ehr_cxr \
--data_ratio 1.0 \
--task in-hospital-mortality \
--labels_set mortality \
--fusion_type attention \
--save_dir checkpoints/mortality/attention_partial