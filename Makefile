pretrain_mae_base_patch16_224_400e:
	python -m torch.distributed.launch --nproc_per_node=8 run_mae_pretraining.py \
		--data_path /opt/data/common/ImageNet/ILSVRC2012 \
		--mask_ratio 0.75 \
		--model pretrain_mae_base_patch16_224 \
		--batch_size 256 \
		--opt adamw \
		--opt_betas 0.9 0.95 \
		--warmup_epochs 10 \
		--epochs 400 \
		--output_dir /xiangli/MAE-pytorch/output/pretrain_mae_base_patch16_224_400e 

finetune_mae_base_patch16_224_400e_50e:
	python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
		--data_path /opt/data/common/ImageNet/ILSVRC2012 \
		--model vit_base_patch16_224 \
		--finetune /xiangli/MAE-pytorch/output/pretrain_mae_base_patch16_224_400e/checkpoint-399.pth \
		--batch_size 128 \
		--opt adamw \
		--opt_betas 0.9 0.999 \
		--weight_decay 0.05 \
		--epochs 50 \
		--dist_eval \
		--output_dir /xiangli/MAE-pytorch/output/finetune_base_patch16_224_400e_50e

finetune_mae_base_patch16_224_400e_100e:
	python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
		--data_path /opt/data/common/ImageNet/ILSVRC2012 \
		--model vit_base_patch16_224 \
		--finetune /xiangli/MAE-pytorch/output/pretrain_mae_base_patch16_224_400e/checkpoint-399.pth \
		--batch_size 128 \
		--opt adamw \
		--opt_betas 0.9 0.999 \
		--weight_decay 0.05 \
		--epochs 100 \
		--dist_eval \
		--output_dir /xiangli/MAE-pytorch/output/finetune_base_patch16_224_400e_100e

pretrain_mae_base_patch16_224_200e:
	python -m torch.distributed.launch --nproc_per_node=8 run_mae_pretraining.py \
		--data_path /opt/data/common/ImageNet/ILSVRC2012 \
		--mask_ratio 0.75 \
		--model pretrain_mae_base_patch16_224 \
		--batch_size 256 \
		--opt adamw \
		--opt_betas 0.9 0.95 \
		--warmup_epochs 5 \
		--epochs 200 \
		--output_dir /xiangli/MAE-pytorch/output/pretrain_mae_base_patch16_224_200e


pretrain_mae_base_target_025_patch16_224_200e:
	python -m torch.distributed.launch --nproc_per_node=8 run_mae_pretraining.py \
		--data_path /opt/data/common/ImageNet/ILSVRC2012 \
		--mask_ratio 0.75 \
		--model pretrain_mae_base_patch16_224 \
		--batch_size 256 \
		--opt adamw \
		--opt_betas 0.9 0.95 \
		--warmup_epochs 5 \
		--epochs 200 \
		--target_shrink 0.25 \
		--output_dir /xiangli/MAE-pytorch/output/pretrain_mae_base_target_025_patch16_224_200e


finetune_mae_base_patch16_224_200e_100e:
	python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
		--data_path /opt/data/common/ImageNet/ILSVRC2012 \
		--model vit_base_patch16_224 \
		--finetune /xiangli/MAE-pytorch/output/pretrain_mae_base_patch16_224_200e/checkpoint-199.pth \
		--batch_size 128 \
		--opt adamw \
		--opt_betas 0.9 0.999 \
		--weight_decay 0.05 \
		--epochs 100 \
		--dist_eval \
		--output_dir /xiangli/MAE-pytorch/output/finetune_base_patch16_224_200e_100e

finetune_mae_base_target_025_patch16_224_200e_100e:
	python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
		--data_path /opt/data/common/ImageNet/ILSVRC2012 \
		--model vit_base_patch16_224 \
		--finetune /xiangli/MAE-pytorch/output/pretrain_mae_base_target_025_patch16_224_200e/checkpoint-199.pth \
		--batch_size 128 \
		--opt adamw \
		--opt_betas 0.9 0.999 \
		--weight_decay 0.05 \
		--epochs 100 \
		--dist_eval \
		--output_dir /xiangli/MAE-pytorch/output/finetune_base_taget_025_patch16_224_200e_100e

pretrain_mae_base_target_05_patch16_224_200e:
	python -m torch.distributed.launch --nproc_per_node=8 run_mae_pretraining.py \
		--data_path /opt/data/common/ImageNet/ILSVRC2012 \
		--mask_ratio 0.75 \
		--model pretrain_mae_base_patch16_224 \
		--batch_size 256 \
		--opt adamw \
		--opt_betas 0.9 0.95 \
		--warmup_epochs 5 \
		--epochs 200 \
		--target_shrink 0.5 \
		--output_dir /xiangli/MAE-pytorch/output/pretrain_mae_base_target_05_patch16_224_200e

finetune_mae_base_target_05_patch16_224_200e_100e:
	python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
		--data_path /opt/data/common/ImageNet/ILSVRC2012 \
		--model vit_base_patch16_224 \
		--finetune /xiangli/MAE-pytorch/output/pretrain_mae_base_target_05_patch16_224_200e/checkpoint-199.pth \
		--batch_size 128 \
		--opt adamw \
		--opt_betas 0.9 0.999 \
		--weight_decay 0.05 \
		--epochs 100 \
		--dist_eval \
		--output_dir /xiangli/MAE-pytorch/output/finetune_base_taget_05_patch16_224_200e_100e
