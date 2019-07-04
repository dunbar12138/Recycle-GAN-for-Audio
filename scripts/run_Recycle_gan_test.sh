#!./scripts/train_recycle.sh
python test.py --dataroot /scratch/kangled/RecycleGAN/flowers/01/ --name Experiment0702 --model cycle_gan  --which_model_netG resnet_6blocks   --dataset_mode unaligned  --no_dropout --gpu 1  --how_many 100  --loadSize 256 --results_dir /scratch/kangled/RecycleGAN/results/ --checkpoints_dir /scratch/kangled/RecycleGAN/checkpoints
