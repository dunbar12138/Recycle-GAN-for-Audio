#!./scripts/train_recycle.sh
python test.py --dataroot /scratch/kangled/RecycleGAN/audio/trainData/ --name Audio0704 --model cycle_gan  --which_model_netG resnet_6blocks   --dataset_mode spectrogram  --no_dropout --gpu 1  --how_many 100  --loadSize 256 --results_dir /scratch/kangled/RecycleGAN/results/ --checkpoints_dir /scratch/kangled/RecycleGAN/checkpoints --input_nc 2 --output_nc 2
