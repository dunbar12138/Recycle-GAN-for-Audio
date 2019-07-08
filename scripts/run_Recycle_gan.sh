#!./scripts/train_recycle.sh
python train.py --dataroot /scratch/kangled/RecycleGAN/audio/trainData/Obama_and_Obama --name Audio0708 --model recycle_gan  --which_model_netG resnet_6blocks --which_model_netP unet_256 --dataset_mode triplet_spectrogram  --no_dropout --gpu 2 --identity 0  --pool_size 0 --checkpoints_dir /scratch/kangled/RecycleGAN/checkpoints --input_nc 2 --output_nc 2
