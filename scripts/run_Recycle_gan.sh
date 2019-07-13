#!./scripts/train_recycle.sh
python train.py --dataroot /home/kangled/datasets/audio/trainData/Obama1_and_Obama2_cyclegan --name Audio0713_Cycle_sameVoice --model cycle_gan  --which_model_netG resnet_6blocks --which_model_netP unet_256 --dataset_mode spectrogram  --no_dropout --gpu 3 --identity 0  --pool_size 0 --checkpoints_dir /scratch/kangled/RecycleGAN/checkpoints --input_nc 2 --output_nc 2
