# Train a VQ-VAE on ImageNet
CUDA_VISIBLE_DEVICES=0,1 python src/train.py  \
   --epoch 100 \
   --lr 1e-4 \
   --weight_decay 1e-4 \
   --opt adam \
   --batch_size 32 \
   --n_accumulate 8 \
   --dropout 0.0 \
   --gpu 6 \
   --seed 0 \
   --save_dir vqvae \
   --warmup_iters 3000 \
   --decay_iters 50000 \
   --dataset imagenet \
   --model vqvae \
   --codebook cosine \
   --f 8 \
   --wandb

# Train a VQ-VAE with the rotation trick..
python3 src/train.py \
 --epoch 100 --lr 1e-4 --weight_decay 1e-4 --opt adam \
 --batch_size 32 --n_accumulate 8 --dropout 0.0 --gpu 5 \
  --seed 0 --save_dir rot_vqvae --warmup_iters 3000 --decay_iters 50000 \
  --dataset imagenet --model rot_vqvae --codebook cosine --f 8 \
  --wandb


# Train a VQ-VAE with Hessian approximation of the gradient.
python3 src/train.py \
 --epoch 100 --lr 1e-4 --weight_decay 1e-4 --opt adam \
 --batch_size 32 --n_accumulate 8 --dropout 0.0 --gpu 0 \
  --seed 0 --save_dir vhp_vqvae --warmup_iters 3000 --decay_iters 50000 \
  --dataset imagenet --model vhp_vqvae --codebook cosine --f 8 \
  --wandb

# Train a VQ-VAE with 2 forward passes (double fp).
python3 src/train.py \
 --epoch 100 --lr 1e-4 --weight_decay 1e-4 --opt adam \
 --batch_size 32 --n_accumulate 8 --dropout 0.0 --gpu 0 \
  --seed 0 --save_dir dfp_vqvae --warmup_iters 3000 --decay_iters 50000 \
  --dataset imagenet --model dfp_vqvae --codebook cosine --f 8 \
  --wandb

