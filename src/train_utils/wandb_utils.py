import wandb

def wandb_watch(model):
  if wandb.run is not None:
    wandb.watch(model, log='all')

def log_trainer_metrics(rec_loss, commit_loss, val_loss, scheduler, epoch):
  if wandb.run is not None:
    wandb.log({'val_loss': val_loss, 'train_recon_loss': rec_loss, 'train_commit_loss': commit_loss,
               'lr': scheduler.get_last_lr()[0]}, step=epoch)

def init_wandb(args):
  wandb.init(
    project='your_project_here',
    config={
      # Optimization parameters.
      'epoch': args.epoch,
      'lr': args.lr,
      'warmup_iters': args.warmup_iters,
      'decay_iters': args.decay_iters,
      'weight_decay': args.weight_decay,
      'opt': args.opt,
      'batch_size': args.batch_size,
      'n_accumulate': args.n_accumulate,
      # Model parameters.
      'dropout': args.dropout,
      'seed': args.seed,
      # Misc. parameters.
      'dataset': args.dataset,
      # Trainer parameters.
      'checkpoint': args.checkpoint,
      'save_dir': args.save_dir,
    }
  )