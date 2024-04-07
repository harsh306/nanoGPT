import time

out_dir = 'out-wiki'
eval_interval = 10
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'wiki'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'wiki'
init_from = 'gpt2' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 4
block_size = 1024
gradient_accumulation_steps = 8
max_iters = 2000

# finetune at constant LR
learning_rate = 0.00005
decay_lr = True
