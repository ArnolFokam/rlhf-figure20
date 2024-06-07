# python train_reward_model.py --train_dataset hh-rlhf --max_seq_len 1024 --seed 0
# python train_reward_model.py --train_dataset hh-rlhf --max_seq_len 1024 --seed 1
# python train_reward_model.py --train_dataset hh-rlhf --max_seq_len 1024 --seed 3

python train_reward_model.py --train_dataset sentiment --max_seq_len 1024 --seed 0
# python train_reward_model.py --train_dataset sentiment --max_seq_len 1024 --seed 1
# python train_reward_model.py --train_dataset sentiment --max_seq_len 1024 --seed 3

# python train_reward_model.py --train_dataset mix --max_seq_len 1024 --seed 0
# python train_reward_model.py --train_dataset mix --max_seq_len 1024 --seed 1
# python train_reward_model.py --train_dataset mix --max_seq_len 1024 --seed 3