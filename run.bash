# Reward modelling
python train_reward_model.py --train_dataset hh-rlhf --max_seq_len 1024 --seed 0
python train_reward_model.py --train_dataset hh-rlhf --max_seq_len 1024 --seed 1
python train_reward_model.py --train_dataset hh-rlhf --max_seq_len 1024 --seed 2

python train_reward_model.py --train_dataset sentiment --max_seq_len 1024 --seed 0
python train_reward_model.py --train_dataset sentiment --max_seq_len 1024 --seed 1
python train_reward_model.py --train_dataset sentiment --max_seq_len 1024 --seed 2

python train_reward_model.py --train_dataset mix --max_seq_len 1024 --seed 0
python train_reward_model.py --train_dataset mix --max_seq_len 1024 --seed 1
python train_reward_model.py --train_dataset mix --max_seq_len 1024 --seed 2

python train_reward_model.py --train_dataset hh-rlhf --max_seq_len 512 --seed 0
python train_reward_model.py --train_dataset hh-rlhf --max_seq_len 512 --seed 1
python train_reward_model.py --train_dataset hh-rlhf --max_seq_len 512 --seed 2

python train_reward_model.py --train_dataset sentiment --max_seq_len 512 --seed 0
python train_reward_model.py --train_dataset sentiment --max_seq_len 512 --seed 1
python train_reward_model.py --train_dataset sentiment --max_seq_len 512 --seed 2

python train_reward_model.py --train_dataset mix --max_seq_len 512 --seed 0
python train_reward_model.py --train_dataset mix --max_seq_len 512 --seed 1
python train_reward_model.py --train_dataset mix --max_seq_len 512 --seed 2

python train_reward_model.py --train_dataset hh-rlhf --max_seq_len 256 --seed 0
python train_reward_model.py --train_dataset hh-rlhf --max_seq_len 256 --seed 1
python train_reward_model.py --train_dataset hh-rlhf --max_seq_len 256 --seed 2

python train_reward_model.py --train_dataset sentiment --max_seq_len 256 --seed 0
python train_reward_model.py --train_dataset sentiment --max_seq_len 256 --seed 1
python train_reward_model.py --train_dataset sentiment --max_seq_len 256 --seed 2

python train_reward_model.py --train_dataset mix --max_seq_len 256 --seed 0
python train_reward_model.py --train_dataset mix --max_seq_len 256 --seed 1
python train_reward_model.py --train_dataset mix --max_seq_len 256 --seed 2

# RLHF
python train_rlhf_model.py --saved_pm_path logs/pm_sentiment_1024_0/model  --seed 0
python train_rlhf_model.py --saved_pm_path logs/pm_hh-rlhf_1024_0/model  --seed 0
python train_rlhf_model.py --saved_pm_path logs/pm_mix_1024_0/model  --seed 0

python train_rlhf_model.py --saved_pm_path logs/pm_sentiment_1024_1/model  --seed 1
python train_rlhf_model.py --saved_pm_path logs/pm_hh-rlhf_1024_1/model  --seed 1
python train_rlhf_model.py --saved_pm_path logs/pm_mix_1024_1/model  --seed 1

python train_rlhf_model.py --saved_pm_path logs/pm_sentiment_1024_2/model  --seed 2
python train_rlhf_model.py --saved_pm_path logs/pm_hh-rlhf_1024_2/model  --seed 2
python train_rlhf_model.py --saved_pm_path logs/pm_mix_1024_2/model  --seed 2