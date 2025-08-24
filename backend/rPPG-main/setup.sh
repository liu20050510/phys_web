conda remove --name RhythmMamba --all -y
conda create -n rppg python=3.8 pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

python main.py --config_file ./configs/user.yaml

python run.py --data_path "./new/"