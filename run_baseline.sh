######## 기본 베이스라인 실행 스크립트 ########
# 예) $ bash run_baseline.sh
# -c : --config / -m : --model / -w : --wandb / -r : --run_name
python src/main.py  -c config/config_baseline.yaml  -m MLP  -w True  -r MLP_baseline
python src/main.py  -c config/config_baseline.yaml  -m CNN  -w True  -r CNN_baseline
python src/main.py  -c config/config_baseline.yaml  -m LSTM  -w True  -r LSTM_baseline


######## 추가 베이스라인 실행 스크립트 ########
# # 학습 없이 저장된 모델을 불러와 추론만 하고자 할 경우
# # 예) 저장된 파일명이 20240827_035641_MLP-best.pt이고, 
# #     configuration이 saved/log/20240827_035641_MLP/config.yaml에 저장되었다면,
# #     $ python main.py  -c saved/log/20240827_035641_MLP/config.yaml  -m MLP  --how p  --ckpt 20240827_035641_MLP-best.pt
# #     로 실행하면 됩니다.
# python main.py  -c saved/log/test/config.yaml  -m MLP  --how p  --ckpt saved/checkpoint/test-e20.pt
