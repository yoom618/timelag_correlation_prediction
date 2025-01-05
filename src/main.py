# main.py

import os
import argparse
import ast
from omegaconf import OmegaConf

import torch
import torch.utils.data as dataloader_module
import torch.optim as optimizer_module
import torch.optim.lr_scheduler as scheduler_module

import dataset as data_module
import models as model_module
from trainer import train_val, test
from utils import Setting, Logger, make_result_df, visualize_result

def main(args):
    ########## 폴더 생성 ##########
    os.makedirs(args.save_dir.log, exist_ok=True)
    os.makedirs(args.save_dir.checkpoint, exist_ok=True)
    os.makedirs(args.save_dir.result, exist_ok=True)

    ########## 시드 고정 ##########
    Setting.seed_everything(args.seed)

    ########## Logger 설정 ##########
    setting = Setting()
    log_path = setting.get_log_path(args)
    logger = Logger(args, log_path)
    logger.args.timestamp = setting.save_time

    ########## 데이터셋 생성 ##########
    # tvt: train으로 학습 -> valid로 검증 -> test로 테스트
    # tv: train으로 학습 -> valid로 검증
    # tt: train+valid로 학습 -> test로 검증
    # p: test로 예측만
    print(f'\n--------------- LOAD DATASETS ---------------')
    # 학습 및 검증 데이터셋 생성
    if args.how in ['tvt', 'tv', 'tt']:
        train_phase = 'train+valid' if args.how == 'tt' else 'train'
        valid_phase = 'test' if args.how == 'tt' else 'valid'
        
        print(f'--------------- load {train_phase} dataset ---------------')
        train_dataset = getattr(data_module, args.dataset.type)(phase=train_phase, 
                                                                **args.dataset.args)
        print(f'--------------- load {valid_phase} dataset ---------------')
        valid_dataset = getattr(data_module, args.dataset.type)(phase=valid_phase,
                                                                **args.dataset.args)
        
        train_dataloader = getattr(dataloader_module, args.dataloader.type)(train_dataset,
                                                                            shuffle=True,
                                                                            **args.dataloader.args)
        valid_dataloader = getattr(dataloader_module, args.dataloader.type)(valid_dataset,
                                                                            shuffle=False,
                                                                            **args.dataloader.args)
    
    # 테스트 데이터셋 생성
    if args.how in ['tvt', 'p']:
        print(f'--------------- load {args.eval_config.dataset_phase} dataset ---------------')
        test_dataset = getattr(data_module, args.dataset.type)(phase=args.eval_config.dataset_phase,
                                                               **args.dataset.args)
        test_dataloader = getattr(dataloader_module, args.dataloader.type)(test_dataset,
                                                                           shuffle=False,
                                                                           **args.dataloader.args)


    ########## 모델 생성 (or 불러오기) ##########
    print(f'--------------- INIT {args.model} ---------------')
    input_size = tuple(train_dataset.__getitem__(0)[0].size())    # e.g. [5, 3]
    output_size = train_dataset.__getitem__(0)[1].size(0)         # e.g. 1

    # model = MLP(args.model_args.MLP, data).to('cuda')와 동일한 코드
    model = getattr(model_module, args.model)(input_size=input_size,
                                               output_size=output_size, 
                                               **args.model_args[args.model]).to(args.device)
    
    print(model)
    

    ########## 학습 및 검증 진행 ##########
    if args.how in ['tvt', 'tv', 'tt']:
        print(f'\n--------------- TRAINING & VALIDATING {args.model} ---------------')
        
        # resume이 참일 경우, 해당 경로에서 모델 불러와서 학습 진행
        if args.train_config.resume.enable:
            model.load_state_dict(torch.load(args.train_config.resume.load_path, weights_only=True))
        
        train_y_pred, valid_y_pred = train_val(args, logger, model, train_dataloader, valid_dataloader)
        

    ########## 테스트 진행 ##########
    if args.how in ['tvt', 'p']:
        print(f'\n--------------- TESTING {args.model} ---------------')
        
        # 학습된 모델 불러오기
        if args.how == 'p':
            model_path = args.checkpoint
        else:
            if args.train_config.save_best_model:
                model_path = os.path.join(args.save_dir.checkpoint, f'{setting.save_time}_{args.model}-best.pt')
            else:
                model_path = os.path.join(args.save_dir.checkpoint, f'{setting.save_time}_{args.model}-e{args.train_config.epochs:02d}.pt')
        model.load_state_dict(torch.load(model_path, weights_only=True))

        _, test_y_pred = test(args, logger, model, test_dataloader)
    
    ########## 결과 저장 ##########
    results = []
    if args.how in ['tvt', 'tv', 'tt'] and args.train_config.save_result:
        import pandas as pd
        print(f'\n--------------- SAVE TRAINING & VALIDATION RESULTS ---------------')
        results.append(make_result_df(args, 'TRAIN', train_phase, train_dataset, train_y_pred))
        results.append(make_result_df(args, 'VALID', valid_phase, valid_dataset, valid_y_pred))

    if args.how in ['tvt', 'p'] and args.eval_config.save_result:
        import pandas as pd
        print(f'\n--------------- SAVE TESTING RESULTS ---------------')
        results.append(make_result_df(args, 'TEST', args.eval_config.dataset_phase, test_dataset, test_y_pred))

    if len(results) > 0:
        result = pd.concat(results, axis=0)
        pd.DataFrame(result).to_excel(setting.get_result_filename(args))
        logger.log(args=args, epoch=-1, message=f'Results are saved in {setting.get_result_filename(args)}')

        visualize_result(result)

    ########## Logger 닫기 ##########
    logger.save_args()
    logger.close()



if __name__ == "__main__":
    
    print(f'--------------- BASIC SETUP ---------------')

    # 프로젝트 폴더로 이동
    src_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(os.path.dirname(src_path))  

    ############ BASIC ENVIRONMENT SETUP ############
    parser = argparse.ArgumentParser(description='parser')
    
    arg = parser.add_argument
    # str2dict = lambda x: {k:int(v) for k,v in (i.split(':') for i in x.split(','))}

    # add basic arguments
    arg('--config', '-c', '--c', type=str, default='config/config_baseline.yaml',
        # required=True,
        help='Configuration 파일을 설정합니다.')
    arg('--how', '--h', type=str,
        choices=['tvt', 'tv', 'tt', 'p'], # tvt: 학습->검증->테스트 / tv: 학습->검증 / tt: 학습(검증데이터포함)->테스트 / p: 예측만
        help='학습, 검증, 테스트를 어떻게 진행할지 설정할 수 있습니다.')
    arg('--checkpoint', '-ckpt', '--ckpt', type=str, 
        help='학습을 생략할 때 사용할 모델을 설정할 수 있습니다. 단, 하이퍼파라미터 세팅을 모두 정확하게 입력해야 합니다.')
    arg('--seed', '-s', '--s', type=int,
        help='데이터분할 및 모델 초기화 시 사용할 시드를 설정할 수 있습니다.')
    arg('--device', '-d', '--d', type=str, 
        choices=['cuda', 'cpu', 'mps'], help='사용할 디바이스를 선택할 수 있습니다.')
    arg('--model', '-m', '--m', type=str, 
        help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--wandb', '--w', '-w', type=ast.literal_eval, 
        help='wandb를 사용할지 여부를 설정할 수 있습니다.')
    arg('--wandb_project', '--wp', '-wp', type=str,
        help='wandb 프로젝트 이름을 설정할 수 있습니다.')
    arg('--run_name', '--rn', '-rn', '--r', '-r', type=str,
        help='wandb에서 사용할 run 이름을 설정할 수 있습니다.')
    
    # 딕셔너리 형태로 입력받을 수 있는 인자들
    arg('--model_args', '--ma', '-ma', type=ast.literal_eval)
    arg('--dataset', '--dset', '-dset', type=ast.literal_eval)
    arg('--dataloader', '--dl', '-dl', type=ast.literal_eval)
    arg('--optimizer', '-opt', '--opt', type=ast.literal_eval)
    arg('--loss', '-l', '--l', type=str)
    arg('--metrics', '-met', '--met', type=ast.literal_eval)
    arg('--lr_scheduler', '-lr', '--lr', type=ast.literal_eval)
    arg('--train_config', '-train', '--train', type=ast.literal_eval)
    arg('--eval_config', '-eval', '--eval', type=ast.literal_eval)
    arg('--save_dir', '-save', '--save', type=ast.literal_eval)
    
    args = parser.parse_args()
    #############################################

    ############ CONFIG w/ YAML FILE ############
    config_args = OmegaConf.create(vars(args))
    config = OmegaConf.load(args.config) if args.config else OmegaConf.create()

    # args에 있는 값이 config에 있는 값보다 우선함. (단, None이 아닌 값일 경우)
    for key in config_args.keys():
        if config_args[key] is not None:
            config.update({key: config_args[key]}, merge=True)

    # 사용되지 않는 정보 삭제
    if config.how == 'p':
        del config.wandb, config.wandb_project, config.run_name
        del config.optimizer, config.loss, config.lr_scheduler, config.train_config

    else:
        del config.checkpoint
    
        if config.wandb == False:
            del config.wandb_project, config.run_name
        
        config.model_args = OmegaConf.create({config.model : config.model_args[config.model]})
        
        config.optimizer.args = {k: v for k, v in config.optimizer.args.items() 
                                    if k in getattr(optimizer_module, config.optimizer.type).__init__.__code__.co_varnames}
        
        if config.lr_scheduler.enable == False:
            del config.lr_scheduler.type, config.lr_scheduler.args
        else:
            config.lr_scheduler.args = {k: v for k, v in config.lr_scheduler.args.items() 
                                            if k in getattr(scheduler_module, config.lr_scheduler.type).__init__.__code__.co_varnames}
        
        if config.train_config.resume.enable == False:
            del config.train_config.resume.load_path

        if config.how not in ['tvt', 'tv', 'tt']:
            del config.train_config

        if config.how not in ['tvt', 'p']:
            del config.eval_config

        
    # # Configuration 콘솔에 출력
    # print(OmegaConf.to_yaml(config))
    #####################################

    ############ W&B Setting ############
    if config.wandb:
        import wandb
        wandb.require("core")
        # https://docs.wandb.ai/ref/python/init 참고
        wandb.init(project=config.wandb_project, 
                   config=OmegaConf.to_container(config, resolve=True),
                   name=config.run_name if config.run_name else None,
                   notes=config.memo if hasattr(config, 'memo') else None,
                   tags=[config.model],
                   resume="allow")
        config.run_href = wandb.run.get_url()

        wandb.run.log_code("./src")  # src 내의 모든 파일을 업로드. Artifacts에서 확인 가능
    ##############################

    ############ MAIN ############
    main(config)

    if args.wandb:
        wandb.finish()
    ##############################

    