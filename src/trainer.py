# trainer.py

import os

import torch

import loss as module_loss
import torch.optim as optimizer_module
import torch.optim.lr_scheduler as scheduler_module
from torch.utils.data import DataLoader

from utils import MetricTracker, METRIC_NAMES


def train_val(args, logger, model, train_dataloader, valid_dataloader):

    if args.wandb:
        import wandb

    # loss 및 metric 설정
    criterion = getattr(module_loss, args.loss)().to(args.device)
    metrics = [getattr(module_loss, metric)().to(args.device) 
               for metric in args.metrics if metric != args.loss]
    metric_tracker_train = MetricTracker('loss', 
                                         *[METRIC_NAMES[metric] for metric in args.metrics if metric != args.loss])
    metric_tracker_valid = MetricTracker(*[METRIC_NAMES[metric] for metric in set([args.loss] + args.metrics)]) \
                           if valid_dataloader is not None else None
        
    # optimizer 및 scheduler 설정
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = getattr(optimizer_module, args.optimizer.type)(trainable_params, **args.optimizer.args)
    if args.lr_scheduler.enable:
        lr_scheduler = getattr(scheduler_module, args.lr_scheduler.type)(optimizer, 
                                                                         **args.lr_scheduler.args)

    # 학습 진행
    save_file_path = None
    for epoch in range(1, args.train_config.epochs + 1):
        
        ### train loop ###
        loss_train = train_1epoch(args, model, train_dataloader, criterion, optimizer, 
                           metric_tracker=metric_tracker_train, epoch=epoch)
        
        errors_train, _ = evaluate(args, model, train_dataloader, metrics, 
                                   metric_tracker=metric_tracker_train, epoch=epoch)
        
        ### valid loop ###
        if valid_dataloader is not None:
            errors_valid, _ = evaluate(args, model, valid_dataloader, [criterion] + metrics, 
                                       metric_tracker=metric_tracker_valid, epoch=epoch,
                                       return_pred=True)
        
        # scheduler 업데이트
        if args.lr_scheduler.enable:
            if args.lr_scheduler.type != 'ReduceLROnPlateau':
                lr_scheduler.step()
            elif valid_dataloader is not None:
                lr_scheduler.step(errors_valid[METRIC_NAMES[args.loss]])
            else:
                raise ValueError('ReduceLROnPlateau cannot be used without validation loss')

        
        # logger에 결과 기록 및 콘솔 출력
        logger.log(args, epoch, verbose=not(epoch % args.train_config.print_period),
                    train_loss=loss_train, train_errs=errors_train, 
                    valid_errs=errors_valid)
        
        # wandb에 결과 기록
        if args.wandb:
            wandb.log({f'Train {METRIC_NAMES[args.loss]}': loss_train}, step=epoch)
            for metric in metrics:
                metric_name = METRIC_NAMES[metric.__class__.__name__]
                wandb.log({f'Train {metric_name}': errors_train[metric_name]}, step=epoch)
            if valid_dataloader is not None:
                for metric in metrics:
                    metric_name = METRIC_NAMES[metric.__class__.__name__]
                    wandb.log({f'Valid {metric_name}': errors_valid[metric_name]}, step=epoch)
        

        # 모델 저장
        # best model 저장
        if args.train_config.save_best_model and valid_dataloader is not None:
            if metric_tracker_valid.is_best(METRIC_NAMES[args.loss]):
                save_file_path = os.path.join(args.save_dir.checkpoint, f'{args.timestamp}_{args.model}-best.pt')
                torch.save(model.state_dict(), save_file_path)
                logger.log(args, epoch, message=f'Save Model in {save_file_path}')
        # save_period에 따라 저장
        elif epoch % args.train_config.save_period == 0:
            save_file_path = os.path.join(args.save_dir.checkpoint, f'{args.timestamp}_{args.model}-e{epoch}.pt')
            torch.save(model.state_dict(), save_file_path)
            logger.log(args, epoch, message=f'Save Model in {save_file_path}')
    
    # 최종 베스트/마지막 모델로 수행한 결과 저장
    if args.train_config.save_result and save_file_path:
        model.load_state_dict(torch.load(save_file_path, weights_only=True))
        train_dataloader_noshuffle = DataLoader(train_dataloader.dataset, batch_size=train_dataloader.batch_size, shuffle=False)
        _, train_y_pred = evaluate(args, model, train_dataloader_noshuffle, return_pred=True)
        _, valid_y_pred = evaluate(args, model, valid_dataloader, return_pred=True)

    return (train_y_pred, valid_y_pred) if args.train_config.save_result else (None, None)


def test(args, logger, model, test_dataloader):
    
    # loss 및 metric 설정
    metrics = [getattr(module_loss, metric)().to(args.device) for metric in args.metrics]
    metric_tracker_test = MetricTracker(*[METRIC_NAMES[metric] for metric in set(args.metrics)])

    # test loop
    errors_test = []
    for epoch in range(1):
        
        ### test loop ###
        error_test, y_pred = evaluate(args, model, test_dataloader, metrics, 
                                      metric_tracker=metric_tracker_test, epoch=epoch,
                                      return_pred=True)
        errors_test.append(error_test)
        
        # logger에 결과 기록 및 콘솔 출력
        logger.log(args, epoch, verbose=True, test_errs=error_test)

    return errors_test, y_pred



def train_1epoch(args, model, dataloader, criterion, optimizer,
          metric_tracker, epoch=-1):
    
    model.train()
    total_loss = []
    for data in dataloader:
        
        x, y, n = data[0].to(args.device), data[1].to(args.device), data[0].size(0)
        
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y.float())
        loss.backward()
        optimizer.step()

        # 결과 기록
        if metric_tracker is not None:
            metric_tracker.update('loss', loss.item() * n, n=n, epoch=epoch)
        total_loss.append(loss.item())
    
    total_loss = sum(total_loss) / len(total_loss)

    return total_loss


def evaluate(args, model, dataloader, metrics=None, metric_tracker=None, epoch=-1, return_pred=False):
    
    model.eval()
    y_hats = []
    total_errors = {METRIC_NAMES[metric.__class__.__name__]: [] 
                    for metric in metrics} if metrics is not None else None
    with torch.no_grad():
        for data in dataloader:
            x, y, n = data[0].to(args.device), data[1].to(args.device), data[0].size(0)
            y_hat = model(x)
            if return_pred : y_hats.extend(y_hat.cpu().numpy().tolist())

            if metrics is not None:
                for metric in metrics:
                    error = metric(y_hat, y).item()
                    if metric_tracker is not None:
                        metric_tracker.update(METRIC_NAMES[metric.__class__.__name__], 
                                              error * n, n=n, epoch=epoch)
                    total_errors[METRIC_NAMES[metric.__class__.__name__]].append(error)
    
    if metrics is not None:
        total_errors = {k: sum(v) / len(v) for k, v in total_errors.items()}

    return total_errors, y_hats




if __name__ == "__main__":

    # test train_val function
    from dataset import PriceDataset

    start_date = '2013-01-01'
    end_date = '2023-10-27'

    
    train_data = PriceDataset(start_date, end_date, 
                              phase='train+valid',
                              in_columns=['USD_Price', 'Gold_Price', 'Silver_Price'],
                              out_columns=['USD_Price'],
                              input_days=5,
                              time_lag=1,
                              data_dir='data')


    test_data = PriceDataset(start_date, end_date, 
                             phase='test',
                             in_columns=['USD_Price', 'Gold_Price', 'Silver_Price'],
                             out_columns=['USD_Price'],
                             input_days=5,
                             time_lag=1,
                             data_dir='data')
    

    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(train_data, batch_size=32, shuffle=False)


    from models import MLP
    input_size = tuple(train_data.__getitem__(0)[0].size())
    output_size = train_data.__getitem__(0)[1].size(0)
    model = MLP(input_size=input_size, n_hidden_list=[512, 512, 512], output_size=output_size)


    from omegaconf import OmegaConf
    args = {
        'model': 'MLP',
        'device': 'cpu',
        'wandb': False,
        'loss': 'MSELoss',
        'metrics': ['MAELoss'],
        'optimizer': {'type': 'Adam', 'args': {'lr': 1e-3}},
        'lr_scheduler': {'enable': True, 'type': 'ReduceLROnPlateau', 'args': {'factor': 0.1, 'patience': 5}},
        'train_config': {'epochs': 20, 'save_period': 5, 'save_best_model': False, 'print_period': 2, 'save_result': True},
        'save_dir': {'checkpoint': 'saved/checkpoint'},
        'timestamp' : 'test'
    }
    args = OmegaConf.create(args)

    from utils import Logger
    log_path = 'saved/log/test'
    os.makedirs(log_path, exist_ok=True)
    logger = Logger(args, log_path)
    logger.save_args()
    os.makedirs('saved/checkpoint', exist_ok=True)

    train_pred, valid_pred = train_val(args, logger, model, train_dataloader, valid_dataloader)
