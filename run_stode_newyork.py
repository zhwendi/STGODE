import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import time
from tqdm import tqdm
from loguru import logger

from args import args
from model import ODEGCN
from utils import generate_dataset, read_data, get_normalized_adj
from eval import masked_mae_np, masked_mape_np, masked_rmse_np

def train(loader, model, optimizer, criterion, device):
    batch_loss = 0
    for idx, (inputs, targets) in enumerate(tqdm(loader)):
        model.train()
        optimizer.zero_grad()
        #
        inputs = inputs.type(torch.cuda.FloatTensor)
        targets =targets.type(torch.cuda.FloatTensor)
        #
        # inputs = inputs.to(device)
        # targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_loss += loss.detach().cpu().item()
    return batch_loss / (idx + 1)


# def train(loader, model, optimizer, criterion, device):
#     batch_losses = []
#     for idx, (inputs, targets) in enumerate(tqdm(loader)):
#         model.train()
#         optimizer.zero_grad()
#
#         inputs = inputs.to(device)
#         targets = targets.to(device)
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#
#         # 计算每个样本的loss并存储
#         batch_losses.append(loss.detach().cpu().numpy())
#
#     # 返回一个包含每个样本loss的列表
#     return batch_losses



# @torch.no_grad()
# def eval(loader, model, std, mean, device):
#     batch_rmse_loss = 0
#     batch_mae_loss = 0
#     batch_mape_loss = 0
#     for idx, (inputs, targets) in enumerate(tqdm(loader)):
#         model.eval()
#         #
#         inputs = inputs.type(torch.cuda.FloatTensor)
#         targets = targets.type(torch.cuda.FloatTensor)
#         #
#         # inputs = inputs.to(device)
#         # targets = targets.to(device)
#         output = model(inputs)
#
#         out_unnorm = output.detach().cpu().numpy() * std + mean
#         target_unnorm = targets.detach().cpu().numpy() * std + mean
#
#         mae_loss = masked_mae_np(target_unnorm, out_unnorm, 0)
#         rmse_loss = masked_rmse_np(target_unnorm, out_unnorm, 0)
#         mape_loss = masked_mape_np(target_unnorm, out_unnorm, 0)
#         batch_rmse_loss += rmse_loss
#         batch_mae_loss += mae_loss
#         batch_mape_loss += mape_loss
#
#     return batch_rmse_loss / (idx + 1), batch_mae_loss / (idx + 1), batch_mape_loss / (idx + 1)


@torch.no_grad()
def eval(loader, model, std, mean, device):
    mae_losses = torch.zeros(loader.dataset[0][1].shape[0])
    rmse_losses = torch.zeros(loader.dataset[0][1].shape[0])
    mape_losses = torch.zeros(loader.dataset[0][1].shape[0])
    for idx, (inputs, targets) in enumerate(tqdm(loader)):
        model.eval()

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        out_unnorm = outputs.detach().cpu().numpy() * std + mean
        target_unnorm = targets.detach().cpu().numpy() * std + mean

        mae_loss = masked_mae_np(target_unnorm, out_unnorm, 0)
        rmse_loss = masked_rmse_np(target_unnorm, out_unnorm, 0)
        mape_loss = masked_mape_np(target_unnorm, out_unnorm, 0)

        mae_losses=mae_losses+(np.mean(mae_loss, axis=(0,2)))
        rmse_losses= rmse_losses+(np.mean(rmse_loss, axis=(0,2)))
        mape_losses=mape_losses+(np.mean(mape_loss, axis=(0,2)))

    return  rmse_losses/ (idx + 1), mae_losses/ (idx + 1), mape_losses/ (idx + 1)




def run_model(data, mean, std, dtw_matrix, sp_matrix, dataset_name, args, device):
    train_loader, valid_loader, test_loader = generate_dataset(data, args)

    # print(f"Number of features in the dataset: {train_loader.dataset[0][1].shape[0]}")
    # print(f"Number of features in the dataset: {train_loader.dataset[0][1].shape[1]}")

    A_sp_wave = get_normalized_adj(sp_matrix).to(device)

    A_se_wave = get_normalized_adj(dtw_matrix).to(device)


    net = ODEGCN(num_nodes=data.shape[1],
                 num_features=data.shape[2],
                 num_timesteps_input=args.his_length,
                 num_timesteps_output=args.pred_length,
                 A_sp_hat=A_sp_wave,
                 A_se_hat=A_se_wave)
    net = net.to(device)
    lr = args.lr
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()

    # best_valid_rmse = 1000
    best_valid_rmse = float('inf')

    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)


    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}=====".format(epoch))
        print('Training...')
        loss = train(train_loader, net, optimizer, criterion, device)
        torch.cuda.empty_cache()
        print('Evaluating...')
        train_rmse, train_mae, train_mape = eval(train_loader, net, std, mean, device)
        valid_rmse, valid_mae, valid_mape = eval(valid_loader, net, std, mean, device)


          # 计算valid_rmse的平均值
        if valid_rmse.mean() < best_valid_rmse:
            best_valid_rmse = valid_rmse.mean()
            print('New best results!')
            # 保存模型参数
            model_path = f'F:/AAA/STGODE/data/纽约/net_params_{os.path.basename(dataset_name)}_{args.num_gpu}.pkl'
            torch.save(net.state_dict(), model_path)


        if args.log:
            logger.info(f'\n##on train data## loss: {loss}, \n')
            logger.info("##on train data## rmse loss:")
            for i, rmse in enumerate(train_rmse):
                logger.info(f"Column {i + 1} rmse: {rmse}")
            logger.info("##on train data## mae loss:")
            for i, mae in enumerate(train_mae):
                logger.info(f"Column {i + 1} mae: {mae}")
            logger.info("##on train data## mape loss:")
            for i, mape in enumerate(train_mape):
                logger.info(f"Column {i + 1} mape: {mape}")

            logger.info("##on valid data## rmse loss:")
            for i, rmse in enumerate(valid_rmse):
                logger.info(f"Column {i + 1} rmse: {rmse}")
            logger.info("##on train data## mae loss:")
            for i, mae in enumerate(valid_mae):
                logger.info(f"Column {i + 1} mae: {mae}")
            logger.info("##on train data## mape loss:")
            for i, mape in enumerate(valid_mape):
                logger.info(f"Column {i + 1} mape: {mape}")
        else:
            print(f'\n##on train data## loss: {loss}')
            print("##on train data## rmse loss:")
            for i, rmse in enumerate(train_rmse):
                print(f"Column {i + 1} rmse: {rmse}")
            print("##on train data## mae loss:")
            for i, mae in enumerate(train_mae):
                print(f"Column {i + 1} mae: {mae}")
            print("##on train data## mape loss:")
            for i, mape in enumerate(train_mape):
                print(f"Column {i + 1} mape: {mape}")

            print("##on valid data## rmse loss:")
            for i, rmse in enumerate(valid_rmse):
                print(f"Column {i + 1} rmse: {rmse}")
            print("##on valid data## mae loss:")
            for i, mae in enumerate(valid_mae):
                print(f"Column {i + 1} mae: {mae}")
            print("##on valid data## mape loss:")
            for i, mape in enumerate(valid_mape):
                print(f"Column {i + 1} mape: {mape}")

        scheduler.step()

    net.load_state_dict(torch.load(f'F:/AAA/STGODE/data/纽约/net_params_{os.path.basename(dataset_name)}_{args.num_gpu}.pkl'))

    test_rmse, test_mae, test_mape = eval(test_loader, net, std, mean, device)

    # 记录测试集的RMSE
    # test_rmse = test_rmse


    # print(f'##on test data## rmse loss: {test_rmse}, mae loss: {test_mae}, mape loss: {test_mape}')
    print("##on test data## rmse loss:")
    for i, rmse in enumerate(test_rmse):
        print(f"Column {i + 1} rmse: {rmse}")
    print("##on test data## mae loss:")
    for i, mae in enumerate(test_mae):
        print(f"Column {i + 1} mae: {mae}")
    print("##on test data## mape loss:")
    for i, mape in enumerate(test_mape):
        print(f"Column {i + 1} mape: {mape}")

    return test_rmse, test_mae, test_mape


def main(args):
    seed = 2
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    device = torch.device('cuda:' + str(args.num_gpu)) if torch.cuda.is_available() else torch.device('cpu')

    if args.log:
        logger.add('log_{time}.log')
    options = vars(args)
    if args.log:
        logger.info(options)
    else:
        print(options)

    # Run the model for each dataset
    data1, mean1, std1, dtw_matrix1, sp_matrix1 = read_data(args, 'F:/AAA/STGODE/data/纽约')
    test_rmse1, test_mae1, test_mape1 = run_model(data1, mean1, std1, dtw_matrix1, sp_matrix1, 'newyork', args, device)

    data2, mean2, std2, dtw_matrix2, sp_matrix2 = read_data(args, 'F:/AAA/STGODE/data/纽约/newyork_STL')
    test_rmse2, test_mae2, test_mape2 = run_model(data2, mean2, std2, dtw_matrix2, sp_matrix2, 'newyork_STL', args,device)

    # Plotting the test RMSE comparison bar chart
    plt.figure(figsize=(10, 6))
    x = range(len(test_rmse1))
    width = 0.35  # 条形图的宽度
    plt.bar(x, test_rmse1, width, label='newyork', color='#B8D4E9')
    plt.bar([p + width + 0.05 for p in x], test_rmse2, width, label='newyork_STL', color='#2F7DBB')
    plt.xlabel('Column')
    plt.ylabel('Total RMSE Loss')
    plt.title('Total RMSE Loss per Column for Two Data Sets')
    plt.legend()
    plt.xticks([p + width / 2 + 0.025 for p in x], x)
    plt.tight_layout()
    # 保存图像
    plt.savefig('F:/AAA/STGODE/data/纽约/newyork_test_rmse_comparison.png')
    plt.show()

    # 创建一个包含两个数据集测试RMSE的DataFrame
    df_results = pd.DataFrame({
        'NEWYORK Test RMSE': test_rmse1,
        'NEWYORK Test MAE': test_mae1,
        'NEWYORK Test MAPE': test_mape1,
        'NEWYORK_STL Test RMSE': test_rmse2,
        'NEWYORK_STL Test MAE': test_mae2,
        'NEWYORK_STL Test MAPE': test_mape2
    })

    # 保存DataFrame到Excel文件
    df_results.to_excel('F:/AAA/STGODE/data/纽约/newyork_test_loss.xlsx', index=False)

if __name__ == '__main__':
    main(args)
