import numpy as np
from utils import *
import random
import vit
import MyData
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=1024, help='the dimension of the model')
parser.add_argument('--depth', type=int, default=8, help='the number of Transformer blocks')
parser.add_argument('--heads', type=int, default=12, help='the number of heads')
parser.add_argument('--mlp_dim', type=int, default=2048, help='the dimension of mlp block')
parser.add_argument('--dim_head', type=int, default=64, help='the dimension of queries, keys, and values')
parser.add_argument('--emb_dropout', type=float, default=0.1, help='dropout rate of positional embedding')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate of Transformer block')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
parser.add_argument('--wd', type=float, default=0.0001, help='weight decay')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--kfold', type=int, default=10, help='the fold number of cross validation')
parser.add_argument('--threshold', type=float, default=0.5, help='threshold')
parser.add_argument('--gamma', type=int, default=2, help='gamma value of focal loss function')
parser.add_argument('--alpha', type=float, default=0.71, help='alpha value of focal loss function')
parser.add_argument('--device', type=str, default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--seed', type=int, default=42, help='seed')
config = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(train_data, train_target, test_data, test_target):
    # set seed
    setup_seed(42)

    print(f'===================================New Training===================================')
    print("Device: ", config.device)
    print("Seed: ", config.seed)

    # load data
    dataset = MyData.MyDataset(train_data, train_target, test_data, test_target)
    train_dataset, test_dataset = MyData.TensorDataset(dataset)

    print('length of train dataset:', int(len(train_dataset)))
    print('number of essential protein in train_dataset:', int(train_dataset.tensors[1].sum()))
    print('length of test dataset:', int(len(test_dataset)))
    print('number of essential protein in test_dataset:', int(test_dataset.tensors[1].sum()))


    # dataloader
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # model
    model = vit.ViT(
        image_size=224,
        patch_size=16,
        num_classes=1,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        dim_head=config.dim_head,
        dropout=config.dropout,
        emb_dropout=config.emb_dropout
    )
    if torch.cuda.is_available():
        model = model.to(config.device)

    # loss function and optimizer
    loss_func, optimizer = compile(config, model)

    # early stopping
    patience = 15

    # save model
    save = './output'

    # 10-fold cross validation
    val_auprs, test_auprs = [], []
    val_aucs, test_aucs = [], []

    test_trues, kfold_test_scores, kfold_test_loss, kfold_test_threshold = [], [], [], []
    # kfold_test_trues = []

    kfold = config.kfold
    skf = StratifiedKFold(n_splits=kfold, random_state=config.seed, shuffle=True)
    for i, (train_index, val_index) in enumerate(skf.split(train_dataset.tensors[0], train_dataset.tensors[1])):
        print(f'\nStart training CV fold {i + 1}:')
        train_sampler, val_sampler = SubsetRandomSampler(train_index), SubsetRandomSampler(val_index)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, shuffle=False)
        val_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=val_sampler, shuffle=False)

        # Train model
        count = 0
        best_val_aupr, best_test_aupr = .0, .0
        best_val_auc, best_test_auc = .0, .0
        min_valid_loss = float('Inf')

        # best_test_trues = []
        best_test_scores = []
        best_test_loss = .0
        best_test_threshold = .0

        best_model = model
        for epoch in range(config.epochs):
            print("------------------------epoch:{}------------------------".format(epoch + 1))
            # Calculate prediction results and losses in [train_one_epoch()  val_one_epoch() val/test_one_epoch()]
            train_trues, train_scores, train_loss, total_train_loss, train_acc, train_pre, train_rec, train_f1, train_auc, train_aupr = train_one_epoch(
                config=config,
                model=model,
                train_loader=train_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                threshold=config.threshold
            )

            val_trues, val_scores, val_loss, total_val_loss, val_acc, val_pre, val_rec, val_f1, val_auc, val_aupr = val_one_epoch(
                config=config,
                model=model,
                val_loader=val_loader,
                loss_func=loss_func,
                threshold=config.threshold
            )

            test_trues, test_scores, test_loss, total_test_loss, test_acc, test_pre, test_rec, test_f1, test_auc, test_aupr= val_one_epoch(
                config=config,
                model=model,
                val_loader=test_loader,
                loss_func=loss_func,
                threshold=config.threshold
            )


            # # save the best model by min loss
            # if val_loss < min_valid_loss:
            #     count = 0
            #     min_valid_loss = val_loss
            #     best_model = model
            #     best_val_auc = val_auc
            #     best_val_aupr = val_aupr
            #
            #     best_test_auc = test_auc
            #     best_test_aupr = test_aupr
            #     best_test_scores = test_scores
            #     best_test_trues = test_trues
            #     best_test_loss = test_loss
            #
            #     print("Found new best model. Min val loss is:{:.6f}   Validation AUC is: {:.6f}. ".format(val_loss, val_auc))
            # else:
            #     count += 1
            #     if count >= patience:
            #         torch.save(best_model, os.path.join(save, 'model_{}_{:.3f}_{:.3f}.pkl'.format(i + 1, best_test_auc, best_test_aupr)))
            #         print(f'Fold {i + 1} training done!!!\n')
            #         break

            # Save the model by auc
            if val_auc > best_val_auc:
                count = 0
                best_model = model
                best_val_auc = val_auc
                best_val_aupr = val_aupr

                best_test_auc = test_auc
                best_test_aupr = test_aupr
                best_test_scores = test_scores
                # best_test_trues = test_trues
                best_test_loss = test_loss

                print("Found new best model. Validation AUC is: {:.6f}. ".format(val_auc))

                # compute best threshold
                test_threshold, f1 = best_f1_thr(val_trues, val_scores)
                best_test_threshold = test_threshold
                print('The best f1 threshold is {:.2f} with the best f1({:.3f}).'.format(test_threshold, f1))
            else:
                count += 1
                if count >= patience:
                    torch.save(best_model, os.path.join(save, 'model_{}_{:.3f}_{:.3f}.pkl'.format(i + 1, best_test_auc, best_test_aupr)))
                    print(f'Fold {i + 1} training done!!!\n')

                    break

        val_auprs.append(best_val_aupr)
        test_auprs.append(best_test_aupr)
        val_aucs.append(best_val_auc)
        test_aucs.append(best_test_auc)

        kfold_test_scores.append(best_test_scores)
        kfold_test_loss.append(best_test_loss)
        kfold_test_threshold.append(best_test_threshold)

        # kfold_test_trues.append(best_test_trues)          # copy test_trues for kfold times

        # print('save kfold_test_scores!')
        np.save(os.path.join(save, 'kfold_test_scores.npy'), np.array(kfold_test_scores))
        # np.save('kfold_test_trues.npy', np.array(kfold_test_trues))
        np.save(os.path.join(save, 'kfold_test_threshold.npy'), np.array(kfold_test_threshold))

    print(f'Finish training.\n')
    for i, (test_auc, test_aupr, thd) in enumerate(zip(test_aucs, test_auprs, kfold_test_threshold)):
        print('Fold {}: test AUC:{:.4f}   test AUPR:{:.4f}.     best_threshold:{:.2f}'.format(i+1, test_auc, test_aupr, thd))

    # Average kfold models' results
    final_test_scores = np.sum(np.array(kfold_test_scores), axis=0) / kfold
    final_test_loss = np.sum(np.array(kfold_test_loss), axis=0) / kfold
    final_test_threshold = np.sum(np.array(kfold_test_threshold), axis=0) / kfold

    # save test_trues final_test_scores and final_test_loss to calculate all metrics_threshold in plot_ROC_PR.py
    print('save test_trues, final_test_scores and final_test_loss!')
    np.save(os.path.join(save, 'test_trues.npy'), np.array(test_trues))
    np.save(os.path.join(save, 'final_test_scores.npy'), np.array(final_test_scores))
    np.save(os.path.join(save, 'final_test_loss.npy'), np.array(final_test_loss))
    np.save(os.path.join(save, 'final_test_threshold.npy'), np.array(final_test_threshold))



    # use from here
    # # Cal the best threshold(f1)
    # best_f1_threshold, best_f1 = best_f1_thr(test_trues, final_test_scores)
    # print('The best f1 threshold is {:.2f} with the best f1({:.3f}).'.format(best_f1_threshold, best_f1))
    #
    # # Select the best threshold by f1
    # final_test_metrics = compute_metrics(test_trues, final_test_scores, best_f1_threshold)[:]
    # print_metrics('Final test', final_test_loss, final_test_metrics)

    # Select the best threshold by best_test_threshold
    print('The best f1 threshold is {:.2f}.'.format(final_test_threshold))
    final_test_metrics = compute_metrics(test_trues, final_test_scores, final_test_threshold)[:]
    print_metrics('Final test', final_test_loss, final_test_metrics)


if __name__ == '__main__':
    # input dataset：PPI5093、PPI3672、PPI2708

    # # PPI5093
    # config.alpha = 0.77
    # train_data = np.load('dataset/PPI5093/dim224/train_test/train_set/train_data.npy')
    # train_target = np.load('dataset/PPI5093/dim224/train_test/train_set/train_target.npy')
    # test_data = np.load('dataset/PPI5093/dim224/train_test/test_set/test_data.npy')
    # test_target = np.load('dataset/PPI5093/dim224/train_test/test_set/test_target.npy')

    # PPI3672
    # config.alpha = 0.74
    # train_data = np.load('dataset/PPI3672/dim224/train_test/train_set/train_data.npy')
    # train_target = np.load('dataset/PPI3672/dim224/train_test/train_set/train_target.npy')
    # test_data = np.load('dataset/PPI3672/dim224/train_test/test_set/test_data.npy')
    # test_target = np.load('dataset/PPI3672/dim224/train_test/test_set/test_target.npy')

    # PPI2708
    config.alpha = 0.71
    train_data = np.load('dataset/PPI2708/dim224/train_test/train_set/train_data.npy')
    train_target = np.load('dataset/PPI2708/dim224/train_test/train_set/train_target.npy')
    test_data = np.load('dataset/PPI2708/dim224/train_test/test_set/test_data.npy')
    test_target = np.load('dataset/PPI2708/dim224/train_test/test_set/test_target.npy')

    #### --------------------ablation dataset only PPI-------------------------------
    # ablation only PPI: PPI5093
    # config.alpha = 0.77
    # train_data = np.load('ablation_dataset/ablation_PPI_only/PPI5093/dim224/train_test/train_set/train_data.npy')
    # train_target = np.load('ablation_dataset/ablation_PPI_only/PPI5093/dim224/train_test/train_set/train_target.npy')
    # test_data = np.load('ablation_dataset/ablation_PPI_only/PPI5093/dim224/train_test/test_set/test_data.npy')
    # test_target = np.load('ablation_dataset/ablation_PPI_only/PPI5093/dim224/train_test/test_set/test_target.npy')

    # ablation only PPI: PPI3672
    # config.alpha = 0.74
    # train_data = np.load('ablation_dataset/ablation_PPI_only/PPI3672/dim224/train_test/train_set/train_data.npy')
    # train_target = np.load('ablation_dataset/ablation_PPI_only/PPI3672/dim224/train_test/train_set/train_target.npy')
    # test_data = np.load('ablation_dataset/ablation_PPI_only/PPI3672/dim224/train_test/test_set/test_data.npy')
    # test_target = np.load('ablation_dataset/ablation_PPI_only/PPI3672/dim224/train_test/test_set/test_target.npy')

    # ablation only PPI: PPI2708
    # config.alpha = 0.71
    # train_data = np.load('ablation_dataset/ablation_PPI_only/PPI2708/dim224/train_test/train_set/train_data.npy')
    # train_target = np.load('ablation_dataset/ablation_PPI_only/PPI2708/dim224/train_test/train_set/train_target.npy')
    # test_data = np.load('ablation_dataset/ablation_PPI_only/PPI2708/dim224/train_test/test_set/test_data.npy')
    # test_target = np.load('ablation_dataset/ablation_PPI_only/PPI2708/dim224/train_test/test_set/test_target.npy')

    #### --------------------ablation dataset only subcellular-------------------------------
    # ablation only sub: PPI5093
    # config.alpha = 0.77
    # train_data = np.load('ablation_dataset/ablation_sub_only/PPI5093/dim224/train_test/train_set/train_data.npy')
    # train_target = np.load('ablation_dataset/ablation_sub_only/PPI5093/dim224/train_test/train_set/train_target.npy')
    # test_data = np.load('ablation_dataset/ablation_sub_only/PPI5093/dim224/train_test/test_set/test_data.npy')
    # test_target = np.load('ablation_dataset/ablation_sub_only/PPI5093/dim224/train_test/test_set/test_target.npy')

    # # ablation only sub: PPI3672
    # config.alpha = 0.74
    # train_data = np.load('ablation_dataset/ablation_sub_only/PPI3672/dim224/train_test/train_set/train_data.npy')
    # train_target = np.load('ablation_dataset/ablation_sub_only/PPI3672/dim224/train_test/train_set/train_target.npy')
    # test_data = np.load('ablation_dataset/ablation_sub_only/PPI3672/dim224/train_test/test_set/test_data.npy')
    # test_target = np.load('ablation_dataset/ablation_sub_only/PPI3672/dim224/train_test/test_set/test_target.npy')

    # ablation only sub: PPI2708
    # config.alpha = 0.71
    # train_data = np.load('ablation_dataset/ablation_sub_only/PPI2708/dim224/train_test/train_set/train_data.npy')
    # train_target = np.load('ablation_dataset/ablation_sub_only/PPI2708/dim224/train_test/train_set/train_target.npy')
    # test_data = np.load('ablation_dataset/ablation_sub_only/PPI2708/dim224/train_test/test_set/test_data.npy')
    # test_target = np.load('ablation_dataset/ablation_sub_only/PPI2708/dim224/train_test/test_set/test_target.npy')


    train(train_data, train_target, test_data, test_target)
