import timeit
import argparse
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as fn
from model import MVSGDR
from data_preprocess import *
import warnings
from metric import *


warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=10, help='k-fold cross validation')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight_decay')
    parser.add_argument('--random_seed', type=int, default=1145, help='random seed')
    # hyper Parameters
    parser.add_argument('--n_patch2', type=int, default=32, help='num of partition')
    parser.add_argument('--neighbor', type=int, default=16, help='graph sim neighbor')
    parser.add_argument('--dropout', default='0.4', type=float, help='dropout of MLP')
    parser.add_argument('--alpha', default='0.3', type=float, help='module')
    parser.add_argument('--beta', default='0.2', type=float, help='module')
    # dataset
    parser.add_argument('--dataset', default='Ldataset', help='dataset')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='train or test negative_rate')
    # embedding device
    parser.add_argument('--drug_embedding_dim', default='269', type=int, help='num of drug emb')
    parser.add_argument('--disease_embedding_dim', default='598', type=int, help='num of disease emb')
    parser.add_argument('--adj_dim', default='867', type=int, help='num of drug and disease')
    parser.add_argument('--co_embedding_dim', default='256', type=int, help='combine emb')
    parser.add_argument('--hidden_embedding_dim', default='128', type=int, help='hidden emb')
    # node number
    parser.add_argument('--drug_num', default='269', type=int, help='num of drug')
    parser.add_argument('--disease_num', default='598', type=int, help='num of disease')
    parser.add_argument('--total_num', default='867', type=int, help='num of total')

    args = parser.parse_args()
    # args.data_dir = 'dataset/data/' + args.dataset+'/'+args.dataset
    args.data_dir = 'dataset/data/' + args.dataset + '/'

    # get data
    data, _ = get_data_new(args)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']

    # data process
    data = data_processing_new(data, args)
    data = k_fold(data, args)

    # similarity graph
    drdr_graph, didi_graph, data = dgl_similarity_graph(data, args)
    drdr_graph = drdr_graph.to(device)
    didi_graph = didi_graph.to(device)

    drug_feature = torch.FloatTensor(data['drs']).to(device)
    disease_feature = torch.FloatTensor(data['dis']).to(device)

    all_sample = torch.tensor(data['all_drdi']).long()

    start = timeit.default_timer()

    cross_entropy = nn.CrossEntropyLoss()

    Metric = (
        'Epoch\t\tTime\t\tLoss\t\tAUC\t\t\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')

    print('Dataset:', args.dataset)

    for i in range(args.k_fold):

        print('fold:', i)
        print(Metric)

        model = MVSGDR(args)
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.9, patience=1000,
                                                         verbose=True, min_lr=0.001)

        best_auc, best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = 0, 0, 0, 0, 0, 0, 0
        X_train = torch.LongTensor(data['X_train'][i]).to(device)
        X_train_p = torch.LongTensor(data['X_train_p'][i]).to(device)
        X_train_n = torch.LongTensor(data['X_train_n'][i]).to(device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(device)
        X_test = torch.LongTensor(data['X_test'][i]).to(device)
        X_test_p = torch.LongTensor(data['X_test_p'][i]).to(device)
        X_test_n = torch.LongTensor(data['X_test_n'][i]).to(device)
        Y_test = data['Y_test'][i].flatten()
        # ALL_train = torch.LongTensor(data['drprdi']).t().to(device)

        graph2 = get_data2(data['X_train'][i], args.adj_dim)
        patch2 = partition_patch(graph2, n_patches=args.n_patch2)

        graph3 = get_data2(data['X_test'][i], args.adj_dim)
        patch3 = partition_patch(graph3, n_patches=args.n_patch2)

        epoch_results = []
        for epoch in range(args.epochs):

            pred, _, neg_link, neg_link_b = model(drdr_graph, didi_graph, drug_feature,
                                                  disease_feature, X_train_p, patch2, X_train_n)
            # loss = model.loss_single(pred, torch.flatten(Y_train))
            loss = model.loss_neg_balance(pred, neg_link, neg_link_b, torch.flatten(Y_train))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.detach().cpu().numpy()

            with torch.no_grad():
                model.eval()

                score, _, _, _ = model(drdr_graph, didi_graph, drug_feature, disease_feature,
                                       X_test_p, patch3, X_test_n)

            test_prob = fn.softmax(score, dim=-1)
            test_score = torch.argmax(score, dim=-1)

            test_prob = test_prob[:, 1]
            test_prob = test_prob.cpu().numpy()

            test_score = test_score.cpu().numpy()

            AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(Y_test, test_score, test_prob)

            scheduler.step(AUC)

            end = timeit.default_timer()
            time = end - start
            show = [epoch + 1, round(time, 2), loss, round(AUC, 5), round(AUPR, 5), round(accuracy, 5),
                    round(precision, 5), round(recall, 5), round(f1, 5), round(mcc, 5)]
            print('\t\t'.join(map(str, show)))
            if AUC > best_auc:
                best_epoch = epoch + 1
                best_auc = AUC
                best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = AUPR, accuracy, precision, recall, f1, mcc
                print('AUC improved at epoch ', best_epoch, ';\tbest_auc:', best_auc)
