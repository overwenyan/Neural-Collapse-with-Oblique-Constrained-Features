import sys
import pickle

import torch
import scipy.linalg as scilin

# part of the code from https://github.com/tding1/Neural-Collapse 

MNIST_TRAIN_SAMPLES = (5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949)
MNIST_TEST_SAMPLES = (980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009)
CIFAR10_TRAIN_SAMPLES = 10 * (5000,)
CIFAR10_TEST_SAMPLES = 10 * (1000,)

class FCFeatures:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in):
        self.outputs.append(module_in)

    def clear(self):
        self.outputs = []


def compute_Sigma_B(mu_c_dict, mu_G):
    Sigma_B = 0
    K = len(mu_c_dict)
    for i in range(K):
        Sigma_B += (mu_c_dict[i] - mu_G).unsqueeze(1) @ (mu_c_dict[i] - mu_G).unsqueeze(0)

    Sigma_B /= K
    return Sigma_B.cpu().numpy()

def compute_Sigma_W(args, model, fc_features, mu_c_dict, dataloader, isTrain=True):

    Sigma_W = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        with torch.no_grad():
            outputs = model(inputs)

        features = fc_features.outputs[0][0]
        fc_features.clear()

        for b in range(len(targets)):
            y = targets[b].item()
            Sigma_W += (features[b, :] - mu_c_dict[y]).unsqueeze(1) @ (features[b, :] - mu_c_dict[y]).unsqueeze(0)

    if args.dataset == 'mnist':
        if isTrain:
            Sigma_W /= sum(MNIST_TRAIN_SAMPLES)
        else:
            Sigma_W /= sum(MNIST_TEST_SAMPLES)
    elif args.dataset == 'cifar10' or args.dataset == 'cifar10_random':
        if isTrain:
            Sigma_W /= sum(CIFAR10_TRAIN_SAMPLES)
        else:
            Sigma_W /= sum(CIFAR10_TEST_SAMPLES)

    return Sigma_W.cpu().numpy()

def compute_ETF(W):
    # NC2
    K = W.shape[0]
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))) / pow(K - 1, 0.5)
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.detach().cpu().numpy().item()


def compute_W_H_relation(W, mu_c_dict, mu_G):
    # NC3
    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K)
    for i in range(K):
        H[:, i] = mu_c_dict[i] - mu_G

    WH = torch.mm(W, H.to(device))
    WH /= torch.norm(WH, p='fro')
    sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device)

    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item(), H

def compute_Wh_b_relation(W, mu_G, b):
    # NC4
    Wh = torch.mv(W, mu_G)
    res_b = torch.norm(Wh - b, p='fro')
    return res_b.detach().cpu().numpy().item()


if __name__ == '__main__':
    print()
