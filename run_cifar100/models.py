import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import geoopt

from torchvision.models import resnet18
# from validate_NC import *

class ResNetAdapt(nn.Module):
    """
    X = resnet18(X), ending at avgpool(.)
    X = fc(X), 512 -> M, supporting Oblique X
    X = last_layer(X) M -> K, supporting Oblique W
    """

    def __init__(self, feature_dim, num_classes,
                 oblique_feature=False, oblique_weight=False, weight_alpha=1):
        super().__init__()
        self.resnet = resnet18()
        self.feature_dim = feature_dim
        self.resnet.fc = nn.Linear(self.resnet.fc.weight.shape[1], feature_dim,
                                   bias=True)
        self.num_classes = num_classes
        self.oblique_feature = oblique_feature
        self.oblique_weight = oblique_weight
        self.weight_alpha = weight_alpha

        if oblique_weight:
            W = geoopt.ManifoldTensor(torch.randn((self.num_classes, self.feature_dim)),
                                      manifold=geoopt.Sphere())  # (K, M), row-wise
            manifold = W.manifold
            self.last_W = geoopt.ManifoldParameter(manifold.projx(W), manifold=geoopt.Sphere())
        else:
            W = torch.empty((self.num_classes, self.feature_dim))
            nn.init.kaiming_normal_(W)
            self.last_W = geoopt.ManifoldParameter(W)

        b = torch.empty((num_classes,))
        nn.init.uniform_(b)
        self.last_b = geoopt.ManifoldParameter(b)  # (K,)

    def forward(self, x):
        # x: (B, M), W: (K, M)
        x = self.resnet(x)
        f_norm = x.norm()
        if self.oblique_feature:
            norm = x.norm(dim=-1, keepdim=True)
            x = x / norm

            with torch.no_grad():
                assert np.allclose(x.detach().cpu().norm(dim=-1), 1)

        # print(f"inside model: W: {self.last_W.shape}, x: {x.shape}")
        x = x @ self.last_W.T + self.last_b

        return x, f_norm

    @torch.no_grad()
    def get_feature(self, x):
        x = self.resnet(x)
        if self.oblique_feature:
            norm = x.norm(dim=-1, keepdim=True)
            x = x / norm

        return x

    @torch.no_grad()
    def get_last_weight_bias(self, x):
        return self.last_W, self.last_b

    @torch.no_grad()
    def plot_last_W(self):
        # assert self.feature_dim in [2, 3], "only can plot 2D or 3D weights"

        if self.feature_dim == 2:
            fig, axis = plt.subplots()
            W = self.last_W.detach().cpu().numpy()
            axis.scatter(W[:, 0], W[:, 1])
            axis.set_aspect("equal")
            axis.grid(True)
            axis.scatter([0], [0], marker="*")
            plt.show()

if __name__ == '__main__':
    M, K = 100, 5 # M means manifold, K means class_numbers
    model = ResNetAdapt(M, K, oblique_feature=True, oblique_weight=True)
    # shape X: torch.Size([128, 3, 32, 32])   shape y: torch.Size([128])
    X_under_train = torch.randn(128, 3, 32, 32)
    y_under_train, fout_under_train = model(X_under_train)
    # print("y_under_train={}, feature={}".format(y_under_train, model.get_feature(X_under_train)))

    print("y_shape: {}, fout_shape: {}".format(y_under_train.shape, model.get_feature(X_under_train).shape))
    # print("weight: {}, weight_shape: {}".format(
            # model.get_last_weight_bias(X_under_train)[0], model.get_last_weight_bias(X_under_train)[0].shape))
    weight_train, bias_train = model.get_last_weight_bias(X_under_train)
    feature_train = model.get_feature(X_under_train)
    print("norm_weight: {}".format(torch.norm(weight_train[0])))
    print("feature weight: {}".format(torch.norm(feature_train[0]) ) )

    opt = geoopt.optim.RiemannianAdam([param for param in model.parameters() if param.requires_grad])
    step_size = 10
    gamma = 0.8
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    # print("model structure: {}".format(model))

    fc_features = FCFeatures()



    print("NC2: compute_ETF: {}".format( compute_ETF(W=weight_train) ))
    
    # print("NC3: compute_W_H_relation: {}".format( compute_W_H_relation(W=weight_train, mu_c_dict=, mu_G=) ))

