from __future__ import print_function
import os
import sys
from pathlib import Path  # if you haven't already done so
file = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from dmgp.models.dmgp_variational import DMGP
from dmgp.utils.sparse_design.design_class import HyperbolicCrossDesign
from dmgp.kernels.laplace_kernel import LaplaceProductKernel
from dataset import Dataset


class Regression:
    def __init__(self, input_dim, output_dim,
                 kernel, design_class,
                 num_mc=1, num_monte_carlo=10, batch_size=128,
                 lr=1.0,
                 gamma=0.999,
                 num_layers=2,
                 num_inducing=4,
                 hidden_dim=64,
                 seed=1,
                 use_cuda=True,
                 option='additive',
                 ):

        if torch.cuda.is_available() and use_cuda:
            self.device = torch.device('cuda:0')
            print("Using CUDA")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        torch.manual_seed(seed)

        self.lr = lr
        self.gamma = gamma

        self.batch_size = batch_size
        self.num_mc = num_mc
        self.num_monte_carlo = num_monte_carlo

        self.model = DMGP(input_dim=input_dim,
                          output_dim=output_dim,
                          num_layers=num_layers,
                          num_inducing=num_inducing,
                          hidden_dim=hidden_dim,
                          kernel=kernel,
                          design_class=design_class,
                          option=option).to(self.device)

        self.reset_optimizer_scheduler()  # do not delete this

    def reset_optimizer_scheduler(self, ):
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.gamma)

    def train(self, train_loader):
        losses = []
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            target = target.to(self.device)
            data = data.to(self.device)

            self.optimizer.zero_grad()
            output_ = []
            kl_ = []
            for mc_run in range(self.num_mc):
                output, kl = self.model(data)
                output_.append(output)
                kl_.append(kl)
            output = torch.mean(torch.stack(output_), dim=0)
            kl = torch.mean(torch.stack(kl_), dim=0)
            nll_loss = F.mse_loss(output, target)
            # ELBO loss
            loss = nll_loss + (kl / self.batch_size)

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        return losses

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                target = target.to(self.device)
                data = data.to(self.device)

                output, kl = self.model(data)
                test_loss += F.mse_loss(output, target, reduction='sum').item() + (
                        kl / self.batch_size)  # sum up batch loss

        test_loss /= len(test_loader.dataset)

        print('Test set: Average loss: {:.4f}'.format(test_loss))

    def evaluate(self, test_loader):
        test_loss = []
        with torch.no_grad():
            for data, target in test_loader:
                target = target.to(self.device)
                data = data.to(self.device)

                predicts = []
                for mc_run in range(self.num_monte_carlo):
                    self.model.eval()
                    output, _ = self.model.forward(data)
                    loss = F.mse_loss(output, target).cpu().data.numpy()
                    test_loss.append(loss)
                    predicts.append(output.cpu().data.numpy())

                pred_mean = np.mean(predicts, axis=0)
                pred_var = np.var(predicts, axis=0)

                print('prediction mean: ', pred_mean, 'prediction var: ', pred_var)

            print('test loss: ', np.mean(test_loss))


def import_data(file):
    import pickle
    results = pickle.load(open(file, 'rb'))
    inputs, outputs = [], []
    for r in results:
        act = r[1]
        inputs.append(np.asarray([act[key] for key in act.keys()]))
        outputs.append(r[3])

    return np.array(inputs), np.array(outputs)


def main():
    dir_name = os.path.abspath(os.path.dirname(__file__))
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch simple sparse DGP regression')
    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        choices=['train', 'test'],
                        help='train | test')
    parser.add_argument('--model',
                        type=str,
                        default='grid',
                        choices=['additive', 'grid'],
                        help='additive | grid')
    parser.add_argument('--input-dim',
                        type=int,
                        default=4,
                        metavar='N',
                        help='input dim size for training (default: 14)')
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=7,
                        metavar='N',
                        help='hidden dim of the hidden layers')
    parser.add_argument('--num_layers',
                        type=int,
                        default=2,
                        metavar='N',
                        help='depth of the model')
    parser.add_argument('--num_inducing',
                        type=int,
                        default=3,
                        metavar='N',
                        help='number of inducing levels')
    parser.add_argument('--num_mc',
                        type=int,
                        default=5,
                        metavar='N',
                        help='number of Monte Carlo runs during training')
    parser.add_argument('--num_monte_carlo',
                        type=int,
                        default=20,
                        metavar='N',
                        help='number of Monte Carlo samples to be drawn for inference')
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr',
                        type=float,
                        default=1.0,
                        metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.999,
                        metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save_dir',
                        type=str,
                        default=os.path.join(dir_name, "checkpoint/bayesian"))
    parser.add_argument(
        '--tensorboard',
        action="store_true",
        help=
        'use tensorboard for logging and visualization of training progress')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs/main_bnn',
        metavar='N',
        help=
        'use tensorboard for logging and visualization of training progress')

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    ############################################################################################################
    inputs = np.random.random((1000, args.input_dim))
    outputs = np.sum(inputs, axis=-1)
    inputs = inputs.astype(np.float32)
    outputs = np.expand_dims(outputs, axis=1).astype(np.float32)

    train_loader = torch.utils.data.DataLoader(Dataset(inputs, outputs), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(Dataset(inputs, outputs), batch_size=args.batch_size, shuffle=True)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ############################################################################################################
    model = Regression(input_dim=inputs.shape[-1], output_dim=1,
                       kernel=LaplaceProductKernel(1.),
                       design_class=HyperbolicCrossDesign,
                       batch_size=args.batch_size, lr=args.lr, gamma=args.gamma,
                       num_layers=args.num_layers, num_inducing=args.num_inducing, hidden_dim=args.hidden_dim,
                       use_cuda=True, option=args.model)

    print(args.mode)
    if args.mode == 'train':
        losses = []
        for epoch in range(args.epochs):
            print("epoch " + str(epoch), end=', ')
            loss = model.train(train_loader)
            model.scheduler.step()
            model.test(test_loader)
            losses += loss
            if epoch % 10 == 0:
                if args.model == 'grid':
                    torch.save(model.model.state_dict(), args.save_dir + "/simple_dgp_regress_grid.pth")
                elif args.model == 'additive':
                    torch.save(model.model.state_dict(), args.save_dir + "/simple_dgp_regress_additive.pth")

        plt.plot(losses)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss')

        figure_dir = os.path.join(dir_name, "figures")
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
        if args.model == 'grid':
            savefigure_path = os.path.join(figure_dir, "dgp_regress_training_grid.png")
        else:
            savefigure_path = os.path.join(figure_dir, "dgp_regress_training_additive.png")
        plt.savefig(savefigure_path, format='png', dpi=300)

    elif args.mode == 'test':
        if args.model == 'grid':
            checkpoint = args.save_dir + '/simple_dgp_regress_grid.pth'
        else:
            checkpoint = args.save_dir + '/simple_dgp_regress_additive.pth'
        model.model.load_state_dict(torch.load(checkpoint))
        model.evaluate(train_loader)
        model.evaluate(test_loader)


if __name__ == '__main__':
    main()