import torch
import torch.nn.functional as F
import os, argparse
import json

from torchmeta.utils.data import BatchMetaDataLoader
from collections import OrderedDict
from maml.datasets import get_benchmark_by_name
from maml.metalearners import ModelAgnosticMetaLearning


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MAML')

    # General
    parser.add_argument('folder', type=str,
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--dataset', type=str,
        choices=['sinusoid', 'omniglot', 'miniimagenet','doublenmnist','doublenmnistsequence'], default='doublenmnist',
        help='Name of the dataset (default: omniglot).')
    parser.add_argument('--output-folder', type=str, default=None,
        help='Path to the output folder to save the model.')
    parser.add_argument('--num-ways', type=int, default=5, help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--num-shots', type=int, default=10, help='Number of training example per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-shots-test', type=int, default=10, help='Number of test example per class. If negative, same as the number of training examples `--num-shots` (default: 15).')

    # Model
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels in each convolution layer of the VGG network '
        '(default: 64).')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=25,
        help='Number of tasks in a batch of tasks (default: 25).')
    parser.add_argument('--num-steps', type=int, default=1,
        help='Number of fast adaptation steps, ie. gradient descent '
        'updates (default: 1).')
    parser.add_argument('--num-epochs', type=int, default=50,
        help='Number of epochs of meta-training (default: 50).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batch of tasks per epoch (default: 100).')
    parser.add_argument('--step-size', type=float, default=0.1,
        help='Size of the fast adaptation step, ie. learning rate in the '
        'gradient descent update (default: 0.1).')
    parser.add_argument('--first-order', action='store_true',
        help='Use the first order approximation, do not use higher-order '
        'derivatives during meta-optimization.')
    parser.add_argument('--meta-lr', type=float, default=0.001,
        help='Learning rate for the meta-optimizer (optimization of the outer '
        'loss). The default optimizer is Adam (default: 1e-3).')

    # Misc
    parser.add_argument('--num-workers', type=int, default=4,
        help='Number of workers to use for data-loading (default: 6).')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')

    args = parser.parse_args()

    if args.num_shots_test <= 0:
        args.num_shots_test = args.num_shots

    device = torch.device('cuda' if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')

    if (args.output_folder is not None):
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)

        folder = os.path.join(args.output_folder,
                              time.strftime('%Y-%m-%d_%H%M%S'))
        os.makedirs(folder)

        args.folder = os.path.abspath(args.folder)
        args.model_path = os.path.abspath(os.path.join(folder, 'model.th'))
        # Save the configuration in a config.json file
        with open(os.path.join(folder, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)


    device = torch.device('cuda' if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')

    benchmark = get_benchmark_by_name(args.dataset,
                                      args.folder,
                                      args.num_ways,
                                      args.num_shots,
                                      args.num_shots_test,
                                      hidden_size=args.hidden_size)

    meta_train_dataloader = BatchMetaDataLoader(benchmark.meta_train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)
#    meta_val_dataloader = BatchMetaDataLoader(benchmark.meta_val_dataset,
#                                              batch_size=args.batch_size,
#                                              shuffle=True,
#                                              num_workers=args.num_workers,
#                                              pin_memory=True)

    benchmark = get_benchmark_by_name(args.dataset,
                                  args.folder,
                                  args.num_ways,
                                  args.num_shots,
                                  args.num_shots_test,
                                  hidden_size=args.hidden_size)

    net = benchmark.model

    from create_nmnist_small import *
    from decolle.utils import parse_args, train, test, accuracy, save_checkpoint, load_model_from_checkpoint, prepare_experiment, write_stats, cross_entropy_one_hot

    train_dl, test_dl = create_dataloader()

    #if net.with_output_layer:
    #    loss = cross_entropy_one_hot
    loss = torch.nn.CrossEntropyLoss()

    opt = torch.optim.Adamax(net.get_trainable_parameters(), lr=1.2e-4, betas = (0,.95))
    #opt = torch.optim.SGD(net.get_trainable_parameters(), lr=args.step_size)

    params = OrderedDict(net.meta_named_parameters())
    data,target = next(iter(train_dl))
    net.init_parameters(data.cuda(), params = None)

    out = next(iter(meta_train_dataloader))
    mtrain_dl = out['train']
    mtest_dl = out['train']
    net.load_state_dict(torch.load('model.th'))

    for i in range(args.num_epochs):
        print('Switch task')
        loss_ = 0
        batch_acc = 0
        for j in range(args.batch_size):
    #        for data,targets in train_dl:
            data, targets = mtrain_dl            
            idx = np.arange(data[j].shape[0])
            np.random.shuffle(idx)
            data = data[j][idx[:100]]
            targets = targets[j][idx[:100]]
    #        updated_params = OrderedDict()
    #        for k, p in params.items():
    #            if p.grad is not None:
    #                p.grad.detach_()
    #                p.grad.zero_()
    #            updated_params[k]=torch.nn.Parameter(p)
    #            updated_params[k].grad = None
            net.zero_grad()
            opt.zero_grad()
            net.train()
            #net.init(data.cuda(), 50)
            out = net(data.cuda(), params=None)
            loss_ += loss(out,targets.cuda())

            data, targets = mtest_dl            
            data = data[j]
            targets = targets[j]
            with torch.no_grad():
                out = net(data.cuda(), params=None)
                batch_acc += torch.sum(out.argmax(1) == targets.cuda()).div(float(out.shape[0]))
        loss_.backward()
        opt.step()
        #for k,p in net.meta_named_parameters():
            #print(k,p.data.mean().cpu().numpy(),p.data.std().cpu().numpy())

        print(loss_, batch_acc/args.batch_size)



#    meta_test_dataloader = BatchMetaDataLoader(benchmark.meta_test_dataset,
#                                               batch_size=args.batch_size,
#                                               shuffle=True,
#                                               num_workers=args.num_workers,
#                                               pin_memory=True)
#
#    metalearner = ModelAgnosticMetaLearning(benchmark.model,
#                                        first_order=args.first_order,
#                                        num_adaptation_steps=args.num_steps,
#                                        step_size=args.step_size,
#                                        loss_function=benchmark.loss_function,
#                                        device=device)
#
#
#    results = metalearner.evaluate(meta_val_dataloader,
#                                   max_batches=args.num_batches,
#                                   verbose=args.verbose,
#                                   desc='Test')

