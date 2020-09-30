import torch.nn.functional as F

from collections import namedtuple
from torchmeta.datasets import Omniglot, MiniImagenet
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose

from maml.model import ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid, ModelConvDoubleNMNIST, ModelDECOLLE
from maml.utils import ToTensor1D
from torchmeta.utils.data import CombinationMetaDataset



Benchmark = namedtuple('Benchmark', 'meta_train_dataset meta_val_dataset '
                                    'meta_test_dataset model loss_function')

def get_benchmark_by_name(name,
                          folder,
                          num_ways,
                          num_shots,
                          num_shots_test,
                          hidden_size=None):

    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test)
    if name == 'sinusoid':
        transform = ToTensor1D()

        meta_train_dataset = Sinusoid(num_shots + num_shots_test,
                                      num_tasks=1000000,
                                      transform=transform,
                                      target_transform=transform,
                                      dataset_transform=dataset_transform)
        meta_val_dataset = Sinusoid(num_shots + num_shots_test,
                                    num_tasks=1000000,
                                    transform=transform,
                                    target_transform=transform,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Sinusoid(num_shots + num_shots_test,
                                     num_tasks=1000000,
                                     transform=transform,
                                     target_transform=transform,
                                     dataset_transform=dataset_transform)

        model = ModelMLPSinusoid(hidden_sizes=[40, 40])
        loss_function = F.mse_loss

    elif name == 'omniglot':
        class_augmentations = [Rotation([90, 180, 270])]
        transform = Compose([Resize(28), ToTensor()])

        meta_train_dataset = Omniglot(folder,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_train=True,
                                      class_augmentations=class_augmentations,
                                      dataset_transform=dataset_transform,
                                      download=True)
        meta_val_dataset = Omniglot(folder,
                                    transform=transform,
                                    target_transform=Categorical(num_ways),
                                    num_classes_per_task=num_ways,
                                    meta_val=True,
                                    class_augmentations=class_augmentations,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Omniglot(folder,
                                     transform=transform,
                                     target_transform=Categorical(num_ways),
                                     num_classes_per_task=num_ways,
                                     meta_test=True,
                                     dataset_transform=dataset_transform)

        model = ModelConvOmniglot(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    elif name == 'miniimagenet':
        transform = Compose([Resize(84), ToTensor()])

        meta_train_dataset = MiniImagenet(folder,
                                          transform=transform,
                                          target_transform=Categorical(num_ways),
                                          num_classes_per_task=num_ways,
                                          meta_train=True,
                                          dataset_transform=dataset_transform,
                                          download=True)
        meta_val_dataset = MiniImagenet(folder,
                                        transform=transform,
                                        target_transform=Categorical(num_ways),
                                        num_classes_per_task=num_ways,
                                        meta_val=True,
                                        dataset_transform=dataset_transform)
        meta_test_dataset = MiniImagenet(folder,
                                         transform=transform,
                                         target_transform=Categorical(num_ways),
                                         num_classes_per_task=num_ways,
                                         meta_test=True,
                                         dataset_transform=dataset_transform)

        model = ModelConvMiniImagenet(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    elif name == 'doublenmnist':
        from torchneuromorphic.doublenmnist_torchmeta.doublenmnist_dataloaders import DoubleNMNIST,Compose,ClassNMNISTDataset,CropDims,Downsample,ToCountFrame,ToTensor,ToEventSum,Repeat,toOneHot
        from torchneuromorphic.utils import plot_frames_imshow
        from matplotlib import pyplot as plt
        from torchmeta.utils.data import CombinationMetaDataset

        root = 'data/nmnist/n_mnist.hdf5'
        chunk_size = 300
        ds = 2
        dt = 1000
        transform = None
        target_transform = None

        size = [2, 32//ds, 32//ds]

        transform = Compose([
            CropDims(low_crop=[0,0], high_crop=[32,32], dims=[2,3]),
            Downsample(factor=[dt,1,ds,ds]),
            ToEventSum(T = chunk_size, size = size),
            ToTensor()])

        if target_transform is None:
            target_transform = Compose([Repeat(chunk_size), toOneHot(num_ways)])

        loss_function = F.cross_entropy

        meta_train_dataset = ClassSplitter(DoubleNMNIST(root = root, meta_train=True, transform = transform, target_transform = target_transform, chunk_size=chunk_size,  num_classes_per_task=num_ways), num_train_per_class = num_shots, num_test_per_class = num_shots_test)
        meta_val_dataset = ClassSplitter(DoubleNMNIST(root = root, meta_val=True, transform = transform, target_transform = target_transform, chunk_size=chunk_size,  num_classes_per_task=num_ways), num_train_per_class = num_shots, num_test_per_class = num_shots_test)
        meta_test_dataset = ClassSplitter(DoubleNMNIST(root = root, meta_test=True, transform = transform, target_transform = target_transform, chunk_size=chunk_size,  num_classes_per_task=num_ways), num_train_per_class = num_shots, num_test_per_class = num_shots_test)

        model = ModelConvDoubleNMNIST(num_ways, hidden_size=hidden_size)

    elif name == 'doublenmnistsequence':
        from torchneuromorphic.doublenmnist_torchmeta.doublenmnist_dataloaders import DoubleNMNIST,Compose,ClassNMNISTDataset,CropDims,Downsample,ToCountFrame,ToTensor,ToEventSum,Repeat,toOneHot
        from torchneuromorphic.utils import plot_frames_imshow
        from matplotlib import pyplot as plt
        from torchmeta.utils.data import CombinationMetaDataset

        root = 'data/nmnist/n_mnist.hdf5'
        chunk_size = 300
        ds = 2
        dt = 1000
        transform = None
        target_transform = None

        size = [2, 32//ds, 32//ds]

        transform = Compose([
            CropDims(low_crop=[0,0], high_crop=[32,32], dims=[2,3]),
            Downsample(factor=[dt,1,ds,ds]),
            ToCountFrame(T = chunk_size, size = size),
            ToTensor()])

        if target_transform is None:
            target_transform = Compose([Repeat(chunk_size), toOneHot(num_ways)])

        loss_function = F.cross_entropy

        meta_train_dataset = ClassSplitter(DoubleNMNIST(root = root, meta_train=True, transform = transform, target_transform = target_transform, chunk_size=chunk_size,  num_classes_per_task=num_ways), num_train_per_class = num_shots, num_test_per_class = num_shots_test)
        meta_val_dataset = ClassSplitter(DoubleNMNIST(root = root, meta_val=True, transform = transform, target_transform = target_transform, chunk_size=chunk_size,  num_classes_per_task=num_ways), num_train_per_class = num_shots, num_test_per_class = num_shots_test)
        meta_test_dataset = ClassSplitter(DoubleNMNIST(root = root, meta_test=True, transform = transform, target_transform = target_transform, chunk_size=chunk_size,  num_classes_per_task=num_ways), num_train_per_class = num_shots, num_test_per_class = num_shots_test)

        model = ModelDECOLLE(num_ways)

    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(name))

    return Benchmark(meta_train_dataset=meta_train_dataset,
                     meta_val_dataset=meta_val_dataset,
                     meta_test_dataset=meta_test_dataset,
                     model=model,
                     loss_function=loss_function)
