import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
import torchvision.datasets as datasets
from torchvision.transforms import InterpolationMode
from timm.data import Mixup


def label_names_all(root_dir='../datasets'):
    path = os.path.join(root_dir, 'tiny-imagenet-200/words.txt')
    labels = {}
    with open(path, 'r') as file:
        for line in file:
            key, value = line.strip().split('\t')
            labels[key] = value
    return labels


def label_names_train(root_dir='../datasets'):
    path = os.path.join(root_dir, 'tiny-imagenet-200/train')
    labels = {}
    indices = {}
    idx = 0
    description = label_names_all(root_dir=root_dir)
    for item in sorted(os.listdir(path)):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            labels[item] = [idx, description[item]]
            indices[idx] = [item, description[item]]
            idx += 1
    return labels, indices


class test_dataset(Dataset):
    def __init__(self, root_dir='../datasets', transform=None):
        self.root_dir = os.path.join(root_dir, 'tiny-imagenet-200/test/images')
        self.transform = transform
        self.image_files = sorted(os.listdir(self.root_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        return image


class val_dataset(Dataset):
    def __init__(self, root_dir='../datasets', transform=None):
        self.root_dir = os.path.join(root_dir, 'tiny-imagenet-200/val/images')
        self.labels_name, self.indices = label_names_train(root_dir=root_dir)
        self.transform = transform
        self.labels_file = os.path.join(root_dir, 'tiny-imagenet-200/val/val_annotations.txt')
        self.labels = self.load_labels(self.labels_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name, label = self.labels[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def load_labels(self, labels_file):
        labels = []
        with open(labels_file, 'r') as file:
            for line in file:
                line_parts = line.strip().split()
                img_name = line_parts[0]
                label = line_parts[1]
                label = self.labels_name[label][0]
                labels.append((img_name, label))
        return labels


def get_tinyimagenet_dataloaders(
    data_dir='../datasets',
    transform_train=None,
    transform_val=None,
    transform_test=None,
    batch_size=64,
    image_size=384,
    train_size='default'
):
    # Updated transforms using bicubic interpolation and full-image resize
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.25)
        ])
    if transform_val is None:
        transform_val = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    dataset_train = datasets.ImageFolder(root=os.path.join(data_dir, 'tiny-imagenet-200/train'), transform=transform_train)
    dataset_val = val_dataset(root_dir=data_dir, transform=transform_val)
    dataset_test = test_dataset(root_dir=data_dir, transform=transform_test)

    if train_size != 'default':
        total_train = len(dataset_train)
        temp_val_size = total_train - train_size
        dataset_train, dataset_temp_val = random_split(dataset_train, [train_size, temp_val_size])
        dataset_val = ConcatDataset([dataset_temp_val, dataset_val])

    # Create Mixup/CutMix object from timm with the desired probabilities and smoothing
    num_classes = len(dataset_train.classes) if hasattr(dataset_train, 'classes') else 200
    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        label_smoothing=0.1,
        num_classes=num_classes
    )

    # Custom collate function to apply mixup/cutmix on the training batch
    def mixup_collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images, dim=0)
        targets = torch.tensor(targets, dtype=torch.long)
        images, targets = mixup_fn(images, targets)
        return images, targets

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=mixup_collate_fn)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
