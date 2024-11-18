import torch
import torch.nn as nn    
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import vit_pytorch.vit as vit    


def get_dataloader(batch_size):
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        }
    # RandomResizedCrop():reshape image
    # RandomHorizaontalFlip(): random flip image
    # Normalize: RGB mean, std
    train_dataset = torchvision.datasets.CIFAR10('./p10_dataset', train=True, transform=data_transform["train"], download=False)
    test_dataset = torchvision.datasets.CIFAR10('./p10_dataset', train=False, transform=data_transform["val"], download=False)
    print('训练数据集长度: {}'.format(len(train_dataset)))
    print('测试数据集长度: {}'.format(len(test_dataset)))
    # DataLoader创建数据集
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader,test_dataloader


if __name__ == "__main__":
    device = torch.device("cuda:0")
    model = vit.ViT(image_size=224, patch_size=16, num_classes=10, dim=32, depth=2, heads=4, mlp_dim=32, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.)
    model.load_state_dict(torch.load("best_acc.pth"))
    model = model.to(device)
    model.eval()
    eval_loss = 0
    eval_acc = 0  
    train_dataloader, test_dataloader = get_dataloader(batch_size=64)
    
    with torch.no_grad():
        for imgs, targets in test_dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = model(imgs)

            _, pred = output.max(1)
            num_correct = (pred == targets).sum().item()

            acc = num_correct / imgs.shape[0]
            eval_acc += acc
        
        eval_losses = eval_loss / (len(test_dataloader))
        eval_acc = eval_acc / (len(test_dataloader))
    print(f"eval_acc: {eval_acc}")
