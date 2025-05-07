import os
import numpy as np
import torch
import torchvision
import argparse
from modules import transform, resnet, network, contrastive_loss, clip, transform_mask
from utils import yaml_config_hook, save_model, losses
from torch.utils import data
from npy import NPY
import time
from evaluation import evaluation
import copy


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index



# 训练函数
def train(data_loader, model, optimizer, criterion_instance, criterion_cluster, scl_criterion, epoch, total_epochs, lambda_s=0.1):
    loss_epoch = 0
    model.train()

    # 根据当前 epoch 动态调整置信度阈值
    confidence_threshold = 0.2 + (epoch / total_epochs) * 0.6  
    
    for step, ((x_i, x_j, x_s), _, indices) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i, x_j, x_s = x_i.cuda(), x_j.cuda(), x_s.cuda()

        # 网络前向
        z_i, z_j, z_s, c_i, c_j, c_s = model(x_i, x_j, x_s)

        # 1. 对比学习损失
        loss_instance = criterion_instance(z_i, z_j) + criterion_instance(z_j, z_s)
        loss_cluster = (criterion_cluster(c_i, c_j) + criterion_cluster(c_i, c_s) + criterion_cluster(c_j, c_s))

        # 实时生成伪标签（动态伪标签）
        with torch.no_grad():
            c = model.forward_cluster(x_i, use_instance_classifier=False)
            batch_pseudo_labels = torch.argmax(c, dim=1)
            confidence = torch.max(c, dim=1)[0]
            high_confidence_mask = confidence > confidence_threshold  # 筛选高置信度伪标签
            batch_pseudo_labels = torch.where(
                high_confidence_mask, batch_pseudo_labels, torch.full_like(batch_pseudo_labels, -1)
            )
        # loss_p = compute_pseudo_label_loss(c_i, batch_pseudo_labels) if batch_pseudo_labels is not None else 0

        # 3. 计算 SCL 损失（仅对高置信度样本）
        if high_confidence_mask.sum() > 0:
            loss_scl = scl_criterion(z_i[high_confidence_mask], 
                                     batch_pseudo_labels[high_confidence_mask], 
                                     confidence[high_confidence_mask])
        else:
            loss_scl = torch.tensor(0.0, device=z_i.device)

        # confidence_criterion = losses.ConfidenceBasedCE(confidence_threshold, apply_class_balancing=True).cuda()

        # if high_confidence_mask.sum() > 0:
        #     loss_confidence = confidence_criterion(anchors_weak=z_i[high_confidence_mask], anchors_strong=z_s[high_confidence_mask]) + confidence_criterion(anchors_weak=z_j[high_confidence_mask], anchors_strong=z_s[high_confidence_mask])
        # else:
        #     loss_confidence = torch.tensor(0.0, device=z_i.device)
        
        # # 3. 视角一致性损失（KL 散度）
        # kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
        # loss_c = kl_loss(torch.log(c_i), c_j) + kl_loss(torch.log(c_i), c_s) + kl_loss(torch.log(c_j), c_s)

        # 4. 动态调整权重
        # dynamic_lambda_p = min(0.5, lambda_p + epoch * 0.005)  # 每个 epoch 增加 0.01
        dynamic_lambda_s = min(1, lambda_s + epoch * 0.005)  # 每个 epoch 增加 0.01
        # dynamic_lambda_conf = min(1, lambda_conf + epoch * 0.005)  # 每个 epoch 增加 0.01
        # dynamic_lambda_c = max(0.5, lambda_c - epoch * 0.005)  # 每个 epoch 减少 0.005

        # 5. 总损失
        # loss = loss_instance + loss_cluster + dynamic_lambda_p * loss_p + dynamic_lambda_c * loss_c
        loss = loss_instance + loss_cluster + dynamic_lambda_s * loss_scl   # + dynamic_lambda_conf * loss_confidence
        loss.backward()
        optimizer.step()

        # 打印损失
        if step % 50 == 0:
            print(f"Step [{step}/{len(data_loader)}]\t"
                  f"loss_instance: {loss_instance.item():.4f}\t"
                  f"loss_cluster: {loss_cluster.item():.4f}\t"
                  f"loss_scl: {loss_scl.item():.4f}\t")
        loss_epoch += loss.item()
        
    return loss_epoch




def inference(loader, model, device):
    model.eval()
    feature_vector = []
    labels_vector = []
    for step, (x, y, ) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            c = model.forward_cluster(x, use_instance_classifier=False)
            c = torch.argmax(c, dim=1)
        c = c.detach()
        feature_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))

    if args.dataset == "CIFAR-100":  # super-class
        super_label = [
            [72, 4, 95, 30, 55],
            [73, 32, 67, 91, 1],
            [92, 70, 82, 54, 62],
            [16, 61, 9, 10, 28],
            [51, 0, 53, 57, 83],
            [40, 39, 22, 87, 86],
            [20, 25, 94, 84, 5],
            [14, 24, 6, 7, 18],
            [43, 97, 42, 3, 88],
            [37, 17, 76, 12, 68],
            [49, 33, 71, 23, 60],
            [15, 21, 19, 31, 38],
            [75, 63, 66, 64, 34],
            [77, 26, 45, 99, 79],
            [11, 2, 35, 46, 98],
            [29, 93, 27, 78, 44],
            [65, 50, 74, 36, 80],
            [56, 52, 47, 59, 96],
            [8, 58, 90, 13, 48],
            [81, 69, 41, 89, 85],
        ]
        labels_vector_copy = copy.copy(labels_vector)
        for i in range(20):
            for j in super_label[i]:
                labels_vector[labels_vector_copy == j] = i
            
    # 动态评估聚类指标
    nmi, ari, f, acc = evaluation.evaluate(labels_vector, feature_vector)
    print(f"NMI = {nmi:.4f}, ARI = {ari:.4f}, F = {f:.4f}, ACC = {acc:.4f}")
    return nmi, ari, f, acc


if __name__ == "__main__":
    global best_acc  # 引用全局变量
    best_acc = 0.0
    
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("SACC-main/config/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # prepare data
    if args.dataset == "CIFAR-10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform_mask.Transforms(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform_mask.Transforms(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "CIFAR-100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform_mask.Transforms(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform_mask.Transforms(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 20
    elif args.dataset == "ImageNet-10":
        dataset = NPY(
            root=args.dataset_dir, download=True, 
            transform=transform_mask.Transforms(size=args.image_size, blur=True),
        )
        class_num = 10
    elif args.dataset == "ImageNet-dogs":
        dataset = NPY(
            root=args.dataset_dir, download=True, 
            transform=transform_mask.Transforms(size=args.image_size, blur=True),
        )
        class_num = 15
    elif args.dataset == "tiny-ImageNet":
        dataset = torchvision.datasets.ImageFolder(
            root='~/datasets/tiny-imagenet-200/train',
            transform=transform_mask.Transforms(s=0.5, size=args.image_size),
        )
        class_num = 200
    else:
        raise NotImplementedError

    dataset = IndexedDataset(dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    #####eval######
    if args.dataset == "CIFAR-10":
        train_dataset1 = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            train=True,
            download=True,
            transform=transform_mask.Transforms(size=args.image_size).test_transform,
        )
        test_dataset1 = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            train=False,
            download=True,
            transform=transform_mask.Transforms(size=args.image_size).test_transform,
        )
        dataset1 = data.ConcatDataset([train_dataset1, test_dataset1])
        class_num = 10
    elif args.dataset == "CIFAR-100":
        train_dataset1 = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform_mask.Transforms(size=args.image_size).test_transform,
        )
        test_dataset1 = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform_mask.Transforms(size=args.image_size).test_transform,
        )
        dataset1 = data.ConcatDataset([train_dataset1, test_dataset1])
        class_num = 20
    elif args.dataset == "STL-10":
        train_dataset1 = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="train",
            download=True,
            transform=transform_mask.Transforms(size=args.image_size).test_transform,
        )
        test_dataset1 = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="test",
            download=True,
            transform=transform_mask.Transforms(size=args.image_size).test_transform,
        )
        dataset1 = torch.utils.data.ConcatDataset([train_dataset1, test_dataset1])
        class_num = 10
    elif args.dataset == "ImageNet-10":
        dataset1 = NPY(
            root=args.dataset_dir, download=True, 
            transform=transform_mask.Transforms(size=args.image_size, blur=True).test_transform,
        )
        class_num = 10
    elif args.dataset == "ImageNet-dogs":
        dataset1 = NPY(
            root=args.dataset_dir, download=True, 
            transform=transform_mask.Transforms(size=args.image_size, blur=True).test_transform,
        )
        class_num = 15
    elif args.dataset == "tiny-ImageNet":
        dataset1 = torchvision.datasets.ImageFolder(
            root='~/datasets/tiny-imagenet-200/train',
            transform=transform_mask.Transforms(size=args.image_size).test_transform,
        )
        class_num = 200
    else:
        raise NotImplementedError
    eval_loader = torch.utils.data.DataLoader(
        dataset1,
        batch_size=2000,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )


    # initialize model
    # res = resnet.get_resnet(args.resnet)
    # model = network.Network(res, args.feature_dim, class_num)
    clip = clip.CLIP()
    model = network.Network(clip, args.feature_dim, class_num)
    model = model.float()
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    model = model.to('cuda')
    # optimizer / loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)  #Adam 

    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=1e-6
    )
    
    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)  # 获取保存的最佳准确率
        
    loss_device = torch.device("cuda")
    scl_criterion = contrastive_loss.SoftContrastiveLoss(tau=args.instance_temperature).to(loss_device)
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(
        loss_device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)

    # train
    # pseudo_labels = np.full(len(dataset), -1)  # 初始化为 -1，占位符，大小等于拼接后的数据集
    for epoch in range(args.start_epoch, args.epochs):
        # 训练与动态更新伪标签
        start_time = time.time()
        lr = optimizer.param_groups[0]["lr"]
        # Step 1: 训练网络
        loss_epoch = train(train_loader, model, optimizer, criterion_instance, criterion_cluster, scl_criterion, epoch, args.epochs, args.lambda_s)
        scheduler.step()  # 更新学习率
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader):.4f}\t LR: {lr:.6f} \t Time: {epoch_time:.2f}s")

        # 动态评估聚类效果
        print("### Creating features from model ###")
        nmi, ari, f, acc = inference(eval_loader, model, device)

        # 保存最佳模型
        if acc > best_acc:  # 如果当前 ACC 大于历史最佳
            best_acc = acc
            save_model(args, model, optimizer, scheduler, epoch, best_acc)  # 保存模型
            print(f"New best model saved with ACC: {best_acc:.4f} at Epoch {epoch}")
    
        print(f"Current ACC: {acc:.4f}, Best ACC: {best_acc:.4f}")
        save_model(args, model, optimizer, scheduler, epoch, best_acc)
        
        # 保存模型
        if epoch % 100 == 0:
            save_model(args, model, optimizer, scheduler, epoch, best_acc)
    save_model(args, model, optimizer, scheduler, args.epochs, best_acc)


  
