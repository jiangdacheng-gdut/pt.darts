""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
from config import SearchConfig
import utils
from models.search_cnn import SearchCNNController
from architect import Architect
from visualize import plot

# 创建一个 SearchConfig 对象，包含神经网络搜索相关的配置参数
config = SearchConfig()

# 设置使用的设备为 GPU（如果有 CUDA 可用）
device = torch.device("cuda")

# 初始化 TensorBoard SummaryWriter，用于可视化训练过程
# log_dir 参数设置了 TensorBoard 日志文件的存储路径
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))

# 将 config（网络搜索配置）以 Markdown 格式添加到 TensorBoard 文本中
writer.add_text('config', config.as_markdown(), 0)

# 获取一个日志记录器，用于记录训练过程中的信息
# 日志文件将保存在 config.path 目录下，并以 config.name 命名
logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))

# 打印配置参数，将参数信息记录到日志文件中
config.print_params(logger.info)

data_transforms = transforms.Compose([
    # transforms.Resize(224),
    # transforms.CenterCrop(224),
    # transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
        transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768])
])

def main():
    # 输出日志信息，表示训练开始
    logger.info("Logger is set - training start")

    # 设置默认的 GPU 设备 ID
    torch.cuda.set_device(config.gpus[0])

    # 设置随机种子，确保实验的可重复性
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # 为了加速卷积操作，启用 cuDNN 的自动调优功能
    torch.backends.cudnn.benchmark = True

    # 获取数据集及其元信息（输入尺寸、输入通道数、类别数等）
    # 通过 utils.get_data 函数加载数据集，并返回元信息和训练数据
    input_size, input_channels, n_classes, train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False)

    input_channels = 3
    n_classes = 10
    train_data = datasets.ImageFolder('/root/dacheng/pt.darts/data/cv_ids_25/train', transform=data_transforms)
    valid_data = datasets.ImageFolder('/root/dacheng/pt.darts/data/cv_ids_25/test', transform=data_transforms)

    # Step 2: Load dataset
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True, 
                                               num_workers=config.workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, shuffle=False, 
                                               num_workers=config.workers, pin_memory=True)

    # 定义交叉熵损失函数，并将其发送到设备（GPU）
    net_crit = nn.CrossEntropyLoss().to(device)

    # 创建 SearchCNNController 对象，用于执行网络搜索
    # 输入参数包括输入通道数、初始通道数、类别数、层数等
    model = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers,
                                net_crit, device_ids=config.gpus)
    # model = SearchCNNController(16, 36, 10, 20, net_crit, )

    # 将模型发送到设备（GPU）
    model = model.to(device)

    # 为模型权重设置优化器，使用随机梯度下降（SGD）
    # 设置学习率、动量和权重衰减参数
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)

    # 为架构参数（alphas）设置优化器，使用 Adam 优化器
    # 设置学习率、beta 参数和权重衰减参数
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)

    # # 将数据集分为训练集和验证集
    # n_train = len(train_data)
    # split = n_train // 2
    # indices = list(range(n_train))
    # # 使用随机子集采样器创建训练集和验证集的索引
    # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    # valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])

    # 创建训练集和验证集的数据加载器
    # train_loader = torch.utils.data.DataLoader(train_data,
    #                                            batch_size=config.batch_size,
    #                                            sampler=train_sampler,
    #                                            num_workers=config.workers,
    #                                            pin_memory=True)
    # valid_loader = torch.utils.data.DataLoader(train_data,
    #                                            batch_size=config.batch_size,
    #                                            sampler=valid_sampler,
    #                                            num_workers=config.workers,
    #                                            pin_memory=True)

    # 初始化学习率调度器，使用余弦退火策略调整学习率
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)

    # 创建 Architect 对象，用于更新模型的权重和架构参数（alphas）
    architect = Architect(model, config.w_momentum, config.w_weight_decay)

    # 训练循环
    best_top1 = 0.
    for epoch in range(config.epochs):
        # 更新学习率调度器
        lr_scheduler.step()
        # 获取当前学习率
        lr = lr_scheduler.get_lr()[0]

        # 打印架构参数（alphas）
        model.print_alphas(logger)

        # 训练模型
        train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch)

        # 验证模型
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(valid_loader, model, epoch, cur_step)

        # 记录日志
        # 获取当前模型的基因型
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))

        # 将基因型绘制为图像并保存
        plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
        caption = "Epoch {}".format(epoch+1)
        plot(genotype.normal, plot_path + "-normal", caption)
        plot(genotype.reduce, plot_path + "-reduce", caption)

        # 保存模型
        # 如果当前模型在验证集上的表现优于之前的模型，则将其标记为最佳模型
        if best_top1 < top1:
            best_top1 = top1
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)
        print("")

    # 记录最佳模型的准确率和基因型
    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))

def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch):
    # 初始化记录准确率和损失的工具
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    # 计算当前步数
    cur_step = epoch * len(train_loader)
    # 将学习率写入 TensorBoard
    writer.add_scalar('train/lr', lr, cur_step)

    # 将模型设置为训练模式
    model.train()

    # 遍历训练集和验证集的数据
    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        # 清空GPU缓存
        torch.cuda.empty_cache()
        
        # 将数据移动到 GPU 上
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        # 阶段 2：架构优化步骤 (更新 alpha)
        alpha_optim.zero_grad()
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)
        alpha_optim.step()

        # 阶段 1：子网络优化步骤 (更新 w)
        w_optim.zero_grad()
        logits = model(trn_X)
        loss = model.criterion(logits, trn_y)
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        # 计算准确率并更新记录器
        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        # 如果当前步数是设定的输出频率的整数倍，或者到达训练集末尾，输出训练信息
        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))

        # 将训练损失、top1 准确率和 top5 准确率写入 TensorBoard
        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        # 更新当前步数
        cur_step += 1

    # 训练完成后，输出当前 epoch 的最终 top1 准确率
    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

def validate(valid_loader, model, epoch, cur_step):
    # 初始化平均损失、top1 准确率和 top5 准确率的计数器
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    # 切换模型为评估模式
    model.eval()

    # 不计算梯度，减少内存使用
    with torch.no_grad():
        # 遍历验证集的每个批次
        for step, (X, y) in enumerate(valid_loader):
            # 将输入数据和标签转移到设备上
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            # 用模型进行预测
            logits = model(X)
            # 计算损失
            loss = model.criterion(logits, y)

            # 计算 top1 和 top5 准确率
            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            # 更新平均损失、top1 准确率和 top5 准确率
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            # 如果当前步数是设定的输出频率的整数倍，或者到达验证集末尾，输出验证信息
            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5))

    # 将验证损失、top1 准确率和 top5 准确率写入 TensorBoard
    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/top5', top5.avg, cur_step)

    # 输出当前 epoch 的最终 top1 准确率
    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    # 返回 top1 准确率
    return top1.avg


if __name__ == "__main__":
    main()
