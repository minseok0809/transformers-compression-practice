import os
import glob
import math
import time
import torch
import visdom
import argparse
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.quantization
from torch.autograd import Variable
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from timm.models.layers import trunc_normal_
from torchvision.datasets.cifar import CIFAR10

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main():
    
    parer = argparse.ArgumentParser()
    parer.add_argument('--epoch', type=int, default=50)
    parer.add_argument('--batch_size', type=int, default=128)
    parer.add_argument('--lr', type=float, default=0.001)
    parer.add_argument('--step_size', type=int, default=100)
    parer.add_argument('--root', type=str, default='./vit_cifar10/data')
    parer.add_argument('--log_dir', type=str, default='./vit_cifar10/log_checkpoint')
    parer.add_argument('--name', type=str, default='vit_cifar10')
    parer.add_argument('--rank', type=int, default=0)
    parer.add_argument('--bitwidth', type=int, default=32)
    args = parer.parse_args()   
        
    # Embedding Layer
    class EmbeddingLayer(nn.Module):
        def __init__(self, in_chans, embed_dim, img_size, patch_size):
            super().__init__()
            self.num_tokens = (img_size // patch_size) ** 2
            self.embed_dim = embed_dim
            self.project = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.num_tokens += 1
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.embed_dim))
            
            nn.init.normal_(self.cls_token, std=1e-6)
            trunc_normal_(self.pos_embed, std=.02)

        def forward(self, x):
            B, C, H, W = x.shape
            embedding = self.project(x)
            z = embedding.view(B, self.embed_dim, -1).permute(0, 2, 1)  # BCHW -> BNC

            # concat cls token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            z = torch.cat([cls_tokens, z], dim=1)

            # add position embedding
            z = z + self.pos_embed
            return z

    # Quantization Neural Network
    class QuantNeuralNet(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, bitwidth):
            min_val = x.min()
            max_val = x.max()
            ctx.save_for_backward(x, bitwidth, min_val, max_val)
            x = 2 * (x - min_val) / (max_val - min_val) - 1
            factor = torch.Tensor([1]) << (bitwidth-1)
            return torch.round(x.to(device) * factor.to(device))

        @staticmethod
        def backward(ctx, grad_out):
            x, bitwidth, min_val, max_val = ctx.saved_tensors
            x = x.to(device)
            bitwidth = bitwidth.to(device)
            grad_x = grad_out * (x.norm(dim=-1) < 2**(bitwidth-1)).unsqueeze(-1).to(x.dtype)
            return grad_x, None
        
    # Linear Quantization
    class LinearQuant(nn.Module):
        def __init__(self, in_features, out_features, bias=True, bitwidth=args.bitwidth): 
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = Parameter(torch.Tensor(out_features, in_features))
            self.bitwidth = Variable(torch.Tensor([bitwidth]), requires_grad=False)
            if bias:
                self.bias = Parameter(torch.Tensor(out_features))
            else:
                self.register_parameter('bias', None)
                
            self.reset_parameters()

        def reset_parameters(self):
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

        def apply_linear(self, x, weight, bitwidth, bias=None):
            weight = QuantNeuralNet.apply(weight, bitwidth)
            # print("\n\n", weight, "\n\n")
            if bias is not None:
                bias = QuantNeuralNet.apply(torch.jit._unwrap_optional(bias), bitwidth)
            if x.dim() == 2 and bias is not None:
            # fused op is marginally faster
                ret = torch.addmm(bias, x, weight.t())
            else:
                output = x.matmul(weight.t())
                if bias is not None:
                    output += bias
                ret = output
            return QuantNeuralNet.apply(ret, bitwidth)
        
        def forward(self, x):
            # print("xHERE: ", self.weight.size())
            # print("\n\n", self.weight, "\n\n")
            return self.apply_linear(x, self.weight, self.bitwidth, self.bias)
        
    # Multihead Attention Layer
    class MSA(nn.Module):
        def __init__(self, dim=192, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
            super().__init__()
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            self.num_heads = num_heads
            head_dim = dim // num_heads
            self.scale = head_dim ** -0.5
            
            bit_width_list = [2, 4, 8, 16]
            if all(args.bitwidth != bit_width for bit_width in bit_width_list):
                self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
                self.attn_drop = nn.Dropout(attn_drop)
                self.proj = nn.Linear(dim, dim)
                self.proj_drop = nn.Dropout(proj_drop)
            
            if any(args.bitwidth == bit_width for bit_width in bit_width_list):
                # self.qkv = LinearQuant(dim, dim * 3, bias=qkv_bias)
                self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
                self.attn_drop = nn.Dropout(attn_drop)
                self.proj = nn.Linear(dim, dim)
                # self.proj = LinearQuant(dim, dim)
                self.proj_drop = nn.Dropout(proj_drop)
                
        def forward(self, x, bitwidth=args.bitwidth):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        
    # Multi-Layer Perceptron
    class MLP(nn.Module):
        def __init__(self, in_features, hidden_features, act_layer=nn.GELU, bias=True, drop=0.):
            super().__init__()
            out_features = in_features

            bit_width_list = [2, 4, 8, 16]
            if all(args.bitwidth != bit_width for bit_width in bit_width_list):
                self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
                self.act = act_layer()
                self.drop1 = nn.Dropout(drop)
                self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
                self.drop2 = nn.Dropout(drop)
            if any(args.bitwidth == bit_width for bit_width in bit_width_list):
                self.fc1 = LinearQuant(in_features, hidden_features, bias=bias)
                self.act = act_layer()
                self.drop1 = nn.Dropout(drop)
                self.fc2 = LinearQuant(hidden_features, out_features, bias=bias)
                self.drop2 = nn.Dropout(drop)                                
                
        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop1(x)
            x = self.fc2(x)
            x = self.drop2(x)
            return x
        
    # Encoder Block
    class Block(nn.Module):
        def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                    drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
            super().__init__()
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)  
            self.attn = MSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    # Vision Transformers  
    class ViT(nn.Module):
        def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=192, depth=12,
                    num_heads=12, mlp_ratio=2., qkv_bias=False, drop_rate=0., attn_drop_rate=0.):
            super().__init__()
            self.num_classes = num_classes
            self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
            norm_layer = nn.LayerNorm
            act_layer = nn.GELU

            self.patch_embed = EmbeddingLayer(in_chans, embed_dim, img_size, patch_size)
            self.blocks = nn.Sequential(*[
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, norm_layer=norm_layer, act_layer=act_layer)
                for i in range(depth)])
        
            self.norm = norm_layer(embed_dim)
            # Classifier head(s)
            # bit_width_list = [2, 4, 8, 16]
            # if all(args.bitwidth != bit_width for bit_width in bit_width_list):
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
            # if any(args.bitwidth == bit_width for bit_width in bit_width_list):
                # self.head = LinearQuant(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
                
        def forward(self, x):
            x = self.patch_embed(x)
            x = self.blocks(x)
            x = self.norm(x)
            x = self.head(x)[:, 0]
            return x
                
    vis = visdom.Visdom(port="8097") # python -m visdom.server
    
    transform_cifar = tfs.Compose([
        tfs.RandomCrop(32, padding=4),
        tfs.RandomHorizontalFlip(),
        tfs.ToTensor(),
        tfs.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010)),
    ])

    test_transform_cifar = tfs.Compose([tfs.ToTensor(),
                                        tfs.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                    std=(0.2023, 0.1994, 0.2010)),
                                        ])
    train_set = CIFAR10(root=args.root,
                        train=True,
                        download=True,
                        transform=transform_cifar)

    test_set = CIFAR10(root=args.root,
                    train=False,
                    download=True,
                    transform=test_transform_cifar)

    train_loader = DataLoader(dataset=train_set,
                            shuffle=True,
                            batch_size=args.batch_size)

    test_loader = DataLoader(dataset=test_set,
                            shuffle=False,
                            batch_size=args.batch_size)

    if args.name == "vit_cifar10":
        log_path = './vit_cifar10/log/vit_cifar10_log.xlsx'

    elif args.name == "quant16_vit_cifar10":
        log_path = './vit_cifar10/log/quant16_vit_cifar10_log.xlsx'
        
    elif args.name == "quant8_vit_cifar10":
        log_path = './vit_cifar10/log/quant8_vit_cifar10_log.xlsx'

    elif args.name == "quant4_vit_cifar10":
        log_path = './vit_cifar10/log/quant4_vit_cifar10_log.xlsx'
        
    elif args.name == "quant2_vit_cifar10":
        log_path = './vit_cifar10/log/quant2_vit_cifar10_log.xlsx'
    
    
    device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
    
    parallel = False
    if torch.cuda.device_count() >= 1:
        parallel = True
        model = torch.nn.DataParallel(ViT(), device_ids=[0]) # (revised)
            
    model = ViT().to(device)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.lr,
                                weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-5)
    os.makedirs(args.log_dir, exist_ok=True)
        
    training_log_df = pd.DataFrame({'Epoch':[0], 'Train Loss':[0], 'Train Accuracy':[0], 'Train Time':[0]})
                
    for epoch in range(args.epoch):
        model.train()
        train_pbar = tqdm(train_loader)
        train_pbar.set_description("Epoch " + str(epoch + 1))
        
        correct = 0
        val_avg_loss = 0
        total = 0
        
        for idx, (img, target) in enumerate(train_pbar):
            img = img.to(device)  # [N, 3, 32, 32]
            target = target.to(device)  # [N]
            # output, attn_mask = model(img, True)  # [N, 10]
            output = model(img)  # [N, 10]
            loss = criterion(output, target)
            pred, idx_ = output.max(-1)
            correct += torch.eq(target, idx_).sum().item()
            total += target.size(0)
            val_avg_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            
            if idx % args.step_size == 0:
                vis.line(X=torch.ones((1, 1)) * idx + epoch * len(train_loader),
                        Y=torch.Tensor([loss]).unsqueeze(0),
                        update='append',
                        win='training_loss',
                        opts=dict(x_label='step',
                                y_label='loss',
                                title='loss',
                                legend=['total_loss']))
            
                
            if idx == ( len(train_loader) - 1):
                
                accuracy = correct / total

                print('\tTrain Loss : {:.4f}\t' 'Train Accuracy : {:.4f}\t'.format(loss, accuracy))

                elapsed = train_pbar.format_dict['elapsed']
                elapsed_str = train_pbar.format_interval(elapsed)  
                
                if len(elapsed_str) == 5:
                    elapsed_str = "00:" + elapsed_str
                    elapsed_str = str(datetime.datetime.strptime(elapsed_str, '%H:%M:%S').time())  
                
                training_log_df.loc[len(training_log_df)] = [epoch + 1, round(loss.item(), 4), round(accuracy, 4), elapsed_str]
                training_log_df = training_log_df.astype({'Epoch':'int'})
        time.sleep(1)
        train_pbar.close() 
        save_path = os.path.join(args.log_dir, args.name, 'saves')
        os.makedirs(save_path, exist_ok=True)

        checkpoint = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()}

        torch.save(checkpoint, os.path.join(save_path, args.name + '.{}.pth.tar'.format(epoch)))
        
    print("\n")
    evaluation_log_df = pd.DataFrame({'Eval Time':[0], "Test Loss":[0], 'Valid Average Loss':[0], 'Test Accuracy':[0]})

    for epoch in range(args.epoch):
        file_path = 'vit_cifar10/log_checkpoint/' + args.name + '/saves/' + args.name + '.' + str(epoch) + '.pth.tar'
        checkpoint = torch.load(file_path, map_location="cuda:0")
        model = ViT().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  
        
        test_pbar = tqdm(test_loader)
        test_pbar.set_description("Epoch " + str(epoch + 1))
        
        correct = 0
        val_avg_loss = 0
        total = 0
        
        for idx, (img, target) in enumerate(test_pbar):
            
            img = img.to(device)  # [N, 3, 32, 32]
            target = target.to(device)  # [N]
            output = model(img)  # [N, 10]
            loss = criterion(output, target)
            output = torch.softmax(output, dim=1)
            
            pred, idx_ = output.max(-1)
            correct += torch.eq(target, idx_).sum().item()
            total += target.size(0)
            val_avg_loss += loss.item()
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if idx % args.step_size == 0:
                if vis is not None:
                    vis.line(X=torch.ones((1, 1)) * idx + epoch * len(test_loader),
                            Y=torch.Tensor([loss]).unsqueeze(0),
                            update='append',
                            win='test_loss',
                            opts=dict(x_label='step',
                                    y_label='loss',
                                    title='loss',
                                    legend=['total_loss']))
            
            if idx == ( len(test_loader) - 1):
                
                val_avg_loss = val_avg_loss / len(test_loader)
                accuracy = correct / total
                
                print('\tTest Loss : {:.4f}\t' 'Valid Average Loss : {:.4f}\t' 'Test Accuracy : {:.4f}\t'.format(loss, val_avg_loss, accuracy))
                
                elapsed = train_pbar.format_dict['elapsed']
                elapsed_str = train_pbar.format_interval(elapsed)  
        
                if len(elapsed_str) == 5:
                    elapsed_str = "00:" + elapsed_str
                    elapsed_str = str(datetime.datetime.strptime(elapsed_str, '%H:%M:%S').time())   
                    
                evaluation_log_df.loc[len(evaluation_log_df)] = [elapsed_str, round(loss.item(), 4), round(val_avg_loss, 4), round(accuracy, 4)]
        time.sleep(1)
        test_pbar.close()
        scheduler.step()
    
    log_df = pd.concat([training_log_df, evaluation_log_df], axis = 1)
    log_df = log_df.drop([0], axis = 0)
    log_df = log_df[['Epoch', 'Train Time', 'Eval Time', 'Train Loss', 'Train Accuracy',
                 'Test Loss', 'Valid Average Loss', 'Test Accuracy']]
    log_df.to_excel(log_path, index=False)
        
    
if __name__ == '__main__':
    main()       