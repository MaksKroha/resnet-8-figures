import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn.functional import dropout2d


class ResNet(nn.Module):
    def __init__(self, lr, dropout_p, first_blk_act=True, second_blk_act=True, third_blk_act=True):
        super(ResNet, self).__init__()
        self.dropout_p = dropout_p
        self.crossEntropy = nn.CrossEntropyLoss()
        self.init_conv = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

        # start of a residual block
        if first_blk_act:
            self.res_block1_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.res_block1_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.res_block1_conv1 = nn.Identity()
            self.res_block1_conv2 = nn.Identity()
        self.res_block1_norm1 = nn.BatchNorm2d(64) if first_blk_act else nn.Identity()
        self.res_block1_norm2 = nn.BatchNorm2d(64) if first_blk_act else nn.Identity()
        # end of a residual block
        # sum of skip connection and output of a block
        # dropout
        # activation

        # start of a block
        if second_blk_act:
            self.res_block2_conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
            self.res_block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.res_block2_conv1 = nn.Identity()
            self.res_block2_conv2 = nn.Identity()
        self.res_block2_norm1 = nn.BatchNorm2d(128) if second_blk_act else nn.Identity()
        self.res_block2_norm2 = nn.BatchNorm2d(128) if second_blk_act else nn.Identity()

        self.res_block2_shortcut = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
        # end of a block
        # sum of skip connection and output of a block
        # dropout
        # activation

        # start of a block
        if third_blk_act:
            self.res_block3_conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
            self.res_block3_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.res_block3_conv1 = nn.Identity()
            self.res_block3_conv2 = nn.Identity()
        self.res_block3_norm1 = nn.BatchNorm2d(256) if third_blk_act else nn.Identity()
        self.res_block3_norm2 = nn.BatchNorm2d(256) if third_blk_act else nn.Identity()

        self.res_block3_shortcut = nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)
        # end of a block
        # sum of skip connection and output of a block
        # dropout
        # global average pooling

        self.fcn = nn.Linear(256, 10)

        self.optimizer = torch.optim.Adam(self.parameters(), lr)

    def forward(self, tensor, first_blk_act=True, second_blk_act=True, third_blk_act=True):
        tensor = self.init_conv(tensor)

        # first residual block
        if first_blk_act:
            out = relu(self.res_block1_norm1(self.res_block1_conv1(tensor)))
            out = self.res_block1_norm2(self.res_block1_conv2(out))
            out = relu(tensor + out)

            out = dropout2d(out, self.dropout_p)
        else:
            out = tensor

        # second residual block
        skip = self.res_block2_shortcut(out)
        if second_blk_act:
            tensor = relu(self.res_block2_norm1(self.res_block2_conv1(out)))
            tensor = self.res_block2_norm2(self.res_block2_conv2(tensor))
            tensor = relu(skip + tensor)

            tensor = dropout2d(tensor, self.dropout_p)
        else:
            tensor = skip


        # third residual block
        skip = self.res_block3_shortcut(tensor)
        if third_blk_act:
            out = relu(self.res_block3_norm1(self.res_block3_conv1(tensor)))
            out = self.res_block3_norm2(self.res_block3_conv2(out))
            out = skip + out

            out = dropout2d(out, self.dropout_p)
        else:
            out = skip

        # global average pooling
        out = torch.mean(out, dim=(2, 3))  # D0xD1xD2xD3

        # fully connected layer
        # returning logits for a cross entropy loss
        return self.fcn(out)

    def backward(self, logits, labels):
        loss = self.crossEntropy(logits, labels)

        loss.backward()
        self.optimizer.step()

        self.optimizer.zero_grad()
