import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.rmsprop import RMSprop


def save_model(
        model,
        optimizer,
        lr,
        batch_size,
        input_dim,
        output_dim,
        total_epochs,
        current_epoch,
        current_step,
        num_channels,
        num_classes,
        num_stacks,
        threshold,
        save_path='checkpoint.pth'):

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr': lr,
        'batch_size': batch_size,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'total_epochs': total_epochs,
        'current_epoch': current_epoch,
        'current_step': current_step,
        'num_channels': num_channels,
        'num_stacks': num_stacks,
        'num_classes': num_classes,
        'threshold': threshold
    }, save_path)


def load_model(load_path='checkpoint.pth'):
    checkpoint = torch.load(load_path)

    model = HourglassNetwork(
        num_channels=checkpoint['num_channels'],
        num_stacks=checkpoint['num_stacks'],
        num_classes=checkpoint['num_classes'],
        input_shape=(checkpoint['input_dim'], checkpoint['input_dim'], 3)
    )
    optimizer = RMSprop(model.parameters(), checkpoint['lr'])

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, checkpoint['batch_size'], checkpoint['input_dim'],


def conv2d(in_channels, out_channels, kernel_size, stride, padding_type='same', activation='', include_batchnorm=False):
    if padding_type == 'same':
        padding = (kernel_size-1) // 2
    elif padding_type == 'valid':
        padding = 0

    sequential = nn.Sequential()

    sequential.add_module(
        name='conv_layer',
        module=nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
    )

    # COMMENT BatchNorm before ReLU

    if include_batchnorm:
        sequential.add_module(
            name='batch_norm',
            module=nn.BatchNorm2d(out_channels)
        )
    if activation == 'relu':
        sequential.add_module(
            name='relu',
            module=nn.ReLU()
        )
    return sequential


class HourglassNetwork(nn.Module):

    def __init__(self, num_channels, num_stacks, num_classes, input_shape, depth=4):
        super(HourglassNetwork, self).__init__()

        self.num_stacks = num_stacks

        self.front_module = FrontModule(
            num_channels=num_channels,
            input_shape=input_shape
        )

        self.hourglass_modules = nn.ModuleList()
        self.out_residuals = nn.ModuleList()
        self.upper_1s = nn.ModuleList()
        self.upper_2s = nn.ModuleList()
        self.heatmaps = nn.ModuleList()
        self.lower_2s = nn.ModuleList()

        for s in range(num_stacks):
            self.hourglass_modules.append(
                HourglassModule(num_channels=num_channels, depth=depth)
            )

            self.out_residuals.append(ResidualModule(num_channels, num_channels))
            self.upper_1s.append(
                conv2d(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=1,
                    stride=1,
                    padding_type='same',
                    activation='relu',
                    include_batchnorm=True
                )
            )
            self.heatmaps.append(
                conv2d(
                    in_channels=num_channels,
                    out_channels=num_classes,
                    kernel_size=1,
                    stride=1
                )
            )

            if s < num_stacks-1:
                self.upper_2s.append(
                    conv2d(
                        in_channels=num_channels,
                        out_channels=num_channels,
                        kernel_size=1,
                        stride=1
                    )
                )
                self.lower_2s.append(
                    conv2d(
                        in_channels=num_classes,
                        out_channels=num_channels,
                        kernel_size=1,
                        stride=1
                    )
                )

    def forward(self, x):
        heatmaps = []
        x = self.front_module(x)

        for s in range(self.num_stacks):
            out = self.hourglass_modules[s](x)
            out = self.out_residuals[s](out)
            out = self.upper_1s[s](out)

            heatmap = self.heatmaps[s](out)

            if s < self.num_stacks-1:
                out = self.upper_2s[s](out)
                lower = self.lower_2s[s](heatmap)
                x = x + out + lower

            heatmaps.append(heatmap)

        return heatmaps


class HourglassModule(nn.Module):

    def __init__(self, num_channels, depth):
        super(HourglassModule, self).__init__()

        self.depth = depth

        self.left_residuals = nn.ModuleList()
        self.side_residuals = nn.ModuleList()
        self.middle_residuals = nn.ModuleList()
        self.right_residuals = nn.ModuleList()

        for d in range(depth):
            self.left_residuals.append(ResidualModule(num_channels, num_channels))
            self.side_residuals.append(ResidualModule(num_channels, num_channels))
            self.right_residuals.append(ResidualModule(num_channels, num_channels))

        for m in range(1):
            self.middle_residuals.append(ResidualModule(num_channels, num_channels))

    def forward(self, x):
        sides = []

        for d in range(self.depth):
            sides.append(self.side_residuals[d](x))

            x = F.max_pool2d(x, 2, stride=2)
            x = self.left_residuals[d](x)

        x = self.middle_residuals[0](x)

        for d in range(self.depth):
            x = self.right_residuals[d](x)
            x = F.interpolate(x, scale_factor=2)
            x = x + sides[self.depth-d-1]

        return x


class FrontModule(nn.Module):

    def __init__(self, num_channels, input_shape):
        super(FrontModule, self).__init__()

        assert num_channels == 256

        self.conv = conv2d(
            in_channels=input_shape[-1],
            out_channels=num_channels // 4,                                 # 64
            kernel_size=7,
            stride=2,
            padding_type='same',
            activation='relu',
            include_batchnorm=True
        )

        self.residual1 = ResidualModule(num_channels // 4, num_channels // 2)     # 128
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.residual2 = ResidualModule(num_channels // 2, num_channels // 2)     # 128
        self.residual3 = ResidualModule(num_channels // 2, num_channels)          # 256

    def forward(self, x):

        out = self.conv(x)
        out = self.residual1(out)
        out = self.maxPool(out)
        out = self.residual2(out)
        out = self.residual3(out)

        return out


class ResidualModule(nn.Module):

    """
        See Bottleneck block for if else statement
        Maybe channels wont fit
    """

    def __init__(self, input_channels, output_channels):
        super(ResidualModule, self).__init__()

        self.layer0 = None

        if input_channels != output_channels:
            self.layer0 = conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
                stride=1,
                padding_type='same',
                include_batchnorm=False
            )

        self.layer1 = conv2d(
                in_channels=output_channels,
                out_channels=output_channels // 2,
                kernel_size=1,
                stride=1,
                padding_type='same',
                activation='relu',
                include_batchnorm=True
        )

        self.layer2 = conv2d(
                in_channels=output_channels // 2,
                out_channels=output_channels // 2,
                kernel_size=3,
                stride=1,
                padding_type='same',
                activation='relu',
                include_batchnorm=True
        )

        self.layer3 = conv2d(
                in_channels=output_channels // 2,
                out_channels=output_channels,
                kernel_size=1,
                stride=1,
                padding_type='same',
                activation='relu',
                include_batchnorm=True
        )

    def forward(self, x):
        if self.layer0 is not None:
            x = self.layer0(x)

        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out += residual
        return out