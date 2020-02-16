# partial script (model No.104)

class TransformDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data
        self.data_num = len(data)
    def __len__(self):
        return self.data_num
    def __getitem__(self, idx):
        if self.transform:
            out_data = self.transform(self.data)[0][idx]
            out_data = out_data.reshape(img_h,img_w)[None,:,:]
        else:
            print("develope tensor with transform")
        return out_data

#----------------------- Convolutional Autoencoder -----------------------------------
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(6, 12, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(12,  6, kernel_size = 2, stride=1, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 1, kernel_size = 2, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z


def invmsefunction(input, target, size_average=None, reduce=None, reduction='mean'):
    if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(target.size(), input.size()),
                      stacklevel=2)
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    if target.requires_grad:
        ret = (input - target) ** 2
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    else:
        expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        ret = torch._C._nn.mse_loss(expanded_input, expanded_target, nn._reduction.get_enum(reduction))
        ret = 1 / (ret + 1e-7)
    return ret


class InvMSELoss(nn.Module):
    def __init__(self):
        super(InvMSELoss, self).__init__()

    def forward(self, inputs, targets):
        loss = invmsefunction(inputs, targets)
        return loss
