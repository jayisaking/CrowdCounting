## P2PNet SOTA
There are three parts of the model, like the original: backbone, decoder and Regression & Classification, which I denote as Back, Decode, and R&C (I used the same structure in both Classification and Regression.)
## Block Description
Here are some blocks design that I will mention later.
### Residual Block  
```
def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size = 3, 
                     stride = stride, padding = 1, bias = True)
    
# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.bn3 = nn.BatchNorm2d(in_channels)
    def forward(self, x):
        residual = self.bn3(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
```
### Original R&C
```
 out = self.conv1(x)
 out = self.act1(out)
 out = self.conv2(out)
 out = self.act2(out)
 out = self.output(out)
```
## Ablation Study

| Back | Decode | R&C | learning rate (down stream / backbone) | MSE | MAE | Input Resolution | Activation |
| ---- | -----  | --- | -------------------------------------  | --- | --- | ---------------- | ---------- |
| ConvNeXt Tiny | Add a 1x1 Convolution Before P5 and P4, Respectively| Add a 1x1 Convolution Between the original two Convolution Layers before output layer | 3e-4 / 1e-5 | 84.85 | 52.53 | 128 x 128 | RELU |
| ConvNeXt Tiny | Add a 1x1 Convolution Before P5 and P4, Respectively| Add a 1x1 Convolution Between the original two Convolution Layers before output layer | 3e-4 / 1e-5 | 87.49 | 55.74 | 128 x 128 | LeakyRELU |
| ConvNeXt Tiny | Add a 1x1 Convolution Before P5 and P4, Respectively| Add a 1x1 Convolution Between the original two Convolution Layers before output layer | 4e-4 / 1e-5 | 87.13 | 55.23 | 256 x 256 | RELU |
| ConvNeXt Tiny | Add a 1x1 Convolution Before P5 and P4, Respectively| Add a 1x1 Convolution Between the original two Convolution Layers before output layer | 3e-4 / 5e-5 | 89.75 | 56.53 | 256 x 256 | LeakyRELU |
| ConvNeXt Small | Add a 1x1 Convolution Before P5 and P4, Respectively| Add a 1x1 Convolution Between the original two Convolution Layers before output layer | 3e-4 / 1e-5 | 89.71 | 57.21 | 128 x 128 | RELU |
| ConvNeXt Base | Add a 1x1 Convolution Before P5 and P4, Respectively| Add a 1x1 Convolution Between the original two Convolution Layers before output layer| 3e-4 / 1e-5 | 93.01 | 58.33 | 128 x 128 | RELU |
| ConvNeXt Tiny | As Original | Extract Resnet101 Layers, replacing the original whole structure | 3e-4 / 1e-5 | 91.45 | 56.91 | 128 x 128 | RELU |
| ConvNeXt Tiny | As Original | Add 1 Residual Block Between the original two Convolution Layers before output layer| 3e-4 / 1e-5 | 90.23 | 56.01 | 128 x 128 | RELU |
| ConvNeXt Tiny | As Original | Cut down the original two Convolution Layers before output layer to one | 3e-4 / 1e-5 | 91.80 | 57.21 | 128 x 128 | RELU |
| ConvNeXt Tiny | As Original | Add 5 Residual Blocks Between the original two Convolution Layers| 3e-4 / 1e-5 | 92.16 | 57.87 | 128 x 128 | RELU |
| ConvNeXt Tiny | As Original | Increase the original two Convolution Layers before output layer to 5| 3e-4 / 1e-5 | 89.03 | 59.92 | 128 x 128 | RELU |
| ConvNeXt Tiny | As Original | Increase the original two Convolution Layers before output layer to 10| 3e-4 / 1e-5 | 88.74 | 60.64 | 128 x 128 | RELU |
| ConvNeXt Tiny | As Original | Using Inception Net Mechanism, replacing the original whole structure| 3e-4 / 1e-5 | 89.34 | 58.35 | 128 x 128 | RELU |
| ConvNeXt Tiny | As Original | As Original | 3e-4 / 1e-5 | 95.93 | 60.69 | 128 x 128 | RELU |



*If I didn't mention some hyperparameters, they remain the same as default in P2PNet Original Code.  
## P.S.
1.  To use my design, specify `--backbone convnext_tiny` in train.py/ run_test.py
2.  The log of SOTA result (52.53/ 84.85) is in the log folder. Since the training process was accidentally interrupted by technical issues, the log starts from epochs 983 (though it says from 0).
3.  Testing MAE/ MSE might differ from the SOTA result due to some randomness.
4.  The using of the code is basically the same as the original P2PNet Code.
5.  Specify `--data_root ./data` to use the data I prepared when training, because I made some adjustments to fit my folder structure and text format (this is trivial).
6.  The data folder contain the ShanghaiTech A. I only test my model on ShanghaiTech A
7.  Models are in ./ckpt.