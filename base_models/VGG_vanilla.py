from torch import nn


class VGGVanilla(nn.Module):
     """
     VGG model implementation base on the paper by:

     @article{simonyan2014very,
     title={Very deep convolutional networks for large-scale image recognition},
     author={Simonyan, Karen and Zisserman, Andrew},
     journal={arXiv preprint arXiv:1409.1556},
     year={2014}
     }
     """
     # architecture configurations
     arc_config = {
     "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
     "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
     "C": [64, 64, "M", 128, 128, "M", 256, 256, 1, 256, "M", 512, 512, 1, 512, "M", 512, 512, 1, 512, "M"],
     "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
     "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
     }

     def __init__(self, vgg_type: str = "A", output_classes: int = 1000, p_dropout: float = 0.5):
          """
               
          """
          super().__init__()
          self.conv_layers = self._make_layers(vgg_type) # stack of CNN layers

          # average pool
          self.avg_pool = nn.AdaptiveAvgPool2d((7,7))

          # Fully-Connected (FC) layers
          self.fc_layer = nn.Sequential(
               nn.Linear(512 * 7 * 7, 4096),
               nn.ReLU(),
               nn.Dropout(p=p_dropout),
               nn.Linear(4096, 4096),
               nn.ReLU(),
               nn.Dropout(p=p_dropout),
               nn.Linear(4096, output_classes)
          )
    
     def _make_layers(self, vgg_type: str):
          """
          VGG CNN stack based on provided configuration settings

          Args:
               vgg_type: VGG ConvNet configuration

          Return:
               CNN_stack: developed architecture
          """

          CNN_layers = []
          input_channel = 3 # 3 ==> initial number of channel
          config = self.arc_config[vgg_type]

          is_conv_1 = False # flag for convolution with size 1

          for l in config:
               # add MaxPooling
               if l == "M":
                    CNN_layers.append(nn.MaxPool2d((2,2), stride=(2,2)))
               
               # set flag tochange convolution size
               elif l==1:
                    is_conv_1 = True
               
               else:
                    if not is_conv_1:
                         CNN_layers += [nn.Conv2d(input_channel, l, kernel_size=3, stride=1, padding=1), nn.ReLU()]
                    else:
                         CNN_layers += [nn.Conv2d(input_channel, l, kernel_size=1, stride=1, padding=0), nn.ReLU()]
                         is_conv_1 = False # turn off flag

                    input_channel = l

          CNN_stack = nn.Sequential(*CNN_layers)
          return CNN_stack

     def forward(self, x):
          """
          forward pass
          """
          x = self.conv_layers(x)
          x = self.avg_pool(x)
          x = x.view(x.shape[0], -1)
          x = self.fc_layer(x)
          return x

