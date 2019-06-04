import torch.nn as nn

# PyTorch에서 쓰는 여러 방법 중 하나로,
# 클래스로 모델을 만들어봅시다.
class CNNModel(nn.Module):
  
  # 여기서 레이어를 정의합니다.
  # 보통 학습가능한 파라미터가 들어가는 레이어를 여기서 먼저 정의해주고,
  # 아래 `forward`에서 가져다 씁니다.
  def __init__(self, nb_classes):
    super(CNNModel, self).__init__()
    
    # conv 레이어를 하나 둡니다. 7x7 필터를 2픽셀씩 건너뛰면서 3채널을 64채널로 늘려줍니다.
    self.conv1 = nn.Conv2d(in_channels = 3, 
                           out_channels = 16, 
                           kernel_size = 7, 
                           stride = 2)
    
    # pool 레이어를 하나 둡니다. conv를 거친 결과를 2x2커널로 2픽셀씩 건너뛰면서 최댓값만 취합니다.
    # 이로서 이미지의 가로세로가 절반으로 줄어듭니다.
    self.pool1 = nn.MaxPool2d(kernel_size=2,
                              stride=2)
    
    # pool 레이어의 출력물에 ReLU 액티베이션을 넣어줍니다. 음수는 모두 0이 됩니다.
    self.relu1 = nn.ReLU()
    
    # conv1->pool1->relu1을 한번 더 반복합니다.
    # 이때 Conv만 3x3, stride 1을 사용해 64 채널을 128 채널로 늘려줍니다. 나머지는 동일합니다.
    self.conv2 = nn.Conv2d(in_channels = 16, 
                           out_channels = 32, 
                           kernel_size = 3, 
                           stride = 1)
    self.pool2 = nn.MaxPool2d(kernel_size=2,
                              stride=2)
    self.relu2 = nn.ReLU()
    
    # 같은 층을 한번 더 반복합니다. 128 -> 256
    self.conv3 = nn.Conv2d(in_channels = 32, 
                           out_channels = 64, 
                           kernel_size = 3, 
                           stride = 1)
    self.pool3 = nn.MaxPool2d(kernel_size=2,
                              stride=2)
    self.relu3 = nn.ReLU()
    
    # 같은 층을 한번 더 반복합니다. 256 -> 512
    self.conv4 = nn.Conv2d(in_channels = 64, 
                           out_channels = 128, 
                           kernel_size = 3, 
                           stride = 1)
    self.pool4 = nn.MaxPool2d(kernel_size=2,
                              stride=2)
    self.relu4 = nn.ReLU()
    
    
    # conv 레이어의 출력물을 fully connected 레이어에 넣어 분류하기 위해서는
    # flatten 과정을 거쳐야 합니다.
    # conv 레이어의 출력물은 [32, 128, 5, 5]로,
    # 이를 [32, 128*5*5], 즉 [32, 3200]로 변환합니다.
    # 즉, 하나의 이미지는 3200개의 숫자로 변환되며, 이를
    # 아래 2개의 fully connected layer에 통과시킵니다.
    
    self.fc1 = nn.Linear(in_features = 3200, 
                         out_features = 512)
    self.relu_fc1 = nn.ReLU()
    
    # 마지막으로 4096개의 숫자를 우리가 가진 클래스만큼의 숫자로 변환해줍니다.
    self.fc2 = nn.Linear(512, nb_classes)
    
    
  # forward propagation  
  def forward(self, inputs):
    
    # 1st conv sequence
    x = self.conv1(inputs)
    x = self.relu1(x)
    x = self.pool1(x)
    
    # 2nd conv sequence
    x = self.conv2(x)
    x = self.relu2(x)
    x = self.pool2(x)
    
    # 3rd conv sequence
    x = self.conv3(x)
    x = self.relu3(x)
    x = self.pool3(x)
    
    # 4th conv sequence
    x = self.conv4(x)
    x = self.relu4(x)
    x = self.pool4(x)
    
    # flatten
    minibatch_size = inputs.size(0)
    flattened_x = x.view(minibatch_size, -1)
    
    
    
    # 1st fc sequence
    x = self.fc1(flattened_x)
    x = self.relu_fc1(x)
    
    # 2nd fc sequence
    x = self.fc2(x)
    
    return x
