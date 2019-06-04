import os
import torch
import torch.nn as nn
import matplotlib.pylab as plt

# 생성한 더미 이미지를 시각화
def visualize_tensor(a_tensor):
      a_np = a_tensor.numpy().transpose(1, 2, 0)
      plt.imshow(a_np, cmap='jet')
      plt.show()

# __main__
if __name__ == '__main__':
    # 더미 이미지 생성
    _input = torch.rand((3, 5, 5))
    visualize_tensor(_input)

    input_ch = 3
    output_ch = 2
    filter_size = 3
    stride = 2
    padding = 1

    conv_filter = nn.Conv2d(input_ch, output_ch, filter_size, stride, padding)

    # batch_size, input_ch, image_height, image_width
    _input_batch = _input.unsqueeze(0)
    print(_input_batch.size())
    # 출력: torch.Size([1, 3, 5, 5])

    # 출력 채널이 2로 되었습니다.
    # batch_size, output_ch, filter_height, filter_width
    _output = conv_filter(_input_batch)
    print(_output.size())
    # 출력: torch.Size([1, 2, 3, 3])

    # .weight로 컨볼루션 레이어의 웨이트 파라미터에 접근할 수 있습니다.
    print(conv_filter.weight)
    # 3 x 3 x 3 x 2 행렬의 값이 출력됨

    # output_ch, input_ch, filter_height, filter_width
    print(conv_filter.weight.size())
    print("Conv layer 파라미터 수:", 2 * 3 * 3 * 3)
    # 출력: torch.Size([2, 3, 3, 3])
    # 출력: Conv layer 파라미터 수: 54

    print(_input_batch.size())
    print("Input의 차원 수:", 1 * 3 * 5 * 5)
    # 출력: Input의 차원 수: 75

    print(_output.size())
    print("Output의 차원 수:", 2 * 3 * 3)
    # 출력: Output의 차원 수: 18

    # fully connected layer(DNN)으로 같은 아웃풋 차원을 얻으려 했다면?
    fc = nn.Linear(75, 18)
    fc_output = fc(_input_batch.view(1, -1))
    print("FC 출력:", fc_output.size())
    print("FC 웨이트 차원:", fc.weight.size())
    print("FC 파라미터 수:", 18 * 75)
    # 출력: FC 출력: torch.Size([1, 18])
    # 출력: FC 웨이트 차원: torch.Size([18, 75])
    # 출력: FC 파라미터 수: 1350
