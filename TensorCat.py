import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as vdatasets

# 데이터셋 생성
def create_dataset(dataset_dir): 
    # raw image를 가공하여 모델에 넣을 수 있는 인풋으로 변환한다.
    data_transforms = {
        'TRAIN': transforms.Compose([transforms.Resize((224, 224)), # 1.  사이즈를 224, 224로 통일한다.
                                     transforms.RandomHorizontalFlip(), # 좌우반전으로 데이터셋 2배 뻥튀기
                                     transforms.ToTensor(), # 2.  PIL이미지를 숫자 텐서로 변환한다.
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #3.  숫자에 mu(앞 리스트)를 빼고 sigma(뒷 리스트)로 나눠서 노멀라이즈한다.
                                     ]),
        'VAL': transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        }
    
    # 이미지 데이터셋의 형태로 트레이닝과 밸리데이션 데이터셋을 준비한다.
    image_datasets = {x: vdatasets.ImageFolder(os.path.join(dataset_dir, x), data_transforms[x])
                      for x in ['TRAIN', 'VAL']}
    
    # 레이블의 클래스 수도 구한다.  vgg pet dataset은 총 37종의 개/고양이를 가지고 있다.
    nb_classes = len(image_datasets['TRAIN'].classes)  
    return image_datasets, nb_classes

# 생성된 데이터셋을 불러오기
def create_dataloaders(image_datasets, training_batch_size, validation_batch_size, isShuffle):
    dataloaders = {'TRAIN': torch.utils.data.DataLoader(image_datasets['TRAIN'], batch_size=training_batch_size, shuffle=isShuffle),
                   'VAL': torch.utils.data.DataLoader(image_datasets['VAL'], batch_size=validation_batch_size, shuffle=isShuffle)}
    return dataloaders

# 손실함수와 최적화 함수를 지정
def prepare_loss_function_and_optimizer(lr, model):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return loss_function, optimizer

# 학습 시키기
def train(model, optimizer, loss_function, data_iterator, epoch):  
    print("training epoch {}".format(epoch))

    # model을 학습 모드로 바꿔줍니다.
    model.train()
    
    nb_corrects = 0
    nb_data = 0
    loss_list = []
  
    for ix, (inputs, targets) in enumerate(data_iterator):    
        # GPU에 데이터를 올려야 GPU에 올라간 모델이 처리할 수 있습니다.
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
    
        # 모델에 inputs를 넣어 출력값 outputs를 얻습니다.
        outputs = model(inputs)
    
        # 출력값과 실제값의 오차를 계산합니다.
        loss = loss_function(outputs, targets)
        loss_list.append(loss.item())
    
        # 실제 맞춘 갯수와 전체 갯수를 업데이트합니다.
        nb_corrects += (outputs.argmax(1) == targets).sum().item()
        nb_data += len(targets)
    
        # optimizer를 먼저 깔끔하게 초기화합니다.
        optimizer.zero_grad()
    
        # loss를 역전파합니다.
        loss.backward()
    
        # optimizer를 사용해 모델의 파라미터를 업데이트합니다.
        optimizer.step()
    
        if ix % 100 == 0:
          print(">> [{}] | loss: {:.4f}".format(ix, loss))
    
    epoch_accuracy = nb_corrects / nb_data
    epoch_avg_loss = torch.tensor(loss_list).mean().item()
  
    print("[training {:03d}] avg_loss: {:.4f} | accuracy: {:.4f}".format(epoch, epoch_avg_loss, epoch_accuracy))
    
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = CNNModel(nb_classes).to(DEVICE)    

    # train(model, optimizer, loss_function, dataloaders['TRAIN'], 0)

# __main__
DATASET_DIR = "data"
if __name__ == '__main__':
    image_datasets, nb_classes = create_dataset(DATASET_DIR)

    # nb_classes == 37 현재 분류된 데이터 수(개와 고양이 수)
    #print(nb_classes)

    # 32개씩 랜덤하게 학습할 이미지를 넘겨준다
    dataloaders = create_dataloaders(image_datasets, 32, 32, True)
    inputs, targets = next(iter(dataloaders['TRAIN']))

    # [32, 3, 224, 224] == [배치 사이즈, 채널(RGB), 높이, 너비]
    # 텐서플로우/케라스의 경우, [배치 사이즈, 높이, 너비, 채널] 순으로 다르다.
    #print(inputs.size()) 
    #print(inputs)

    # targets == 클래스 인덱스
    # 분류된 클래스와 인덱스 페어를 출력
    #print(targets)
    #print(image_datasets['TRAIN'].class_to_idx)

    # 인덱스로 클래스명을 얻어오기 (9번은 러시안 블루)
    #int2label = {v: k for k, v in image_datasets['TRAIN'].class_to_idx.items()}
    #print(int2label[9])

    # loss_function과 optimizer를 준비합니다.
    loss_function, optimizer = prepare_loss_function_and_optimizer(lr, model)