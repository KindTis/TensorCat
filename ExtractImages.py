# 0.  트레이닝 / 밸리데이션 데이터를 준비한다.
import os
import shutil
import glob
import re

import pandas as pd
from sklearn.model_selection import train_test_split

# helper function for dataset preparation
def extract_label(img_path):
  """
  이미지 패스에서 레이블을 추출합니다.
  arguments:
  img_path (string): ex)"images/Egyptian_Mau_63.jpg"
  
  return:
  label (string): ex)Egyption_Mau
  """
  label = re.search(r'images\\(.*)(_(.+).jpg)', img_path).group(1)
  return label


# helper function for dataset preparation
def copy_files(x, y, dataset_dir, category):
  print(">> copying files to {}".format(category))
  for image_path, label in zip(x, y):
    
    # 개별 레이블의 폴더가 없으면 만든다.
    new_img_dir = os.path.join(dataset_dir, category, label)
    if not os.path.exists(new_img_dir):
      os.makedirs(new_img_dir)
    
    # 해당 이미지를 복사하여 이동시킨다.
    new_image_path = os.path.join(new_img_dir, os.path.basename(image_path))
    shutil.copy(image_path, new_image_path)
  
  print(">> {} dataset created.".format(category))

  
RAW_DATA_DIR = 'images'
def prepare_images(raw_data_dir):
  print(">> preparing image data..")
  
  # 이미지 파일을 찾아서 레이블 값을 추출한다.
  image_paths = glob.glob("{}/*.jpg".format(raw_data_dir))
  image_df = pd.DataFrame(image_paths)
  image_df.columns = ['img_path']
  image_df['label'] = image_df.img_path.map(extract_label)
  
  # 데이터셋을 트레이닝 데이터와 밸리데이션 데이터로 나눈다.  (이미지 경로, 레이블)
  print(">> spliting training / validation dataset")
  train_x, val_x, train_y, val_y = train_test_split(image_df.img_path, image_df.label, # 분리할 이미지 경로와 레이블
      stratify=image_df.label, # 레이블을 기준으로 분할
      random_state=42, # 랜덤 시드
      test_size=0.1 # 밸리데이션 데이터 비율
  )
  
  # 데이터셋 폴더를 만든다.
  if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)
    
  # 트레이닝 데이터셋을 만든다.
  copy_files(train_x, train_y, DATASET_DIR, 'TRAIN')
  
  # 벨리데이션 데이터셋을 만든다.
  copy_files(val_x, val_y, DATASET_DIR, 'VAL')
  
  print(">> job done.")
  

DATASET_DIR = "data"
if not os.path.exists(DATASET_DIR):
  print("preparing image data")
  prepare_images(RAW_DATA_DIR)
else:
  print("image data are already prepared")