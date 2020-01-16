import pandas as pd
import json
import random
import os

# frame = pd.DataFrame(columns=['class_name','id'])
# frame = frame.append(pd.DataFrame({'class_name':'pgps','id':0},index=[0]))
# frame = frame.append(pd.DataFrame({'class_name':'pgbx','id':1},index=[0]))
# frame = frame.append(pd.DataFrame({'class_name':'pghb','id':2},index=[0]))
# frame = frame.append(pd.DataFrame({'class_name':'pgdx','id':3},index=[0]))
# frame = frame.append(pd.DataFrame({'class_name':'pgdd','id':4},index=[0]))
# frame = frame.append(pd.DataFrame({'class_name':'pmzc','id':5},index=[0]))
# frame = frame.append(pd.DataFrame({'class_name':'pmyc','id':6},index=[0]))
# frame.to_csv('../data/class_a.csv',columns=['class_name','id'],index=False)
#
#
# frame = pd.DataFrame(columns=['class_name','id'])
# frame = frame.append(pd.DataFrame({'class_name':'tbwx','id':0},index=[0]))
# frame = frame.append(pd.DataFrame({'class_name':'tbqz','id':1},index=[0]))
# frame = frame.append(pd.DataFrame({'class_name':'tbqp','id':2},index=[0]))
# frame.to_csv('../data/class_b.csv',columns=['class_name','id'],index=False)
#
#




data_file_dir = '../data/annotations.json'

data_file_a = '../data/annotations_val_a.csv'
data_file_b = '../data/annotationa_val_b.csv'
frame_a = pd.DataFrame(columns=['image_name','x1','y1','x2','y2','class_name'])
frame_b = pd.DataFrame(columns=['image_name','x1','y1','x2','y2','class_name'])

# 读取含有所有行的txt文件
# 读取json中的各个行
with open(data_file_dir, 'r') as f:
    data = json.load(f)
    print(data.keys())
    # print(data['categories'])
    # print(data['images'])
    # print(data['annotations'])
    categories_dict = {'0':'background','1':'pgps','2':'pgbx','3':'pghb','4':'pgdx','5':'pgdd',
                       '6':'tbwx','7':'tbqz','8':'tbqp','9':'pmzc','10':'pmyc' }
    annotations = data['annotations']
    i = 0
    j = 0
    data_images = data['images']
    random.shuffle(data_images)
    for image_info in data_images[int(0.7*len(data_images)):]:
        image_name = os.path.join('../data/train/', image_info['file_name'])
        image_id = image_info['id']
        if image_info['height'] == 492:

            for anno in annotations:
                if anno['image_id'] == image_id and anno['category_id'] != 0:
                    frame_a = frame_a.append(pd.DataFrame({'image_name':image_name,'x1':int(anno['bbox'][0]),
                            'y1':int(anno['bbox'][1]),'x2':int(anno['bbox'][0]+anno['bbox'][2]),
                            'y2':int(anno['bbox'][1]+anno['bbox'][3]),'class_name':categories_dict[str(anno['category_id'])]},index=[i]))
                    i += 1
                    print('------------------',i)
        else:
            image_id = image_info['id']
            for anno in annotations:
                if anno['image_id'] == image_id and anno['category_id'] != 0:
                    frame_b = frame_b.append(pd.DataFrame({'image_name':image_name,'x1':int(anno['bbox'][0]),
                            'y1':int(anno['bbox'][1]),'x2':int(anno['bbox'][0]+anno['bbox'][2]),
                            'y2':int(anno['bbox'][1]+anno['bbox'][3]),'class_name':categories_dict[str(anno['category_id'])]},index=[i]))
                    j += 1
                    print('+++++++++++++++++++++',j)

frame_a.to_csv(data_file_a,columns=['image_name','x1','y1','x2','y2','class_name'],index=False,header=None)
frame_b.to_csv(data_file_b,columns=['image_name','x1','y1','x2','y2','class_name'],index=False,header=None)


