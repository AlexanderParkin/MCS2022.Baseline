import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--data_path', default='../CompCars/data/',
                    help='path to dataset')
parser.add_argument('--annotation_path', default='annotation/',
                    help='path to save annotation')


def main():
    args = parser.parse_args()
    img_path = os.path.join(args.data_path, 'image/')
    label_path = os.path.join(args.data_path, 'label/')

    filelist = []
    for root, dirs, files in os.walk(img_path):
        for file in files:
            filelist.append(os.path.join(root, file))

    labellist = []
    for root, dirs, files in os.walk(label_path):
        for file in files:
            labellist.append(os.path.join(root, file))

    filelist = sorted(filelist)
    labellist = sorted(labellist)

    filelist = [x.replace(img_path, '') for x in filelist]

    full_data = pd.DataFrame(columns=['image_name'], data=np.array(filelist).T)
    full_data['class'] = full_data['image_name'].apply(lambda x: x.split('/')[1]).astype(int)

    le = preprocessing.LabelEncoder()
    le.fit(full_data['class'].values)
    full_data['class'] = le.transform(full_data['class'])

    x1s = []
    y1s = []
    x2s = []
    y2s = []

    print('Read annotations')
    for i in tqdm(range(len(labellist))):
        result = pd.read_csv(labellist[i], header=None).loc[2].values[0].split(' ')
        result = [int(x) for x in result]
        x1s.append(result[0])
        y1s.append(result[1])
        x2s.append(result[2])
        y2s.append(result[3])

    full_data['x_1'] = x1s
    full_data['y_1'] = y1s
    full_data['x_2'] = x2s
    full_data['y_2'] = y2s

    print(full_data)

    np.random.seed(42)
    full_data = full_data.sample(frac=1)
    train = full_data[:int(0.9*len(full_data))]
    val = full_data[int(0.9*len(full_data)):]

    print(f'Train size: {len(train)}, Val size: {len(val)}')

    if not os.path.isdir(args.annotation_path):
        os.mkdir(args.annotation_path)

    train[['image_name', 'x_1', 'y_1', 'x_2', 'y_2', 'class']].to_csv(args.annotation_path + 'train.txt', index=False)
    val[['image_name', 'x_1', 'y_1', 'x_2', 'y_2', 'class']].to_csv(args.annotation_path + 'val.txt', index=False)


if __name__ == '__main__':
    main()
