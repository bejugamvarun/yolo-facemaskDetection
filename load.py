from convert import load_darknet_weights
from models import YoloV3
import numpy as np


def main():
    yolo = YoloV3(classes=80)
    yolo.summary()
    print('model created')
    load_darknet_weights(yolo, './weights/yolov3.weights')
    print('weights loaded')
    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo(img)
    print(output)
    print('sanity check passed')
    yolo.save_weights('./checkpoints/yolov3.tf')
    print('weights saved')


if __name__ == '__main__':
    # file = open('./weights/yolov3-tiny.weights', 'rb')
    # # print(f'before np.fromfile :{len(file.read())}')
    # data = np.fromfile(file, dtype=np.int32)
    # print(f'len of data{len(data)}')
    # print(f'after np.fromfile :{len(file.read())}')
    # print(data)
    # file.close()
    main()
