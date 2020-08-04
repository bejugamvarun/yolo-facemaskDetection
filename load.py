from convert import load_darknet_weights
from models import YoloV3
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('weights', './weights/yolov3.weights',
                    'path to darknet weights file')
flags.DEFINE_string('output', './checkpoints/yolov3.tf', 'path to save converted tf checkpoints model')


def main():
    yolo = YoloV3(classes=80)
    yolo.summary()
    logging.info('model created with 80 classes')
    load_darknet_weights(yolo, FLAGS.weights)
    logging.info('Darknet weights loaded')
    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo(img)
    logging.info(output)
    logging.info('test passed')
    yolo.save_weights(FLAGS.output)
    logging.info('weights saved in Checkpoint format')


if __name__ == '__main__':
    # file = open('./weights/yolov3-tiny.weights', 'rb')
    # # print(f'before np.fromfile :{len(file.read())}')
    # data = np.fromfile(file, dtype=np.int32)
    # print(f'len of data{len(data)}')
    # print(f'after np.fromfile :{len(file.read())}')
    # print(data)
    # file.close()
    # main()
    try:
        app.run(main)
    except SystemExit:
        pass