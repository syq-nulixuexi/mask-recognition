import os
import json
import glob
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from model_v2 import MobileNetV2


def main():
    im_height = 224
    im_width = 224
    num_classes = 2
    cap = cv2.VideoCapture(0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    feature = MobileNetV2(include_top=False)
    model = tf.keras.Sequential([feature,
                                 tf.keras.layers.GlobalAvgPool2D(),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(num_classes),
                                 tf.keras.layers.Softmax()])
    weights_path = './save_weights/resMobileNetV2.ckpt'
    assert len(glob.glob(weights_path + "*")), "cannot find {}".format(weights_path)
    model.load_weights(weights_path)

    while True:
        rval, cap_img = cap.read()
        cap_img = cv2.flip(cap_img, 1, 1)
        img = Image.fromarray(cap_img)
        # resize image to 224x224
        img = img.resize((im_width, im_height))
        # scaling pixel value to (-1,1)
        img = np.array(img).astype(np.float32)
        img = ((img / 255.) - 0.5) * 2.0
        # Add the image to a batch where it's the only member.
        img = (np.expand_dims(img, 0))
        result = np.squeeze(model.predict(img))
        predict_class = np.argmax(result)
        result_class_text = "Predict Result: " + class_indict[str(predict_class)]
        result_prob_text = "Accuracy: " + str(result[predict_class])
        fps = cap.get(cv2.CAP_PROP_FPS)  # 读取帧率
        fps_text = "fps: " + str(fps)
        cv2.putText(cap_img, fps_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 2)
        cv2.putText(cap_img, result_class_text, (120, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 2)
        cv2.putText(cap_img, result_prob_text, (360, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 2)
        cv2.imshow('im', cap_img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
