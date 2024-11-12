import os
import numpy as np
import cv2

project_root_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_npy = np.asarray(img, dtype=np.float32)
    img_npy = img_npy.transpose([2, 0, 1])
    img_npy /= 255.0
    return img_npy

def get_test_numpy():
    img_dir = os.path.join(project_root_dir, "images/images5")
    img_paths = []
    for img_name in ["dog.jpg", "eagle.jpg", "giraffe.jpg", "kite.jpg", "person.jpg"]:
        img_paths.append(os.path.join(img_dir, img_name))

    tensor = None
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img_npy = preprocess(img)
        if tensor is None:
            tensor = np.array([img_npy])
        else:
            tensor = np.vstack((tensor, [img_npy]))
    
    return tensor

if __name__ == "__main__":
    img_batch = get_test_numpy()
    test_data_dir =os.path.join(project_root_dir, "tests/test_data")
    if os.path.exists(test_data_dir) == False:
        os.mkdir(test_data_dir)
    test_data_path = os.path.join(test_data_dir, "simulator_api_test")
    np.save(test_data_path, img_batch)
