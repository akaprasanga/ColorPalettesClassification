import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import os
from PIL import Image
import pickle
import pandas as pd
import pyvips
import cv2
import colorsys

dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}

def np_to_vips(img):
    height, width, bands = img.shape
    linear = img.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                      dtype_to_format[str(img.dtype)])
    return vi


def rgb2hex(color):
    return '{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])


def hex2rgb(color):
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def hex_to_grayscale(hex_color):
    rgb_tuple = hex2rgb(hex_color)
    return 0.2126 * rgb_tuple[0] + 0.7152 * rgb_tuple[1] + 0.0722 * rgb_tuple[2]

def hex2hsv(color):
    color = hex2rgb(color)
    h, s, v = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
    return h, s, v

def get_feature_vector(hex_color_list):
    rgb_list = []
    gray_scale_list = []
    for each in hex_color_list:
        h, s, v = hex2hsv(each)
        r, g, b = hex2rgb(each)
        rgb_list.append(r / 255)
        rgb_list.append(g / 255)
        rgb_list.append(b / 255)
        # rgb_list.append(h)
        # rgb_list.append(s)
        # rgb_list.append(v)
        gs = hex_to_grayscale(each) / 255
        gray_scale_list.append(gs)
    gs_std = np.std(np.array(gray_scale_list))
    gray_scale_list.append(gs_std)
    feature_row = np.array(rgb_list + gray_scale_list)
    return feature_row.reshape(1, -1)

def create_dataset():
    # summer_data = np.loadtxt("color classes/summer.txt", dtype='str')
    winter_data = np.loadtxt("color classes/Greeen-4.txt", dtype='str')
    # print(summer_data)

    winter_feature = np.zeros((winter_data.shape[0], 37))
    for i in range(0, winter_data.shape[0]):
        color_list = list(winter_data[i, :])
        rgb_list = []
        gray_scale_list = []
        for each in color_list:
            # h, s, v = hex2hsv(each)
            r, g, b = hex2rgb(each)
            rgb_list.append(r/255)
            rgb_list.append(g/255)
            rgb_list.append(b/255)
            # rgb_list.append(h)
            # rgb_list.append(s)
            # rgb_list.append(v)
            gs = hex_to_grayscale(each)/255
            gray_scale_list.append(gs)
        gs_std = np.std(np.array(gray_scale_list))
        gray_scale_list.append(gs_std)
        gray_scale_list.append(1)
        feature_row = np.array(rgb_list+gray_scale_list)
        winter_feature[i, :] = feature_row
    np.save('trainablesets/green-4', winter_feature)

# create_dataset()
def train_classifier():
    blue = np.load('trainablesets/blue-4.npy')
    # summer_feature = summer_feature[:summer_feature.shape[0]-1, :]
    green = np.load('trainablesets/green-4.npy')
    # winter_feature = winter_feature[:350, :]
    red = np.load('trainablesets/red-4.npy')
    total_x = np.vstack((blue, green, red))
    #
    training_array = shuffle(total_x)
    X = training_array[:, :-1]
    y = training_array[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

    # svc = SVC()
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, verbose=True, learning_rate_init=0.01)
    mlp.fit(X_train, y_train)
    # svc.fit(X_train, y_train)
    mlp_predicted = mlp.predict(X_test)
    # svc_predicted = svc.predict(X_test)

    mlp_acc = accuracy_score(y_test, mlp_predicted)
    # svc_acc = accuracy_score(y_test, svc_predicted)
    accuracy_string = str(int(mlp_acc*100))+"B0G1R2"
    pickle.dump(mlp, open(accuracy_string, 'wb'))
    print(mlp_acc)
    # tn, fp, fn, tp = confusion_matrix(y_test, mlp_predicted).ravel()
    # print('tn = ', tn, 'Fp=', fp, 'Fn=', fn, 'tp=', tp)

def get_colors_from_images():
    for path, subdirs, files in os.walk("D:\Work\images\Designs (Soft colors)"):
        for name in files:
            print (os.path.join(path, name))
            if name != "Thumbs.db":
                c = Image.open(os.path.join(path, name)).convert('RGB').getcolors()
                rgb_colors = [x[1] for x in c]
                hex_colors = [rgb2hex(x) for x in rgb_colors]
                if len(hex_colors) >= 5:
                    write_str = ''
                    for i, each in enumerate(hex_colors):
                        if i < 5:
                            if i == 4:
                                write_str = write_str + each + '\n'
                            else:
                                write_str = write_str + each + '\t'

                    with open('Soft.txt', 'a') as the_file:
                        the_file.write(write_str)

def save_four_color_palettes():
    with open("ColorCategorization-Data/allcombinations.txt") as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    # content = [x.strip('\t') for x in content]
    for each in content:
        each = each.split('\n')[0]
        colors = each.split('\t')
        # print(colors)
        if len(colors)>4:
            write_str = ''
            for i, e in enumerate(colors):
                if i < 5:
                    if i == 4:
                        write_str = write_str + e + '\n'
                    else:
                        write_str = write_str + e + '\t'

            with open('avilable/totalcolors-filtered.txt', 'a') as the_file:
                the_file.write(write_str)


def predict_paletes():
    color_db = np.array(pd.read_excel("sorted_1000.xlsx"))
    color_db = color_db[:-1, :]
    model = pickle.load(open("91MLPClassifier-Bright0-Soft1", 'rb'))
    predicted_bright = []
    print(predicted_bright.__dir__())
    predicted_green = []
    # predicted_red = []
    for i in range(0, color_db.shape[0]):
        blank_img = np.zeros((32, 32, 3), dtype='uint8')
        hex_list = list(color_db[i, 10:15])
        rgb_list = [hex2rgb(x) for x in hex_list]
        features = get_feature_vector(hex_list)
        result = model.predict(features)
        # print(result)
        blank_img[:, :6, :] = np.array(rgb_list[0])
        blank_img[:, 6:12, :] = np.array(rgb_list[1])
        blank_img[:, 12:18, :] = np.array(rgb_list[2])
        blank_img[:, 18:24, :] = np.array(rgb_list[3])
        blank_img[:, 24:, :] = np.array(rgb_list[4])
        if result[0] == 0:
            # predicted_bright = np.hstack((predicted_bright, blank_img))
            # predicted_bright = np.append((predicted_bright, blank_img), axis=1)
            predicted_bright.append(blank_img)
        elif result[0] == 1:
            predicted_green.append(np_to_vips(blank_img))
        # elif result[0] == 2:
        #     predicted_red.append(np_to_vips(blank_img))
    # predicted_bright = predicted_bright[1:]
    predicted_bright = np.asarray(predicted_bright)
    np.save('color_gan_data', predicted_bright)
    # joined_blue = pyvips.Image.arrayjoin(predicted_blue, across=1, shim=8)
    # joined_blue.write_to_file('blue.png')
    #
    # joined_green = pyvips.Image.arrayjoin(predicted_green, across=1, shim=8)
    # joined_green.write_to_file('green.png')
    #
    # joined_red = pyvips.Image.arrayjoin(predicted_red, across=1, shim=8)
    # joined_red.write_to_file('red.png')
        # i = Image.fromarray(blank_img.astype('uint8'))
        # i.save('1.png')
    # for i, img in enumerate(predicted_brights):
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite("bright_palletes/"+str(i)+'.png', img)

save_four_color_palettes()
# create_dataset()
# train_classifier()
# predict_paletes()
