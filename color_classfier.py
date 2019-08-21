import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import os
from sklearn.cluster import KMeans
from PIL import Image
import pickle
import pandas as pd
cwd = os.getcwd()
vips_path = cwd+'\\libvips'
os.environ['PATH'] = vips_path
import pyvips
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

def hex2hls(color):
    rgb_tuple = hex2rgb(color)
    h, l, s = colorsys.rgb_to_hls(rgb_tuple[0]/255, rgb_tuple[1]/255, rgb_tuple[2]/255)
    return tuple((h*360, l*100, s*100))

def hls2rgb(color):
    h, l, s = color[0]/360, color[1]/100, color[2]/100
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return tuple((int(r*255), int(g*255), int(b*255)))

def rgb2hls(color):
    r, g, b = color[0]/255, color[1]/255, color[2]/255
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return tuple((int(h*360), int(l*100), int(s*100)))

def rgb2hex(color):
    return '{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])

def hex_to_grayscale(hex_color):
    rgb_tuple = hex2rgb(hex_color)
    return 0.2126 * rgb_tuple[0] + 0.7152 * rgb_tuple[1] + 0.0722 * rgb_tuple[2]

def get_feature_vector(hex_color_list):
    rgb_list = []
    gray_scale_list = []
    for each in hex_color_list:
        h, s, v = hex2hsv(each)
        r, g, b = hex2rgb(each)
        rgb_list.append(r / 255)
        rgb_list.append(g / 255)
        rgb_list.append(b / 255)
        gs = hex_to_grayscale(each) / 255
        gray_scale_list.append(gs)
    gs_std = np.std(np.array(gray_scale_list))
    gray_scale_list.append(gs_std)
    feature_row = np.array(rgb_list + gray_scale_list)
    return feature_row.reshape(1, -1)

def keras_classifier():
    import tensorflow.keras
    import tensorflow.keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Activation
    from tensorflow.keras.optimizers import SGD
    from tensorflow.python.framework import ops
    ops.reset_default_graph()    # Generate dummy data
    import numpy as np

    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(64, activation='relu', input_dim=3))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    total_x = np.load('trainablesets/l2trainingarray.npy')
    training_array = shuffle(total_x)
    X = training_array[:, :-1]
    y = training_array[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)


    model.fit(x_train, y_train,
              epochs=200,
              batch_size=10)
    score = model.evaluate(x_test, y_test, batch_size=10)
    print(score)

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
    # blue = np.load('trainablesets/blue-4.npy')
    # summer_feature = summer_feature[:summer_feature.shape[0]-1, :]
    # green = np.load('trainablesets/green-4.npy')
    # winter_feature = winter_feature[:350, :]
    # red = np.load('trainablesets/red-4.npy')
    # total_x = np.vstack((blue, green, red))
    #
    total_x = np.load('trainablesets/l2trainingarray.npy')
    training_array = shuffle(total_x)
    X = training_array[:, :-1]
    y = training_array[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

    # svc = SVC(kernel='poly', verbose=True)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, verbose=True, learning_rate_init=0.01)
    mlp.fit(X_train, y_train)
    # svc.fit(X_train, y_train)
    mlp_predicted = mlp.predict(X_test)
    # svc_predicted = svc.predict(X_test)

    # mlp_acc = accuracy_score(y_test, mlp_predicted)
    svc_acc = accuracy_score(y_test, mlp_predicted)
    accuracy_string = str(int(svc_acc*100))+"L2"
    pickle.dump(mlp, open(accuracy_string, 'wb'))
    print(svc_acc)

    kmeans = KMeans(n_clusters=10)
    kmeans.fit(X_train)
    kmeans_predicted = kmeans.predict(X_test)
    kmeans_acc = accuracy_score(y_test, kmeans_predicted)
    print("Kmeans acc =", kmeans_acc)
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

def pca_analysis():
    import numpy as np
    quantized_img = Image.open('D:\Work\graph\Colorlevels.png').convert('RGB').getcolors()
    c = [x[1] for x in quantized_img]
    hls_colors = [rgb2hls(x) for x in c]
    colors_from_img = np.array(hls_colors)

    colours = np.array([
        [27, 17, 20], [121, 42, 65], [79, 15, 19], [79, 23, 17],
        [70, 15, 11], [84, 29, 27], [123, 44, 32], [138, 44, 34],
        [111, 28, 21], [86, 21, 16], [203, 65, 46], [193, 91, 63],
        [107, 24, 17], [109, 44, 30], [82, 24, 17], [127, 63, 33],
        [176, 97, 36], [155, 76, 32], [144, 91, 36], [185, 125, 50],
        [176, 140, 47], [174, 139, 87]
    ])

    ncolors = (colors_from_img - colors_from_img.mean(axis=0)) / colors_from_img.std(axis=0)  # always a smart idea to standardize pre-PCA
    S = np.corrcoef(ncolors.T)
    w, eignvectors = np.linalg.eig(S)

    reduced_data = np.dot(ncolors, eignvectors[:, np.argmax(w)])
    collection = np.argsort(reduced_data)
    collected_img = []
    for i in range(0, collection.shape[0]):
        empty = np.zeros((200, 600, 3), dtype='uint8')
        index = np.where(collection == i)[0][0]
        empty[:, :, :] = np.array(hls2rgb(tuple(colors_from_img[index])))
        collected_img.append(np_to_vips(empty))

    joined_image = pyvips.Image.arrayjoin(collected_img, across=1, shim=5)
    joined_image.write_to_file("sorted_colors_hls.jpg")


# pca_analysis()
# save_four_color_palettes()
# create_dataset()
train_classifier()
# predict_paletes()
# keras_classifier()
