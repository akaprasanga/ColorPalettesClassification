import cv2
import numpy as np
import pandas as pd
import pyvips
import colorsys
import pyvips
from PIL import Image


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
    return h*255, s*255, v*255

def hex2hls(color):
    color = hex2rgb(color)
    h, l, s = colorsys.rgb_to_hls(color[0]/255, color[1]/255, color[2]/255)
    return h*255, l*255, s*255


def cv2hsv_conversion():
    green = np.uint8([[[158, 8, 67]]]) #here insert the bgr values which you want to convert to hsv
    hsvGreen = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
    print(hsvGreen)

    lowerLimit = hsvGreen[0][0][0] - 10, 100, 100
    upperLimit = hsvGreen[0][0][0] + 10, 255, 255


    print(upperLimit)
    print(lowerLimit)
    blank = np.zeros((200, 400, 3), dtype='uint8')
    blank[:, :200, :] = np.array(upperLimit)
    blank[:, 200:, :] = np.array(lowerLimit)

    colored = cv2.cvtColor(blank, cv2.COLOR_HSV2BGR)
    cv2.imwrite('colored.png', colored)

def create_hsv_db(db_name="total_filtered_colors.xlsx"):
    color_db = np.array(pd.read_excel(db_name))
    db_column = ['extra_index', 'id', 'title', 'userName', 'numViews', 'numVotes', 'numComments', 'numHearts',
                      'rank', 'dateCreated', 'color 1', 'color 2', 'color 3', 'color 4', 'color 5', 'wid 1', 'wid 2',
                      'wid 3', 'wid 4', 'wid 5', 'url', 'imageUrl', 'badgeUrl', 'apiUrl', 'r1', 'g1', 'b1', 'r2', 'g2',
                      'b2', 'r3', 'g3', 'b3', 'r4', 'g4', 'b4', 'r5', 'g5', 'b5', 'gs1', 'gs2', 'gs3', 'gs4', 'gs5',
                      'max gs', 'min gs', 'avggs', 'stddev gs', 'wgt r', 'wgt g', 'wgt b', 'wgt gs', 'd12', 'd13',
                      'd14', 'd15', 'd23', 'd24', 'd25', 'd34', 'd35', 'd45', 'max d', 'min d', 'avg d', 'stddev d',
                      'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4', 'x5', 'y5', 'z5', 'l1',
                      'a1', 'b1.1', 'l2', 'a2', 'b2.1', 'l3', 'a3', 'b3.1', 'l4', 'a4', 'b4.1', 'l5', 'a5', 'b5.1',
                      'dl12', 'dl13', 'dl14', 'dl15', 'dl23', 'dl24', 'dl25', 'dl34', 'dl35', 'dl45', 'max dl',
                      'min dl', 'avg dl', 'stddev dl', 'da12', 'da13', 'da14', 'da15', 'da23', 'da24', 'da25', 'da34',
                      'da35', 'da45', 'max da', 'min da', 'avg da', 'stddev da', 'db12', 'db13', 'db14', 'db15', 'db23',
                      'db24', 'db25', 'db34', 'db35', 'db45', 'max db', 'min db', 'avg db', 'stddev db', 'de12', 'de13',
                      'de14', 'de15', 'de23', 'de24', 'de25', 'de34', 'de35', 'de45', 'max de', 'min de', 'avg de',
                      'stddev de', 'h1', 'l1.1', 's1', 'h2', 'l2.1', 's2', 'h3', 'l3.1', 's3', 'h4', 'l4.1', 's4', 'h5',
                      'l5.1', 's5', 'dh12', 'dh13', 'dh14', 'dh15', 'dh23', 'dh24', 'dh25', 'dh34', 'dh35', 'dh45',
                      'dha12', 'dha13', 'dha14', 'dha15', 'dha23', 'dha24', 'dha25', 'dha34', 'dha35', 'dha45',
                      'max dh', 'min dh', 'avg dh', 'stddev dh']
    color_db = color_db[:, 1:]

    hsv_db = np.zeros((color_db.shape[0], 15), dtype='uint8')
    for i in range(0, color_db.shape[0]):
        colorhex_list = list(color_db[i, 0:5])
        hsv_list = [hex2hls(x) for x in colorhex_list]
        plain_list = []
        for each in hsv_list:
            plain_list.append(each[0])
            plain_list.append(each[1])
            plain_list.append(each[2])
        hsv_db[i, :] = np.array(plain_list)
    cols = ['h1', 'l1', 's1', 'h2', 'l2', 's2', 'h3', 'l3', 's3', 'h4', 'l4', 's4', 'h5', 'l5', 's5']
    df = pd.DataFrame(data=hsv_db, columns=cols)
    df.to_excel('total_colors_hls.xlsx')

def search_palettes_rgb(result=10):
    original_db = np.array(pd.read_excel('rgb_color_from_sorted.xlsx'))
    color_db = original_db[:, 1:]
    color_to_search = (224, 2, 33)
    # hsv_color_to_search = colorsys.rgb_to_hsv(color_to_search[0]/255, color_to_search[1]/255, color_to_search[2]/255)
    hsv_color_to_search = color_to_search
    # print(hsv_color_to_search)
    # hsv_color_to_search = tuple([x*255 for x in hsv_color_to_search])
    # print(hsv_color_to_search)
    search_color_array = np.zeros((color_db.shape[0], 3), dtype='uint8')
    search_color_array[:, 0] = hsv_color_to_search[0]
    search_color_array[:, 1] = hsv_color_to_search[1]
    search_color_array[:, 2] = hsv_color_to_search[2]
    search_color_array = np.hstack([search_color_array]*5)

    distance_array = abs(np.subtract(color_db, search_color_array))
    # sum_of_distance = np.linalg.norm(color_db-search_color_array, axis=1)
    sum_of_distance = distance_array[:, 0]+distance_array[:, 3]+distance_array[:, 6]+distance_array[:, 9]+distance_array[:, 12]
    sum_of_distance = sum_of_distance.ravel()
    # sum_of_distance = sum_of_distance.reshape(sum_of_distance.shape[0], 1)
    sum_of_distance = np.vstack((sum_of_distance, original_db[:, 0])).T
    sum_of_distance = sum_of_distance[sum_of_distance[:, 0].argsort()]
    selected_palletes = []
    for j in range(0, result):
        blank_img = np.zeros((200, 1000, 3), dtype='uint8')
        location = sum_of_distance[j, 1]
        hsv_array = original_db[location, 1:]
        shaped = hsv_array.reshape(5, 3)
        for k in range(0, 5):
            # rgb_tuple = colorsys.hsv_to_rgb(shaped[k, 0]/255, shaped[k, 1]/255, shaped[k, 2]/255)
            # blank_img[:, k*200:(k+1)*200, :] = np.array(rgb_tuple)*255
            blank_img[:, k*200:(k+1)*200, :] = np.array(shaped[k])
        selected_palletes.append(np_to_vips(blank_img))
    joined_green = pyvips.Image.arrayjoin(selected_palletes, across=1, shim=8)
    joined_green.write_to_file('rgb.png')

def txt_xlsx():
    txt_format = np.loadtxt("avilable/totalcolors-filtered.txt", dtype='str')
    cols = ['hex1', 'hex2', 'hex3', 'hex4', 'hex5']
    df = pd.DataFrame(data=txt_format, columns=cols)
    df.to_excel('total_filtered_colors.xlsx')

def search_palettes_hsv(result=10):
    original_db = np.array(pd.read_excel('hsv_color_from_sorted.xlsx'))
    color_db = original_db[:, 1:]
    color_to_search = (0, 255, 0)
    hsv_color_to_search = colorsys.rgb_to_hsv(color_to_search[0]/255, color_to_search[1]/255, color_to_search[2]/255)
    # hsv_color_to_search = color_to_search
    # print(hsv_color_to_search)
    hsv_color_to_search = tuple([x*255 for x in hsv_color_to_search])
    # print(hsv_color_to_search)
    search_color_array = np.zeros((color_db.shape[0], 3), dtype='uint8')
    search_color_array[:, 0] = hsv_color_to_search[0]
    search_color_array[:, 1] = hsv_color_to_search[1]
    search_color_array[:, 2] = hsv_color_to_search[2]
    search_color_array = np.hstack([search_color_array]*5)

    distance_array = abs(np.subtract(color_db, search_color_array))
    # sum_of_distance = np.linalg.norm(color_db-search_color_array, axis=1)
    sum_of_distance = distance_array[:, 0]+distance_array[:, 3]+distance_array[:, 6]+distance_array[:, 9]+distance_array[:, 12]
    sum_of_distance = sum_of_distance.ravel()
    # sum_of_distance = sum_of_distance.reshape(sum_of_distance.shape[0], 1)
    sum_of_distance = np.vstack((sum_of_distance, original_db[:, 0])).T
    sum_of_distance = sum_of_distance[sum_of_distance[:, 0].argsort()]
    selected_palletes = []
    for j in range(0, result):
        blank_img = np.zeros((200, 1000, 3), dtype='uint8')
        location = sum_of_distance[j, 1]
        hsv_array = original_db[location, 1:]
        shaped = hsv_array.reshape(5, 3)
        for k in range(0, 5):
            rgb_tuple = colorsys.hsv_to_rgb(shaped[k, 0]/255, shaped[k, 1]/255, shaped[k, 2]/255)
            blank_img[:, k*200:(k+1)*200, :] = np.array(rgb_tuple)*255
            # blank_img[:, k*200:(k+1)*200, :] = np.array(shaped[k])
        selected_palletes.append(np_to_vips(blank_img))
    joined_green = pyvips.Image.arrayjoin(selected_palletes, across=1, shim=8)
    joined_green.write_to_file('hsv.png')


def search_palettes_hls(color, result=10):
    original_db = np.array(pd.read_excel('total_colors_hls.xlsx'))
    color_db = original_db[:, 1:]
    color_to_search = color
    hsv_color_to_search = colorsys.rgb_to_hls(color_to_search[0]/255, color_to_search[1]/255, color_to_search[2]/255)
    # hsv_color_to_search = color_to_search
    # print(hsv_color_to_search)
    hsv_color_to_search = tuple([x*255 for x in hsv_color_to_search])
    # print(hsv_color_to_search)
    search_color_array = np.zeros((color_db.shape[0], 3), dtype='uint8')
    search_color_array[:, 0] = hsv_color_to_search[0]
    search_color_array[:, 1] = hsv_color_to_search[1]
    search_color_array[:, 2] = hsv_color_to_search[2]
    search_color_array = np.hstack([search_color_array]*5)

    distance_array = abs(np.subtract(color_db, search_color_array))
    # sum_of_distance = np.linalg.norm(color_db-search_color_array, axis=1)
    sum_of_distance = distance_array[:, 0]+distance_array[:, 1]+distance_array[:, 3]+distance_array[:, 4]+distance_array[:, 6]+distance_array[:, 7]+distance_array[:, 9]+distance_array[:, 10]+distance_array[:, 12]+distance_array[:, 13]
    sum_of_distance = sum_of_distance.ravel()
    # sum_of_distance = sum_of_distance.reshape(sum_of_distance.shape[0], 1)
    sum_of_distance = np.vstack((sum_of_distance, original_db[:, 0])).T
    sum_of_distance = sum_of_distance[sum_of_distance[:, 0].argsort()]
    selected_palletes = []
    for j in range(0, result):
        blank_img = np.zeros((200, 1000, 3), dtype='uint8')
        location = sum_of_distance[j, 1]
        hsv_array = original_db[location, 1:]
        shaped = hsv_array.reshape(5, 3)
        for k in range(0, 5):
            rgb_tuple = colorsys.hls_to_rgb(shaped[k, 0]/255, shaped[k, 1]/255, shaped[k, 2]/255)
            blank_img[:, k*200:(k+1)*200, :] = np.array(rgb_tuple)*255
            # blank_img[:, k*200:(k+1)*200, :] = np.array(shaped[k])
        selected_palletes.append(np_to_vips(blank_img))
    joined_green = pyvips.Image.arrayjoin(selected_palletes, across=1, shim=8)
    joined_green.write_to_file('hls.png')
    pl = Image.open('hls.png').convert('RGB')
    pl.show()


# def visualize_hsv(color = (117, 90, 126)):
#     blank = np.zeros((200, 1000, 3), dtype='uint8')
#
#     hls = colorsys.rgb_to_hls(color[0]/255, color[1]/255, color[2]/255)
#
#     for i in range(0, 5):


# txt_xlsx()
# create_hsv_db()
# search_palettes_rgb(result=100)
# search_palettes_hsv(result=100)
search_palettes_hls(color=(117, 90, 126), result=100)