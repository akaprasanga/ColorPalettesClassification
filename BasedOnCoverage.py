import numpy as np
import pandas as pd
import pyvips
from PIL import Image
import ntpath
from custom_library import python_utils
import colorsys
from skimage import color


class RecolorBasedOnCoverage:

    def __init__(self, img_name, db_name):
        self.img_name = img_name
        self.db_name = db_name
        self.format_to_dtype = {
            'uchar': np.uint8,
            'char': np.int8,
            'ushort': np.uint16,
            'short': np.int16,
            'uint': np.uint32,
            'int': np.int32,
            'float': np.float32,
            'double': np.float64,
            'complex': np.complex64,
            'dpcomplex': np.complex128,
        }
        self.header_names = ['extra_index', 'id', 'title', 'userName', 'numViews', 'numVotes', 'numComments', 'numHearts', 'rank', 'dateCreated', 'color 1', 'color 2', 'color 3', 'color 4', 'color 5', 'wid 1', 'wid 2', 'wid 3', 'wid 4', 'wid 5', 'url', 'imageUrl', 'badgeUrl', 'apiUrl', 'r1', 'g1', 'b1', 'r2', 'g2', 'b2', 'r3', 'g3', 'b3', 'r4', 'g4', 'b4', 'r5', 'g5', 'b5', 'gs1', 'gs2', 'gs3', 'gs4', 'gs5', 'max gs', 'min gs', 'avggs', 'stddev gs', 'wgt r', 'wgt g', 'wgt b', 'wgt gs', 'd12', 'd13', 'd14', 'd15', 'd23', 'd24', 'd25', 'd34', 'd35', 'd45', 'max d', 'min d', 'avg d', 'stddev d', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4', 'x5', 'y5', 'z5', 'l1', 'a1', 'b1.1', 'l2', 'a2', 'b2.1', 'l3', 'a3', 'b3.1', 'l4', 'a4', 'b4.1', 'l5', 'a5', 'b5.1', 'dl12', 'dl13', 'dl14', 'dl15', 'dl23', 'dl24', 'dl25', 'dl34', 'dl35', 'dl45', 'max dl', 'min dl', 'avg dl', 'stddev dl', 'da12', 'da13', 'da14', 'da15', 'da23', 'da24', 'da25', 'da34', 'da35', 'da45', 'max da', 'min da', 'avg da', 'stddev da', 'db12', 'db13', 'db14', 'db15', 'db23', 'db24', 'db25', 'db34', 'db35', 'db45', 'max db', 'min db', 'avg db', 'stddev db', 'de12', 'de13', 'de14', 'de15', 'de23', 'de24', 'de25', 'de34', 'de35', 'de45', 'max de', 'min de', 'avg de', 'stddev de', 'h1', 'l1.1', 's1', 'h2', 'l2.1', 's2', 'h3', 'l3.1', 's3', 'h4', 'l4.1', 's4', 'h5', 'l5.1', 's5', 'dh12', 'dh13', 'dh14', 'dh15', 'dh23', 'dh24', 'dh25', 'dh34', 'dh35', 'dh45', 'dha12', 'dha13', 'dha14', 'dha15', 'dha23', 'dha24', 'dha25', 'dha34', 'dha35', 'dha45', 'max dh', 'min dh', 'avg dh', 'stddev dh']
        self.width_sorted_color_db = np.array(pd.read_excel(self.db_name))
        self.width_sorted_color_db = self.width_sorted_color_db[:self.width_sorted_color_db.shape[0]-1, :]
        # self.choose_candidate_color_from_db()
        # self.create_width_sorted_table()
        # self.get_width_of_color_from_img(img_name)
        # self.choose_color_from_grayscale()
        self.choose_palette_from_img()

    def rgb2hex(self, color):
        return '{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])

    def hex2rgb(self, color):
        return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    def hex_to_grayscale(self, hex_color):
        rgb_tuple = self.hex2rgb(hex_color)
        return 0.2126*rgb_tuple[0]+0.7152*rgb_tuple[1]+0.0722*rgb_tuple[2]

    def create_width_sorted_table(self):
        color_database = pd.read_excel("1000 Color Values.xlsx")
        data = np.array(color_database)

        for i in range(0, data.shape[0]):
            array_to_sort = data[i, 9:19]
            array_to_sort = array_to_sort.reshape(2, 5)
            array_to_sort = array_to_sort.T
            array_to_sort = array_to_sort[array_to_sort[:, 1].argsort()]
            array_to_sort = array_to_sort.T
            array_to_sort = array_to_sort.ravel()
            data[i, 9:19] = array_to_sort

        sorted_df = pd.DataFrame(data=data[:, :],
                                 columns=self.header_names,
                                 )
        sorted_df.to_excel("sorted_1000.xlsx")

    def replace_color(self, filename, source_list, destination_list):
        img = pyvips.Image.new_from_file(filename)
        if len(img.bandsplit()) == 4:
            img = img.flatten()
        replaced_img = img.copy()
        source_list = [self.hex2rgb(x) for x in source_list]
        destination_list = [self.hex2rgb(x) for x in destination_list]

        for i, each_color in enumerate(source_list):
            replaced_img = (img == each_color).ifthenelse(destination_list[i], replaced_img)

        mem_img = replaced_img.write_to_memory()
        np_3d = np.ndarray(buffer=mem_img,
                           dtype=self.format_to_dtype[replaced_img.format],
                           shape=[replaced_img.height, replaced_img.width, replaced_img.bands])
        return np_3d

    def replace_color_rgb(self, filename, source_list, destination_list):
        img = pyvips.Image.new_from_file(filename)
        if len(img.bandsplit()) == 4:
            img = img.flatten()
        replaced_img = img.copy()
        # source_list = [self.hex2rgb(x) for x in source_list]
        # destination_list = [self.hex2rgb(x) for x in destination_list]

        for i, each_color in enumerate(source_list):
            replaced_img = (img == each_color).ifthenelse(destination_list[i], replaced_img)
        replaced_img = replaced_img.cast('uchar')
        mem_img = replaced_img.write_to_memory()
        np_3d = np.ndarray(buffer=mem_img,
                           dtype=self.format_to_dtype[replaced_img.format],
                           shape=[replaced_img.height, replaced_img.width, replaced_img.bands])
        return np_3d

    def get_width_of_color_from_img(self, filename):
        img = Image.open(filename).convert('RGB')
        area = img.height * img.width
        c = img.getcolors()
        c = sorted(c, key=lambda tup: tup[0])  # sorts in place

        colors = [self.rgb2hex(x[1]) for x in c]
        coverage = [(int(100*x[0]/area))/100 for x in c]
        np_array = np.array(colors+coverage).reshape(2, 5)
        return np_array

    def choose_candidate_color_from_db(self, reduced_img_name):
        db_table = pd.read_excel(self.db_name)
        db_table = np.array(db_table)

        source_img_colors = self.get_width_of_color_from_img(reduced_img_name)
        source_img_coverage = source_img_colors[1, :]
        # print(source_img_coverage.astype('float32'))
        width_table = db_table[:db_table.shape[0]-1, 15:20]

        source_img_width_table = np.vstack([source_img_coverage.astype('float16')]*width_table.shape[0])
        difference_table = abs(width_table-source_img_width_table)
        difference_table[:, -1] = difference_table[:, 0]+difference_table[:, 1]+difference_table[:, 2]+difference_table[:, 3]+difference_table[:, 4]
        difference_table[:, -1] = db_table[:db_table.shape[0]-1, 8]
        sorted_difference_table = difference_table[difference_table[:, -2].argsort()]
        recolored_list = []
        for i in range(0, 3):
            print("closeness =", sorted_difference_table[i, -2], "Rank=", sorted_difference_table[i, -1])
            index = np.where(db_table[:, 8] == sorted_difference_table[i, -1])
            candidate_color_list = list(db_table[index[0], 10:15][0])
            source_lis = list(source_img_colors[0, :])
            recolored_list.append([self.hex2rgb(x) for x in source_lis])
            recolored_list.append([self.hex2rgb(x) for x in candidate_color_list])
        #     color_replaced = self.replace_color(self.img_name, source_list=list(source_img_colors[0, :]), destination_list=candidate_color_list)
        #     recolored_list.append(color_replaced)
        # final = np.hstack((np.asarray(Image.open(self.img_name).convert('RGB')), recolored_list[0]))
        # final2 = np.hstack((recolored_list[1], recolored_list[2]))
        # final = np.vstack((final, final2))
        # final = Image.fromarray(final.astype('uint8'))
        # final.save("Output/"+ntpath.basename(self.img_name))
        return recolored_list

    def sorting_hsv(self, rgb_color_list):
        source_hsv = [colorsys.rgb_to_hsv(x[0] / 255, x[1] / 255, x[2] / 255) for x in rgb_color_list]
        source_hsv = sorted(source_hsv, key=lambda tup: tup[2])  # sorts in place
        hsv_sorted_source_color = [colorsys.hsv_to_rgb(x[0], x[1], x[2]) for x in source_hsv]
        hsv_sorted_source_color = [(int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)) for x in hsv_sorted_source_color]
        return hsv_sorted_source_color

    def choose_color_from_grayscale(self):
        img = Image.open(self.img_name).convert('RGB')
        colors = img.getcolors()
        rgb_colors = [x[1] for x in colors]
        area = img.height*img.width

        source_colors = [self.rgb2hex(x[1]) for x in colors]
        source_grayscale = [self.hex_to_grayscale(x) for x in source_colors]
        source_distribution = [(int(100*x[0]/area))/100 for x in colors]

        grayscale_db = self.width_sorted_color_db[:, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 8, 47]] #last is gray scale and second last is rank
        source_std_table = np.full((grayscale_db.shape[0], 1), np.std(np.array(source_grayscale)))
        gracale_only_table = grayscale_db[:, 11].reshape(source_std_table.shape)
        a = abs(source_std_table-gracale_only_table)
        grayscale_db[:, 11] = a.reshape(source_std_table.shape[0],)
        grayscale_db = grayscale_db[grayscale_db[:, 11].argsort()]

        source_sorted_rgb = self.sorting_hsv(rgb_colors)
        source_sorted_rgb = [self.rgb2hex(x) for x in source_sorted_rgb]
        replaced_img_list = []
        for i in range(0, 3):
            candidate_colors = list(grayscale_db[i, 0:5])
            print(candidate_colors)
            candidate_rgb = [self.hex2rgb(x) for x in candidate_colors]
            candidate_sorted_hsv = self.sorting_hsv(candidate_rgb)
            candidate_sorted_hsv = [self.rgb2hex(x) for x in candidate_sorted_hsv]
            replaced_img = self.replace_color(self.img_name, source_sorted_rgb, candidate_sorted_hsv)
            replaced_img_list.append(replaced_img)
        horizontal_one = np.hstack((np.array(Image.open(self.img_name).convert('RGB')), replaced_img_list[0]))
        horizontal_two = np.hstack((replaced_img_list[1], replaced_img_list[2]))
        replaced_img = np.vstack((horizontal_one, horizontal_two))
        replaced_img = Image.fromarray(replaced_img.astype('uint8'))
        basename = ntpath.basename(self.img_name)
        replaced_img.save("gray_hsv/"+basename)

    def choose_palette_from_img(self):
        img = Image.open(self.img_name).convert('RGB')
        c = img.getcolors()
        # print(c)
        initial_color_number = len(c)
        # result_adaptive = img.convert('P', palette=Image.ADAPTIVE, colors=5)
        # method_one = img.quantize(colors=5, method=0)
        method_two = img.quantize(colors=5, method=1)
        # print(result_adaptive.getcolors())
        # print(method_one.getcolors())
        original_pixels = img.load()
        # pallete_label = method_two.load()

        # print(method_two.getcolors())
        method_two.save('method2.png')
        method_two = Image.open('method2.png').convert('RGB')
        # print(method_two.getcolors())
        reduced_img = method_two.load()

        relation_dict = {}
        for i in range(0, img.width):
            for j in range(0, img.height):
                if len(relation_dict) < initial_color_number:
                    current_pixel = original_pixels[i, j]
                    pixel_label = reduced_img[i, j]
                    if current_pixel not in relation_dict:
                        relation_dict[current_pixel] = pixel_label
                else:
                    break

        # print(i, j)
        print(relation_dict)

        probable_colors = self.choose_candidate_color_from_db("method2.png")
        core_level_dict_one = dict(zip(probable_colors[0], probable_colors[1]))
        core_level_dict_two = dict(zip(probable_colors[2], probable_colors[3]))
        core_level_dict_three = dict(zip(probable_colors[4], probable_colors[5]))

        replacing_dict_one = {}
        replacing_dict_two = {}
        replacing_dict_three = {}


        for k, v in relation_dict.items():
            try:
                r_ratio = v[0]/ k[0]
            except ZeroDivisionError:
                r_ratio = 1
            try:
                g_ratio = v[1]/ k[1]
            except ZeroDivisionError:
                g_ratio = 1
            try:
                b_ratio = v[2]/ k[2]
            except ZeroDivisionError:
                b_ratio = 1

            replacing_dict_one[k] = (int(r_ratio*core_level_dict_one[v][0]), int(g_ratio*core_level_dict_one[v][1]), int(b_ratio*core_level_dict_one[v][2]))
            replacing_dict_two[k] = (int(r_ratio*core_level_dict_two[v][0]), int(g_ratio*core_level_dict_two[v][1]), int(b_ratio*core_level_dict_two[v][2]))
            replacing_dict_three[k] = (int(r_ratio*core_level_dict_three[v][0]), int(g_ratio*core_level_dict_three[v][1]), int(b_ratio*core_level_dict_three[v][2]))

        print(replacing_dict_one)
        print(replacing_dict_two)
        print(replacing_dict_three)
        replaced_img_one = self.replace_color_rgb(self.img_name, list(replacing_dict_one.keys()), list(replacing_dict_one.values()))
        replaced_img_two = self.replace_color_rgb(self.img_name, list(replacing_dict_two.keys()), list(replacing_dict_two.values()))
        replaced_img_three = self.replace_color_rgb(self.img_name, list(replacing_dict_three.keys()), list(replacing_dict_three.values()))

        horizontal = np.hstack((np.asarray(Image.open(self.img_name).convert('RGB')), replaced_img_one))
        horizontal2 = np.hstack((replaced_img_two, replaced_img_three))
        replaced_img = np.vstack((horizontal, horizontal2))
        replaced_img = Image.fromarray(replaced_img.astype('uint8'))
        name = ntpath.basename(self.img_name)
        replaced_img.save("Interpolated/"+name)



if __name__ == "__main__":
    import os
    # img_name = "Rithron.png"
    # Obj = RecolorBasedOnCoverage(img_name=img_name, db_name="sorted_1000.xlsx")
    #
    dir_path, list_of_files = python_utils.get_files_of_folder("D:\\Work\\ColorRecommendation\\images")
    already_completed = []
    for each in list_of_files:
        if each[:7] not in already_completed:
            img = Image.open(os.path.join(dir_path, each)).convert('RGB')
            if len(img.getcolors()) != 5:
                img_name = os.path.join(dir_path, each)
                i = Image.open(img_name)
                i.save('Uniquie/'+each)
                # Obj = RecolorBasedOnCoverage(img_name=img_name, db_name="sorted_1000.xlsx")
            already_completed.append(each[:7])
    # Obj = RecolorBasedOnCoverage(img_name=img_name, db_name="sorted_1000.xlsx")

