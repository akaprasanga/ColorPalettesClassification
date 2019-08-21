from PIL import Image
import numpy as np
from random import randrange
import os
import math

cwd = os.getcwd()
vips_path = cwd+'\\libvips'
os.environ['PATH'] = vips_path
import pyvips


class ApplyPalettes:

    def __init__(self, filename):
        self.img_name = filename
        self.dtype_to_format = {
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
        self.img = Image.open(filename).convert('RGB')
        self.area = self.img.width * self.img.height
        self.colors = self.img.getcolors()
        self.sorted_colors = sorted(self.colors, key=lambda x: x[1])
        self.sorted_colors_only = [x[1] for x in self.sorted_colors]
        self.colors_only = [x[1] for x in self.colors]
        # if len(self.colors) > 5:
        #     self.img = self.img.convert('P', palette=Image.ADAPTIVE, colors=5)
        #     self.colors = self.img.getcolors()

    def rgb2hex(self, color):
        return '{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])

    def hex2rgb(self, color):
        return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    def np_to_vips(self, img):
        height, width, bands = img.shape

        linear = img.reshape(width * height * bands)
        vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                          self.dtype_to_format[str(img.dtype)])
        return vi

    def vips_to_np(self, img):
        mem_img = img.write_to_memory()

        # then make a numpy array from that buffer object
        np_3d = np.ndarray(buffer=mem_img,
                           dtype=self.format_to_dtype[img.format],
                           shape=[img.height, img.width, img.bands])
        return np_3d

    def replace_colors(self, source_list, destination_list):
        img = self.np_to_vips(np.asarray(self.img))
        new_img = img
        # print(source_list)
        # print("from vips", destination_list)
        for i, each in enumerate(source_list):
            new_img = (img == each).ifthenelse(destination_list[i], new_img)
        return new_img

    def sort_colors_according_to_width(self, destination_colors_with_wid):

        source_dist_list = [x[0]/self.area for x in self.colors]
        source_color_list = [x[1] for x in self.colors]

        if len(source_color_list) == 1:
            one_color = []
            one_color.append(destination_colors_with_wid[4])
            return source_color_list, one_color

        elif len(source_color_list) == 2:
            return source_color_list, destination_colors_with_wid[3:5]

        elif len(source_color_list) == 3:
            return source_color_list, destination_colors_with_wid[2:5]

        elif len(source_color_list) == 4:
            adjusted_destination_color_list = destination_colors_with_wid[1:5]
            number = randrange(5)
            if number == 1:
                adjusted_destination_color_list[2], adjusted_destination_color_list[3] = adjusted_destination_color_list[3], adjusted_destination_color_list[2]
            return source_color_list, adjusted_destination_color_list

        else:
            np_source = np.array(source_color_list+source_dist_list).reshape(2, 5)
            np_destination = np.array(destination_colors_with_wid).reshape(2, 5)
            np_source = np_source.reshape(2, 5).T
            np_source = np_source[np_source[:, 1].argsort()]
            np_source = (np_source.T).ravel()

            np_destination = np_destination.reshape(2, 5).T
            np_destination = np_destination[np_destination[:, 1].argsort()]
            np_destination = (np_destination.T).ravel()

            adjusted_source_color_list = list(np_source[:5])
            adjusted_destination_color_list = list(np_destination[:5])

            number = randrange(5)
            if number == 1:
                adjusted_destination_color_list[3], adjusted_destination_color_list[4] = adjusted_destination_color_list[4], adjusted_destination_color_list[3]
            return adjusted_source_color_list, adjusted_destination_color_list

    def sort_based_on_one_column(self, column_name, df, ascending_flag=True):
        return df.sort_values(column_name, ascending=ascending_flag)

    def sort_based_on_two_columns(self, column_one, column_two, df, ascending_flag_one=True, ascending_flag_two=True):
        return df.sort_values([column_one, column_two], ascending=[ascending_flag_one, ascending_flag_two])

    def handle_requests(self, selected_dataframe, category_name, large_dataframe):
        recolored_img = []
        recolored_img.append(self.np_to_vips(np.asarray(self.img)))
        if len(self.colors) > 5:
            try:
                print('tried merging')
                list_of_sorting_columns = ['wid 5', 's5', 'h5']
                for i in range(0, 3):
                    destination_color_list = self.merge_palettes(large_dataframe, len(self.colors_only), category_name, list_of_sorting_columns[i])
                    print(i,"==",destination_color_list)
                    recolored_img.append(self.replace_colors(self.sorted_colors_only, destination_color_list))
                joined_image = pyvips.Image.arrayjoin(recolored_img, across=2, shim=10)
                joined_image = self.vips_to_np(joined_image)
                return joined_image
            except Exception as e:
                print('tried interpolation error = '+e)

                list_of_destination_colors = []
                for index, row in selected_dataframe.iterrows():
                    destination_list = [(row['r1'], row['g1'], row['b1']), (row['r2'], row['g2'], row['b2']),
                                        (row['r3'], row['g3'], row['b3']), (row['r4'], row['g4'], row['b4']),
                                        (row['r5'], row['g5'], row['b5'])]
                    list_of_destination_colors.append(self.colors_only)
                    list_of_destination_colors.append(destination_list)
                replacing_list = self.interpolation_for_more_colors(list_of_destination_colors)
                for i, each_list in enumerate(replacing_list):
                    recolored_img.append(self.replace_colors(self.colors_only, each_list))

                joined_image = pyvips.Image.arrayjoin(recolored_img, across=2, shim=10)
                joined_image = self.vips_to_np(joined_image)
                return joined_image

        else:
            for index, row in selected_dataframe.iterrows():
                destination_list = [(row['r1'], row['g1'], row['b1']), (row['r2'], row['g2'], row['b2']),(row['r3'], row['g3'], row['b3']),(row['r4'], row['g4'], row['b4']),(row['r5'], row['g5'], row['b5']), row['wid 1'], row['wid 2'], row['wid 3'], row['wid 4'], row['wid 5']]
                sorted_source_colors, sorted_destination_colors = self.sort_colors_according_to_width(destination_list)
                recolored_img.append(self.replace_colors(sorted_source_colors, sorted_destination_colors))

            joined_image = pyvips.Image.arrayjoin(recolored_img, across=2, shim=10)
            joined_image = self.vips_to_np(joined_image)
            return joined_image

    def interpolation_for_more_colors(self, probable_colors):
        img = Image.open(self.img_name).convert('RGB')
        initial_color_number = len(self.colors)
        # result_adaptive = img.convert('P', palette=Image.ADAPTIVE, colors=5)
        # method_one = img.quantize(colors=5, method=0)
        method_two = img.quantize(colors=5, method=1)
        original_pixels = img.load()
        method_two.save('method2.png')
        method_two = Image.open('method2.png').convert('RGB')
        palettes_colors = method_two.getcolors()
        palettes_colors = [x[1] for x in palettes_colors]

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
        # print(relation_dict)

        # probable_colors = self.choose_candidate_color_from_db("method2.png")
        core_level_dict_one = dict(zip(palettes_colors, probable_colors[1]))
        core_level_dict_two = dict(zip(palettes_colors, probable_colors[3]))
        core_level_dict_three = dict(zip(palettes_colors, probable_colors[5]))

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

        # print(replacing_dict_one)
        # print(replacing_dict_two)
        # print(replacing_dict_three)
        replacing_list = [list(replacing_dict_one.values()), list(replacing_dict_two.values()), list(replacing_dict_three.values())]
        return replacing_list
        # replaced_img_one = self.replace_color_rgb(self.img_name, list(replacing_dict_one.keys()), list(replacing_dict_one.values()))
        # replaced_img_two = self.replace_color_rgb(self.img_name, list(replacing_dict_two.keys()), list(replacing_dict_two.values()))
        # replaced_img_three = self.replace_color_rgb(self.img_name, list(replacing_dict_three.keys()), list(replacing_dict_three.values()))
        #
        # horizontal = np.hstack((np.asarray(Image.open(self.img_name).convert('RGB')), replaced_img_one))
        # horizontal2 = np.hstack((replaced_img_two, replaced_img_three))
        # replaced_img = np.vstack((horizontal, horizontal2))
        # replaced_img = Image.fromarray(replaced_img.astype('uint8'))
        # name = ntpath.basename(self.img_name)
        # replaced_img.save("Interpolated/"+name)

    def merge_palettes(self, df, required_color_number, category_name, main_sorting_column):
        sorted_df = self.sort_based_on_two_columns(main_sorting_column, category_name + '_closeness', df, False, True)
        iteration_needed = math.ceil((required_color_number - 5) / 4) + 1
        color_list = []
        wid_list = []
        if iteration_needed < len(sorted_df.index):
            sorted_df = sorted_df.head(iteration_needed)
            sorted_df.reset_index(inplace=True)
            for index, row in sorted_df.iterrows():
                if index == 0:
                    color_list.append((row['r5'], row['g5'], row['b5']))
                    wid_list.append(row['wid 5'])
                    color_list.extend([(row['r1'], row['g1'], row['b1']), (row['r2'], row['g2'], row['b2']),
                                       (row['r3'], row['g3'], row['b3']), (row['r4'], row['g4'], row['b4'])])
                    wid_list.extend([row['wid 1'], row['wid 2'], row['wid 3'], row['wid 4']])
                else:
                    wid_list[0] = (wid_list[0] + row['wid 5']) / 2
                    color_list.extend([(row['r1'], row['g1'], row['b1']), (row['r2'], row['g2'], row['b2']),
                                       (row['r3'], row['g3'], row['b3']), (row['r4'], row['g4'], row['b4'])])
                    wid_list.extend([row['wid 1'], row['wid 2'], row['wid 3'], row['wid 4']])

        else:
            print("Less Data to merge")
        wid_list = [x / iteration_needed for x in wid_list]
        wid_list[0] = wid_list[0] * iteration_needed
        wid_list = wid_list[:required_color_number]
        color_list = color_list[:required_color_number]
        np_sorting = np.array(color_list + wid_list).reshape(2, required_color_number).T
        np_sorting = np_sorting[np_sorting[:, 1].argsort()]
        destination_colors = list(np_sorting.T.ravel()[:required_color_number])
        return destination_colors


if __name__ == "__main__":
    Obj = ApplyPalettes("D:\Work\ColorRecommendation\images\\Ambilog.X.00.png")
    selected_colors = [(0,0,0), (255,0,0), (2,0,0), (150,0,0), (100,0,0)]
    selected_wid = [0.4, 0.2, 0.3, 0.05, 0.05]
    # selected_colors = [Obj.rgb2hex(x) for x in selected_colors]
    selected = selected_colors+selected_wid
    Obj.sort_colors_according_to_width(selected)