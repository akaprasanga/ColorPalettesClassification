import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import colorsys



class MergeDatabase:

    def __init__(self, db_name):
        self.db_name = db_name
        self.color_db = np.array(pd.read_excel(self.db_name))
        self.db_column = ['extra_index', 'id', 'title', 'userName', 'numViews', 'numVotes', 'numComments', 'numHearts', 'rank', 'dateCreated', 'color 1', 'color 2', 'color 3', 'color 4', 'color 5', 'wid 1', 'wid 2', 'wid 3', 'wid 4', 'wid 5', 'url', 'imageUrl', 'badgeUrl', 'apiUrl', 'r1', 'g1', 'b1', 'r2', 'g2', 'b2', 'r3', 'g3', 'b3', 'r4', 'g4', 'b4', 'r5', 'g5', 'b5', 'gs1', 'gs2', 'gs3', 'gs4', 'gs5', 'max gs', 'min gs', 'avggs', 'stddev gs', 'wgt r', 'wgt g', 'wgt b', 'wgt gs', 'd12', 'd13', 'd14', 'd15', 'd23', 'd24', 'd25', 'd34', 'd35', 'd45', 'max d', 'min d', 'avg d', 'stddev d', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4', 'x5', 'y5', 'z5', 'l1', 'a1', 'b1.1', 'l2', 'a2', 'b2.1', 'l3', 'a3', 'b3.1', 'l4', 'a4', 'b4.1', 'l5', 'a5', 'b5.1', 'dl12', 'dl13', 'dl14', 'dl15', 'dl23', 'dl24', 'dl25', 'dl34', 'dl35', 'dl45', 'max dl', 'min dl', 'avg dl', 'stddev dl', 'da12', 'da13', 'da14', 'da15', 'da23', 'da24', 'da25', 'da34', 'da35', 'da45', 'max da', 'min da', 'avg da', 'stddev da', 'db12', 'db13', 'db14', 'db15', 'db23', 'db24', 'db25', 'db34', 'db35', 'db45', 'max db', 'min db', 'avg db', 'stddev db', 'de12', 'de13', 'de14', 'de15', 'de23', 'de24', 'de25', 'de34', 'de35', 'de45', 'max de', 'min de', 'avg de', 'stddev de', 'h1', 'l1.1', 's1', 'h2', 'l2.1', 's2', 'h3', 'l3.1', 's3', 'h4', 'l4.1', 's4', 'h5', 'l5.1', 's5', 'dh12', 'dh13', 'dh14', 'dh15', 'dh23', 'dh24', 'dh25', 'dh34', 'dh35', 'dh45', 'dha12', 'dha13', 'dha14', 'dha15', 'dha23', 'dha24', 'dha25', 'dha34', 'dha35', 'dha45', 'max dh', 'min dh', 'avg dh', 'stddev dh']
        self.color_db = self.color_db[:-1, :]
        # self.obj_dict = {2:np.zeros(1, 18), 3:np.zeros(1, 28), 4:np.zeros(1, 38)}

    def rgb2hex(self, color):
        return '{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])

    def hex2rgb(self, color):
        return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    def hex_to_grayscale(self, hex_color):
        rgb_tuple = self.hex2rgb(hex_color)
        return 0.2126*rgb_tuple[0]+0.7152*rgb_tuple[1]+0.0722*rgb_tuple[2]

    def merge_function(self):
        counter = 0
        check_list = []
        cols = ['color 1', 'color 2', 'color 3', 'color 4', 'color 5', 'color 6', 'color 7', 'color 8', 'color 9', 'wid 1', 'wid 2', 'wid 3', 'wid 4', 'wid 5','wid 6', 'wid 7', 'wid 8', 'wid 9']
        new_db = self.color_db[0, 0:18]
        for i in range(0, self.color_db.shape[0]):
            list_one = list(self.color_db[i, 10:15])
            width_one = list(self.color_db[i, 15:20]/2)
            dict_one = dict(zip(list_one, width_one))
            for j in range(0, self.color_db.shape[0]):
                list_two = list(self.color_db[j, 10:15])
                width_two = list(self.color_db[j, 15:20]/2)
                dict_two = dict(zip(list_two, width_two))
                intersection_set = list(set(list_one).intersection(set(list_two)))
                if len(intersection_set) == 1:
                    check_string = str(i)+str(j)
                    if check_string not in check_list:
                        new_width_list = []
                        list_one = list(set(list_one) - set(intersection_set))
                        list_two = list(set(list_two) - set(intersection_set))
                        for each in list_one:
                            new_width_list.append(dict_one[each])
                        for each in list_two:
                            new_width_list.append(dict_two[each])
                        for each in intersection_set:
                            new_width_list.append(dict_one[each]+dict_two[each])
                        new_palette = list_one+list_two+intersection_set
                        new_palette_np = np.array(new_palette)
                        new_width_np = np.array(new_width_list)
                        new_palette_np = np.vstack((new_palette_np, new_width_np))
                        new_palette_np = new_palette_np.T
                        new_palette_np = new_palette_np[new_palette_np[:, 1].argsort()]
                        new_palette_np = new_palette_np.T
                        new_palette_np = new_palette_np.ravel()
                        if new_palette_np.shape[0] == 18:

                            # difference = 18 - new_palette_np.shape[0]
                            # for k in range(0, difference):
                            #     new_palette_np = np.hstack((0, new_palette_np))
                            counter = counter + 1
                            check_list.append(str(i)+str(j))
                            check_list.append(str(j)+str(i))
                            new_db = np.vstack((new_db, new_palette_np))
        new_db = new_db[1:, :]
        df = pd.DataFrame(data=new_db, columns=cols)
        df.to_excel('Merged_based_on_single_color.xlsx')
        print(counter)

    def plot_hue_and_value(self, global_hsv_list, global_hex_list):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for i, val in enumerate(global_hsv_list):
            ax.scatter(global_hsv_list[i][0], global_hsv_list[i][1], global_hsv_list[i][2], c='#'+global_hex_list[i], linewidths= 2, s=80)
            ax.grid(color='#AFEEEE', linestyle='--', linewidth=0.5)
        # plt.grid(color='#AFEEEE', linestyle='--', linewidth=0.5)
        plt.xlabel('HUE')
        plt.ylabel('S')
        plt.clabel('Lightness')
        plt.show()
        # plt.savefig('Output/'+filename, orientation='landscape', dpi=200)

    def visualize_palettes(self):
        blank_img = np.zeros((1, 5, 3), dtype='uint8')
        global_hsv_list = []
        global_hex_list = []
        for i in range(0, 20):
            hex_colors = list(self.color_db[i, 10:15])
            for each in hex_colors:
                global_hex_list.append(each)
            color_list = [self.hex2rgb(x) for x in hex_colors]
            hsv_colors = [colorsys.rgb_to_hsv(x[0], x[1], x[2]) for x in color_list]
            # for j in range(0, 5):
            #     blank_img[0, j, 0] = color_list[j][0]
            #     blank_img[0, j, 1] = color_list[j][1]
            #     blank_img[0, j, 2] = color_list[j][2]
            # pil_img = Image.fromarray(blank_img)
            # pil_img = pil_img.convert('HSV')
            # hsv_colors = pil_img.getcolors()
            # hsv_colors = [x[1] for x in hsv_colors]
            for each in hsv_colors:
                global_hsv_list.append(each)
        self.plot_hue_and_value(global_hsv_list, global_hex_list)




if __name__ == '__main__':
    obj = MergeDatabase("sorted_1000.xlsx")
    obj.visualize_palettes()