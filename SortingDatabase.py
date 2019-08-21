import pandas as pd
import numpy as np
from PIL import Image
import colorsys
import os
import math

cwd = os.getcwd()
vips_path = cwd+'\\libvips'
os.environ['PATH'] = vips_path

import pyvips

class SortingDatabase:

    def __init__(self, filename="10000_with_filters.xlsx", palette_filter='light_filter', color_category=2):
        self.color_db = pd.read_excel(filename)
        self.hls_range = {'red':[330, 360, 20, 80],
                          'yellow':[30, 90, 20, 80],
                          'green': [90, 150, 20, 80],
                          'cyan': [150, 210, 20, 80],
                          'blue': [210, 270, 20, 80],
                          'magenta': [270, 330, 20, 80]}
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
        self.names = ['BlacksandWhites', 'BluesandIndigos', 'GreensandTurquoise', 'GreensandEarth', 'YellowsandLightBrowns',
                 'OrangesandRusts', 'RedsandPinks', 'BurgundyandMaroons', 'BrownsandBeiges', 'PurplesandViolets']
        # self.color_to_search = c
        self.palette_filter = palette_filter
        self.color_category = color_category
        # self.result_count = number
        self.color_db.drop(self.color_db.tail(1).index, inplace=True)
        self.color_db.drop_duplicates(keep=False, inplace=True)
        # df = self.color_db.copy()
        # self.add_category_filter(df)

        # self.merge_palettes(self.color_db, 45, self.names[2], 'wid 5')
        # self.choose_from_one_category(df, self.names[3])
        # self.select_color_category_from_interpolation(df, c1=(42, 47, 59), c2=(103, 108, 120), c3=(173, 179, 190))
        # refned, count = self.choose_from_two_category(self.color_category, self.palette_filter)
        # print("Total Colors in the selected Category::", count)
        # self.create_image_of_result(refned, count)
        # self.add_category_filter(df)
        # df['test_bool'] = (df['light_filter']==False) & (df['dark_filter']==False) & (df['muted_filter']==False) & (df['colorful_filter']==False) & (df['tone_filter']==False)
        # df['test_bool'] = df['colorful_filter']==True
        # left_out = self.sort_based_on_one_column('test_bool', df, False)
        # print(left_out.test_bool.sum())
        # self.create_image_of_result(left_out, 300)

        # self.manage_large_db('HugeSqlDb.xlsx')
        # df = pd.read_excel('10000ColorValuesHLS.xlsx')
        # self.sort_df_with_width(df)
        # self.create_overall_database()
        # self.create_overall_database()
        # self.add_category_filter(df)
        # self.find_tonal_palettes(self.color_db)
        # self.find_between_range(self.color_db)
        # self.columns = self.color_db.columns.values
        # sorted_in_one_col = self.sort_based_on_one_column('lightness', self.color_db, ascending_flag=True)
        # self.create_image_of_result(sorted_in_one_col, result=100)

        # sorted_in_two = self.sort_based_on_two_columns('h5', 'l5', self.color_db)
        # self.create_image_of_result(sorted_in_two, result=300)
        # selected = self.search_nearest_color_rgb(color=(131, 102, 136), df=self.color_db)
        # self.create_image_of_result(selected)
        # self.create_overall_database()

    def np_to_vips(self, img):
        height, width, bands = img.shape
        linear = img.reshape(width * height * bands)
        vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                          self.dtype_to_format[str(img.dtype)])
        return vi

    def sort_based_on_one_column(self, column_name, df, ascending_flag=True):
        return df.sort_values(column_name, ascending=ascending_flag)

    def sort_based_on_two_columns(self, column_one, column_two, df, ascending_flag_one=True, ascending_flag_two=True):
        return df.sort_values([column_one, column_two], ascending=[ascending_flag_one, ascending_flag_two])

    def create_image_of_result(self, df, result=50, tonal=False):
        if tonal:
            color_cols = ['h1', 'l1', 's1', 'h2', 'l2', 's2', 'h3', 'l3', 's3', 'h4', 'l4', 's4', 'h5', 'l5', 's5']
            rgb_df = df[color_cols].head(result)
            img_array = []
            for index, row in rgb_df.iterrows():
                hls_colors = [(row['h1'], row['l1'], row['s1']), (row['h2'], row['l2'], row['s2']), (row['h3'], row['l3'], row['s3']), (row['h4'], row['l4'], row['s4']), (row['h5'], row['l5'], row['s5'])]
                hls_colors = sorted(hls_colors, key=lambda x: x[1])
                rgb_colors = [self.hls2rgb(x) for x in hls_colors]
                blank_img = np.zeros((200, 1000, 3), dtype='uint8')

                # blank_img[:, :200, :] = np.array((row['r1'], row['g1'], row['b1']))
                # blank_img[:, 200:400, :] = np.array((row['r2'], row['g2'], row['b2']))
                # blank_img[:, 400:600, :] = np.array((row['r3'], row['g3'], row['b3']))
                # blank_img[:, 600:800, :] = np.array((row['r4'], row['g4'], row['b4']))
                # blank_img[:, 800:, :] = np.array((row['r5'], row['g5'], row['b5']))

                blank_img[:, :200, :] = np.array(rgb_colors[0])
                blank_img[:, 200:400, :] = np.array(rgb_colors[1])
                blank_img[:, 400:600, :] = np.array(rgb_colors[2])
                blank_img[:, 600:800, :] = np.array(rgb_colors[3])
                blank_img[:, 800:, :] = np.array(rgb_colors[4])
                img_array.append(self.np_to_vips(blank_img))

            joined_green = pyvips.Image.arrayjoin(img_array, across=1, shim=8)
            joined_green.write_to_file('sorted.png')
            pl = Image.open('sorted.png').convert('RGB')
            pl.show()
            return pl
        else:
            color_cols = ['r1', 'g1', 'b1', 'r2', 'g2', 'b2', 'r3', 'g3', 'b3', 'r4', 'g4', 'b4', 'r5', 'g5', 'b5', 'wid 1', 'wid 2', 'wid 3', 'wid 4', 'wid 5']
            rgb_df = df[color_cols].head(result)
            img_array = []
            for index, row in rgb_df.iterrows():
                blank_img = np.zeros((200, 1000, 3), dtype='uint8')
                w_list = [int(1000*row['wid 1']), int(1000*row['wid 2']), int(1000*row['wid 3']), int(1000*row['wid 4']), int(1000*row['wid 5'])]
                wid_l = [w_list[0], w_list[0]+w_list[1], w_list[0]+w_list[1]+w_list[2], w_list[0]+w_list[1]+w_list[2]+w_list[3]]
                blank_img[:, :wid_l[0], :] = np.array((row['r1'], row['g1'], row['b1']))
                blank_img[:, wid_l[0]:wid_l[1], :] = np.array((row['r2'], row['g2'], row['b2']))
                blank_img[:, wid_l[1]:wid_l[2], :] = np.array((row['r3'], row['g3'], row['b3']))
                blank_img[:, wid_l[2]:wid_l[3], :] = np.array((row['r4'], row['g4'], row['b4']))
                blank_img[:, wid_l[3]:, :] = np.array((row['r5'], row['g5'], row['b5']))
                img_array.append(self.np_to_vips(blank_img))

            joined_green = pyvips.Image.arrayjoin(img_array, across=1, shim=8)
            joined_green.write_to_file('sorted.png')
            pl = Image.open('sorted.png').convert('RGB')
            pl.show()
            return pl

    def create_lightness_table(self, df):
        df['lightness_summed'] = df['l1']+df['l2']+df['l3']+df['l4']+df['l5']
        sorted_based_on_column = self.sort_based_on_one_column('lightness_summed', df)
        self.create_image_of_result(sorted_based_on_column, 50)

    def search_nearest_color_rgb(self, color=(255, 0, 0), df=None):
        # df['tmp_r5'], df['tmp_g5'], df['tmp_b5'] = df['r5'].copy(), df['g5'].copy(), df['b5'].copy()
        # df['nearest_value'] = abs(df['tmp_r5']-color[0])+abs(df['tmp_g5']-color[1])+abs(df['tmp_b5']-color[2])
        df['tmp_r5'], df['tmp_g5'], df['tmp_b5'] = color[0], color[1], color[2]
        df['nearest_value'] = ((df['tmp_r5']-df['r5'])**2 + (df['tmp_g5']-df['g5'])**2 + (df['tmp_b5']-df['b5'])**2)**0.5
        sorted_based_on_nearest_color = self.sort_based_on_one_column('nearest_value', df)
        # self.create_image_of_result(sorted_based_on_nearest_color)
        return sorted_based_on_nearest_color

    def search_nearest_color_hls(self, color=(255, 0, 0), df=None):
        search_hls = colorsys.rgb_to_hls(color[0]/255, color[1]/255, color[2]/255)
        search_hls = tuple((search_hls[0]*360, search_hls[1]*100, search_hls[2]*100))
        df['tmp_r5'], df['tmp_g5'], df['tmp_b5'] = df['h5'].copy(), df['l5'].copy(), df['s5'].copy()
        df['nearest_value'] = abs(df['tmp_r5']-search_hls[0])+abs(df['tmp_g5']-search_hls[1])+abs(df['tmp_b5']-search_hls[2])
        sorted_based_on_nearest_color = self.sort_based_on_one_column('nearest_value', df)
        self.create_image_of_result(sorted_based_on_nearest_color)

    def rgb2hex(self, color):
        return '{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])

    def hex2rgb(self, color):
        return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    def hex_to_grayscale(self, hex_color):
        rgb_tuple = self.hex2rgb(hex_color)
        return 0.2126 * rgb_tuple[0] + 0.7152 * rgb_tuple[1] + 0.0722 * rgb_tuple[2]

    def hex2hls(self, color):
        rgb_tuple = self.hex2rgb(color)
        h, l, s = colorsys.rgb_to_hls(rgb_tuple[0]/255, rgb_tuple[1]/255, rgb_tuple[2]/255)
        return tuple((h*360, l*100, s*100))

    def hls2rgb(self, color):
        h, l, s = color[0]/360, color[1]/100, color[2]/100
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return tuple((int(r*255), int(g*255), int(b*255)))

    def handle_requests(self):
        # sorted_based_on_color = self.search_nearest_color_rgb(color=self.color_to_search, df=self.color_db)
        # sorted_based_on_color = sorted_based_on_color.head(self.result_count)
        # sorted_based_on_color = locals()[self.color_category]()
        print(self.color_category)
        func = getattr(self, self.color_category)()
        sorted_based_on_color = func()

        if self.filter_string == "light":
            light_candidate = self.sort_based_on_one_column('light_filter', sorted_based_on_color,False)
            avilable = light_candidate.light_filter.sum()
            print("Light palettes Numbers", avilable)
            if avilable!=0:
                pl = self.create_image_of_result(light_candidate.head(avilable), avilable)

        if self.filter_string == "dark":
            dark_candidate = self.find_dark_palettes(sorted_based_on_color)
            avilable = dark_candidate.boolean.sum()
            print("Dark palettes Numbers", avilable)
            if avilable!=0:
                pl = self.create_image_of_result(dark_candidate, avilable)

        if self.filter_string == "tone":
            tonal_candidate = self.find_tonal_palettes(sorted_based_on_color)
            pl = self.create_image_of_result(tonal_candidate, self.result_count, True)
            # pl.show(title="Tone on Tone")

        if self.filter_string == "muted":
            muted_candidate = self.find_muted_palettes(sorted_based_on_color)
            avilable = muted_candidate.boolean.sum()
            print("Muted palettes Numbers", avilable)
            if avilable!=0:
                pl = self.create_image_of_result(muted_candidate, avilable)

        if self.filter_string == "colorful":
            colorful_candidate = self.find_colorful_palettes(sorted_based_on_color)
            avilable = colorful_candidate.boolean.sum()
            print("Colorful palettes Numbers", avilable)
            if avilable!=0:
                pl = self.create_image_of_result(colorful_candidate, avilable)

    def find_between_range(self, range_list):
        df = self.color_db
        df['boolean'] = (df['avg_hue']>range_list[0]) & (df['avg_hue']<range_list[1]) & \
                        (df['avg_lightness']>range_list[2])  & (df['avg_lightness']<range_list[3]) & \
                        (df['avg_saturation'] > range_list[4]) & (df['avg_saturation'] < range_list[5]) & \
                        (df['h5'] > range_list[6]) & (df['h5'] < range_list[7]) & (df['h4'] > range_list[6]) & (df['h4'] < range_list[7]) &(df['h3'] > range_list[6]) & (df['h3'] < range_list[7]) &(df['h2'] > range_list[6]) & (df['h2'] < range_list[7]) &(df['h1'] > range_list[6]) & (df['h1'] < range_list[7]) &\
                        (df['l5'] > range_list[6]) & (df['l5'] < range_list[7]) & (df['l4'] > range_list[6]) & (df['l4'] < range_list[7]) &(df['l3'] > range_list[6]) & (df['l3'] < range_list[7]) &(df['l2'] > range_list[6]) & (df['l2'] < range_list[7]) &(df['l1'] > range_list[6]) & (df['l1'] < range_list[7]) &\
                        (df['s5'] > range_list[6]) & (df['s5'] < range_list[7]) & (df['s4'] > range_list[6]) & (df['s4'] < range_list[7]) &(df['s3'] > range_list[6]) & (df['s3'] < range_list[7]) &(df['s2'] > range_list[6]) & (df['s2'] < range_list[7]) &(df['s1'] > range_list[6]) & (df['s1'] < range_list[7])
        avilable = df.boolean.sum()
        if avilable > 300:
            avilable = 300
        df = self.sort_based_on_one_column('boolean', df, False)
        self.create_image_of_result(df.head(avilable), avilable)

    def find_tonal_palettes(self, df):
        df['hue_std'] = df[['h1', 'h2', 'h3', 'h4', 'h5']].std(axis=1)
        df['lightness_std'] = df[['l1', 'l2', 'l3', 'l4', 'l5']].std(axis=1)

        tonal_candidate = self.sort_based_on_two_columns('hue_std', 'lightness_std', df, True, False)
        # self.create_image_of_result(tonal_candidate, 100)
        return tonal_candidate

    def find_muted_palettes(self, df):
        df['avg_saturation'] = (df['s1']+df['s2']+df['s3']+df['s4']+df['s5']).divide(5)
        df['avg_lightness'] = (df['l1']+df['l2']+df['l3']+df['l4']+df['l5']).divide(5)
        df['lightness_std'] = df[['l1', 'l2', 'l3', 'l4', 'l5']].std(axis=1)
        df['boolean'] = (df['s5'] < 45) & (df['s4'] < 45) & (df['s3'] < 45) & (df['s2'] < 45) & (df['s1'] < 45) & (df['l5'] < 55) & (df['l4'] < 55) & (df['l3'] < 55) & (df['l2'] < 55) & (df['l1'] < 55)
        muted_candidate = self.sort_based_on_two_columns('boolean', 'lightness_std', df, False, True)
        return muted_candidate

    def find_colorful_palettes(self, df):
        df['avg_saturation'] = (df['s1']+df['s2']+df['s3']+df['s4']+df['s5']).divide(5)
        df['avg_lightness'] = (df['l1']+df['l2']+df['l3']+df['l4']+df['l5']).divide(5)
        df['hue_std'] = df[['h1', 'h2', 'h3', 'h4', 'h5']].std(axis=1)
        # df['boolean'] = (df['avg_saturation']>30) & (df['avg_lightness']>30)
        df['boolean'] = (df['s5'] > 33) & (df['s4'] > 33) & (df['s3'] > 33) & (df['s2'] > 33) & (df['s1'] > 33) & (df['l5'] > 35) & (df['l4'] > 35) & (df['l3'] > 35) & (df['l2'] > 35) & (df['l1'] > 35)

        colorful_candidate = self.sort_based_on_two_columns('boolean', 'hue_std', df, False, False)
        return colorful_candidate

    def find_light_palettes(self, df):
        df['hue_std'] = df[['h1', 'h2', 'h3', 'h4', 'h5']].std(axis=1)
        df['boolean'] = (df['l5'] > 40) & (df['l4'] > 40) & (df['l3'] > 40) & (df['l2'] > 40) & (df['l1'] > 40)
        lightcolor_candidate = self.sort_based_on_two_columns('boolean', 'hue_std', df, False, False)
        return lightcolor_candidate

    def find_dark_palettes(self, df):
        df['hue_std'] = df[['h1', 'h2', 'h3', 'h4', 'h5']].std(axis=1)
        df['boolean'] = (df['l5'] < 50) & (df['l4'] < 50) & (df['l3'] < 50) & (df['l2'] < 50) & (df['l1'] < 50)
        lightcolor_candidate = self.sort_based_on_two_columns('boolean', 'hue_std', df, False, False)
        return lightcolor_candidate

    def find_blackwhites(self, df):
        df['boolean_1'] = (df['h5'] < 250) & (df['h5'] > 200) & (df['s5'] < 40)
        # selected = self.sort_based_on_one_column('boolean', df, False)
        # avilable = selected.boolean.sum()
        # return selected.head(avilable)
        return df

    def find_blueindigos(self, df):
        df['boolean_2'] = (210 < df['h5']) & (df['h5'] < 240) & (df['s5'] > 30)
        # selected = self.sort_based_on_one_column('boolean', df, False)
        # avilable = selected.boolean.sum()
        # return selected.head(avilable)
        return df

    def find_greenturtoise(self, df):
        df['boolean_3'] = (140 < df['h5']) & (df['h5'] < 200) & (df['s5'] < 40) & (df['s5'] > 14)
        # selected = self.sort_based_on_one_column('boolean', df, False)
        # avilable = selected.boolean.sum()
        # return selected.head(avilable)
        return df

    def find_greenearth(self, df):
        df['boolean_4'] = (60 < df['h5']) & (df['h5'] < 90) & (df['s5'] >20)&(df['s5'] < 60)
        # selected = self.sort_based_on_one_column('boolean', df, False)
        # avilable = selected.boolean.sum()
        # return selected.head(avilable)
        return df

    def find_yellowbrowns(self, df):
        df['boolean_5'] = (50 < df['h5']) & (df['h5'] < 60) & (df['s5'] > 60)
        # selected = self.sort_based_on_one_column('boolean', df, False)
        # avilable = selected.boolean.sum()
        # return selected.head(avilable)
        return df

    def find_orangerusts(self, df):
        df['boolean_6'] = (df['h5'] < 45) & (df['s5'] > 20)
        # selected = self.sort_based_on_one_column('boolean', df, False)
        # avilable = selected.boolean.sum()
        # return selected.head(avilable)
        return df

    def find_redpinks(self, df):
        df['boolean_7'] = ((df['h5'] > 340) & (df['s5'] > 35)) | ((df['h5'] < 10) & (df['s5'] > 35))
        # selected = self.sort_based_on_one_column('boolean', df, False)
        # avilable = selected.boolean.sum()
        # return selected.head(avilable)
        return df

    def find_burgundymaroons(self, df):
        df['boolean_8'] = ((df['h5'] > 345) & (df['s5'] > 15) & (df['s5'] < 40) & (df['l5'] < 50)) | ((df['h5'] < 16) & (df['s5'] > 26) & (df['s5'] < 40) & (df['l5'] < 50))
        # selected = self.sort_based_on_one_column('boolean', df, False)
        # avilable = selected.boolean.sum()
        # return selected.head(avilable)
        return df

    def find_brownsbeiges(self, df):
        df['boolean_9'] = ((df['h5'] < 50) & (df['s5'] < 25) ) | ((df['h5'] > 330) & (df['h5'] < 340) & (df['s5'] < 21))
        # selected = self.sort_based_on_one_column('boolean', df, False)
        # avilable = selected.boolean.sum()
        # return selected.head(avilable)
        return df

    def find_purpleviolets(self, df):
        df['boolean_10'] = (df['h5'] > 245) & (df['h5'] < 350) & (df['s5'] < 25)
        # selected = self.sort_based_on_one_column('boolean', df, False)
        # avilable = selected.boolean.sum()
        # return selected.head(avilable)
        return df

# sort the corlors according to width for whole database
    def sort_df_with_width(self, df):
        pandas_col = ["rank", "hex1", "hex2", "hex3", "hex4", "hex5", "wid 1", "wid 2", "wid 3", "wid 4", "wid 5"]
        sorted_db = pd.DataFrame(columns=pandas_col)
        for index, row in df.iterrows():
            array = np.array([row['hex1'], row['hex2'], row['hex3'], row['hex4'], row['hex5'], row['wid 1'], row['wid 2'], row['wid 3'], row['wid 4'], row['wid 5']])
            array = array.reshape(2, 5).T
            array = array[array[:, 1].argsort()]
            array = (array.T).ravel()
            sorted_db.loc[index] = [row['palette_rank'], array[0], array[1], array[2], array[3], array[4], array[5], array[6], array[7], array[8], array[9]]

        sorted_db.to_excel("10000_wid_sorted.xlsx")

# create hls and grayscales values after sorting the width
    def create_overall_database(self):
        initial_db = pd.read_excel("10000_wid_sorted.xlsx")
        columns = ['hex1', 'hex1', 'hex1', 'hex1', 'hex1', 'r1', 'g1', 'b1', 'r2', 'g2', 'b2','r3', 'g3', 'b3','r4', 'g4', 'b4','r5', 'g5', 'b5', 'h1', 'l1', 's1', 'h2', 'l2', 's2', 'h3', 'l3', 's3', 'h4', 'l4', 's4', 'h5', 'l5', 's5', 'gs1', 'gs2', 'gs3', 'gs4', 'gs5','lightness', 'wid 1', 'wid 2', 'wid 3', 'wid 4', 'wid 5', 'rank']
        new_db = pd.DataFrame(columns=columns)
        for index, row in initial_db.iterrows():
            r1, g1, b1 = self.hex2rgb(row['hex1'])
            r2, g2, b2 = self.hex2rgb(row['hex2'])
            r3, g3, b3 = self.hex2rgb(row['hex3'])
            r4, g4, b4 = self.hex2rgb(row['hex4'])
            r5, g5, b5 = self.hex2rgb(row['hex5'])
            h1, l1, s1 = self.hex2hls(row['hex1'])
            h2, l2, s2 = self.hex2hls(row['hex2'])
            h3, l3, s3 = self.hex2hls(row['hex3'])
            h4, l4, s4 = self.hex2hls(row['hex4'])
            h5, l5, s5 = self.hex2hls(row['hex5'])
            gs1 = self.hex_to_grayscale(row['hex1'])
            gs2 = self.hex_to_grayscale(row['hex2'])
            gs3 = self.hex_to_grayscale(row['hex3'])
            gs4 = self.hex_to_grayscale(row['hex4'])
            gs5 = self.hex_to_grayscale(row['hex5'])
            lightness_factor = (l1+l2+l3+l4+l5)/5

            new_db.loc[index] = [row['hex1'], row['hex2'], row['hex3'], row['hex4'], row['hex5'], r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4, b4, r5, g5, b5, h1, l1, s1, h2, l2, s2, h3, l3, s3, h4, l4, s4, h5, l5, s5, gs1, gs2, gs3, gs4, gs5, lightness_factor, row['wid 1'], row['wid 2'], row['wid 3'], row['wid 4'], row['wid 5'], row['rank']]

        new_db.to_excel('10000ColorValuesHLS.xlsx')

    def add_random_width(self):
        small = pd.read_excel("3000ColorValues.xlsx")
        large = pd.read_excel("20000ColorValues.xlsx")

        height = small.shape[0]
        large = large.head(height)

        small['wid 1'] = large['wid 1']
        small['wid 2'] = large['wid 2']
        small['wid 3'] = large['wid 3']
        small['wid 4'] = large['wid 4']
        small['wid 5'] = large['wid 5']

        small.drop(small.columns[[0]], axis=1, inplace=True)
        small.to_excel('3000wid.xlsx', index=False)

# create a category wise boolean
    def add_category_filter(self, df):
        df_copy = df.copy()
        names = ['BlacksandWhites', 'BluesandIndigos', 'GreensandTurquoise', 'GreensandEarth', 'YellowsandLightBrowns',
                 'OrangesandRusts', 'RedsandPinks', 'BurgundyandMaroons', 'BrownsandBeiges', 'PurplesandViolets']
        df_copy['avg_lightness'] = df['lightness']
        df_copy['avg_hue'] = (df['h1']+df['h2']+df['h3']+df['h4']+df['h5'])/5
        df_copy['avg_saturation'] = (df['s1']+df['s2']+df['s3']+df['s4']+df['s5'])/5
        df_copy['hue_std'] = df[['h1', 'h2', 'h3', 'h4', 'h5']].std(axis=1)
        df_copy['lightness_std'] = df[['l1', 'l2', 'l3', 'l4', 'l5']].std(axis=1)
        df_copy['saturation_std'] = df[['s1', 's2', 's3', 's4', 's5']].std(axis=1)

        df_copy['dark_filter'] = (df_copy['avg_lightness']<50) & (df_copy['l1']<50) & (df_copy['l2']<50) & (df_copy['l3']<50) & (df_copy['l4']<50) & (df_copy['l5']<50)
        df_copy['muted_filter'] = (df_copy['avg_saturation']<45) & (df_copy['s1']<50) & (df_copy['s2']<50) & (df_copy['s3']<50) & (df_copy['s4']<50) & (df_copy['s5']<50) & (df_copy['avg_lightness']<45)
        #
        df_copy['light_filter'] = (df_copy['avg_lightness']>40) & (df_copy['l1']>50) & (df_copy['l2']>50) & (df_copy['l3']>50) & (df_copy['l4']>50) & (df_copy['l5']>50)
        df_copy['colorful_filter'] = (df_copy['avg_saturation']>30) & (df_copy['saturation_std']>5) & (df_copy['avg_lightness']>35) & (df_copy['hue_std']>5)& (df_copy['lightness_std']>8)
        df_copy['tone_filters'] = df_copy['hue_std']<11

        filtered_df = self.find_blackwhites(df)
        df_copy[names[0]] = filtered_df['boolean_1']

        filtered_df = self.find_blueindigos(df)
        df_copy[names[1]] = filtered_df['boolean_2']
        filtered_df = self.find_greenturtoise(df)
        df_copy[names[2]] = filtered_df['boolean_3']
        filtered_df = self.find_greenearth(df)
        df_copy[names[3]] = filtered_df['boolean_4']
        filtered_df = self.find_yellowbrowns(df)
        df_copy[names[4]] = filtered_df['boolean_5']
        filtered_df = self.find_orangerusts(df)
        df_copy[names[5]] = filtered_df['boolean_6']
        filtered_df = self.find_redpinks(df)
        df_copy[names[6]] = filtered_df['boolean_7']
        filtered_df = self.find_burgundymaroons(df)
        df_copy[names[7]] = filtered_df['boolean_8']
        filtered_df = self.find_brownsbeiges(df)
        df_copy[names[8]] = filtered_df['boolean_9']
        filtered_df = self.find_purpleviolets(df)
        df_copy[names[9]] = filtered_df['boolean_10']

        column = self.compute_categorical_closeness(df_copy, (55, 60, 72))
        df_copy[names[0]+'_closeness'] = column['closeness']

        column = self.compute_categorical_closeness(df_copy, (62, 80, 158))
        df_copy[names[1]+'_closeness'] = column['closeness']

        column = self.compute_categorical_closeness(df_copy, (37, 119, 142))
        df_copy[names[2]+'_closeness'] = column['closeness']

        column = self.compute_categorical_closeness(df_copy, (69, 82, 32))
        df_copy[names[3]+'_closeness'] = column['closeness']

        column = self.compute_categorical_closeness(df_copy, (205, 182, 63))
        df_copy[names[4]+'_closeness'] = column['closeness']

        column = self.compute_categorical_closeness(df_copy, (180, 104, 60))
        df_copy[names[5]+'_closeness'] = column['closeness']

        column = self.compute_categorical_closeness(df_copy, (205, 88, 109))
        df_copy[names[6]+'_closeness'] = column['closeness']

        column = self.compute_categorical_closeness(df_copy, (98, 41, 58))
        df_copy[names[7]+'_closeness'] = column['closeness']

        column = self.compute_categorical_closeness(df_copy, (131, 92, 67))
        df_copy[names[8]+'_closeness'] = column['closeness']

        column = self.compute_categorical_closeness(df_copy, (115, 89, 125))
        df_copy[names[9]+'_closeness'] = column['closeness']

        print("Colorful::", df_copy.colorful_filter.sum())
        print("Muted::", df_copy.muted_filter.sum())
        print("Light::", df_copy.light_filter.sum())
        print("Dark::", df_copy.dark_filter.sum())
        print("Tone::", df_copy.tone_filters.sum())

        # dark_colors = self.find_dark_palettes(df)
        # df_copy['dark_filter'] = dark_colors['boolean']
        #
        # muted_colors = self.find_muted_palettes(df)
        # df_copy['muted_filter'] = muted_colors['boolean']
        #
        # colorful_colors = self.find_colorful_palettes(df)
        # df_copy['colorful_filter'] = colorful_colors['boolean']

        # df_copy.drop(df_copy.columns[[0]], axis=1, inplace=True)
        # print(df_copy)
        # df_copy = df_copy.iloc[:, 0:]
        # print(df_copy)

        df_copy.to_excel('10000_with_filters.xlsx')



    def compute_categorical_closeness(self, df, median_color=(0, 0, 0)):
        df['closeness'] = ((df['r5']-median_color[0])**2+(df['g5']-median_color[1])**2+(df['b5']-median_color[2])**2)**0.5
        # print(df['closeness'])
        return df

    def choose_from_two_category(self, color_category, palettes_category):
        df = self.color_db
        color_category = self.names[color_category]
        df['boolean'] = (df[color_category]) & (df[palettes_category])
        selected = self.sort_based_on_one_column('boolean', df, False)
        avialable_number = selected.boolean.sum()
        if avialable_number>300:
            selected = selected.head(300)
        else:
            selected = selected.head(avialable_number)
        if palettes_category == 'light_filter':
            selected = self.sort_based_on_two_columns(color_category+'_closeness', 'avg_lightness', selected, True, False)
        if palettes_category == 'dark_filter':
            selected = self.sort_based_on_two_columns(color_category+'_closeness', 'avg_lightness', selected, True, True)
        if palettes_category == 'colorful_filter':
            selected = self.sort_based_on_two_columns(color_category+'_closeness', 'hue_std', selected, True, False)
        if palettes_category == 'muted_filter':
            selected = self.sort_based_on_two_columns(color_category+'_closeness', 'avg_lightness', selected, True, True)
        # self.create_image_of_result(selected, avialable_number)
        return selected, avialable_number

# used for extract data from single category filter
    def choose_from_one_category(self, df, filter_string):
        selected = self.sort_based_on_one_column(filter_string, df, False)
        avilable = selected[filter_string].sum()
        if avilable > 300:
            avilable = 300
        print(avilable)
        self.create_image_of_result(selected.head(avilable), avilable)

# it is used to extract the colors from RGB lines
    def select_color_category_from_interpolation(self, df, c1, c2, c3):
        color_distance = 50
        df['c1'] = (((df['r5']-c1[0])**2+(df['g5']-c1[1])**2+(df['b5']-c1[2])**2)**0.5) < color_distance
        df['c2'] = (((df['r5']-c2[0])**2+(df['g5']-c2[1])**2+(df['b5']-c2[2])**2)**0.5) < color_distance
        df['c3'] = (((df['r5']-c3[0])**2+(df['g5']-c3[1])**2+(df['b5']-c3[2])**2)**0.5) < color_distance

        df['left'] = (df['c1']) & (df['c2'])
        df['right'] = (df['c2']) & (df['c3'])
        df['selected_color'] = (df['left']) | (df['right'])
        selected = self.sort_based_on_one_column('selected_color', df, False)
        available = df.selected_color.sum()
        print(available)
        selected = selected.head(available)
        self.create_image_of_result(selected, available)

        print(df)

if __name__ == "__main__":
    filename = "10000_with_filters.xlsx"
    sorting_obj = SortingDatabase()

    # sorting_obj = SortingDatabase(filename="3000ColorValues.xlsx", c=(200, 0, 0),
    #                               filter_string="dark", number=100)
    # sorting_obj.handle_requests()

    # rgb = (236/255, 166/255, 174/255)
    # hls = colorsys.rgb_to_hls(rgb[0], rgb[1], rgb[2])
    # print(hls[0]*360, hls[1]*100, hls[2]*100)