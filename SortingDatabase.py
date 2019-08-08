import pandas as pd
import numpy as np
import pyvips
from PIL import Image
import colorsys


class SortingDatabase:

    def __init__(self, filename="3000ColorValues.xlsx", c=(0, 0, 0), filter_string='light', number=100):
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
        self.color_to_search = c
        self.filter_string = filter_string
        self.result_count = number
        self.color_db.drop(self.color_db.tail(1).index, inplace=True)
        self.color_db.drop_duplicates(keep=False, inplace=True)

        # sorted_hue = self.sort_based_on_one_column('l5', self.color_db)
        # self.create_image_of_result(sorted_hue[:1000], result=1000)
        # self.find_muted_palettes(self.color_db)
        # self.find_tonal_palettes(self.color_db)
        # self.find_between_range(self.color_db)
        # self.columns = self.color_db.columns.values
        # sorted_in_one_col = self.sort_based_on_one_column('lightness', self.color_db, ascending_flag=True)
        # self.create_image_of_result(sorted_in_one_col, result=100)

        # sorted_in_two = self.sort_based_on_two_columns('h5', 'l5', self.color_db)
        # self.create_image_of_result(sorted_in_two, result=300)
        # self.search_nearest_color_rgb(color=(255, 0, 0), df=self.color_db)
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
            color_cols = ['r1', 'g1', 'b1', 'r2', 'g2', 'b2', 'r3', 'g3', 'b3', 'r4', 'g4', 'b4', 'r5', 'g5', 'b5']
            rgb_df = df[color_cols].head(result)
            img_array = []
            for index, row in rgb_df.iterrows():
                blank_img = np.zeros((200, 1000, 3), dtype='uint8')
                blank_img[:, :200, :] = np.array((row['r1'], row['g1'], row['b1']))
                blank_img[:, 200:400, :] = np.array((row['r2'], row['g2'], row['b2']))
                blank_img[:, 400:600, :] = np.array((row['r3'], row['g3'], row['b3']))
                blank_img[:, 600:800, :] = np.array((row['r4'], row['g4'], row['b4']))
                blank_img[:, 800:, :] = np.array((row['r5'], row['g5'], row['b5']))
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
        df['nearest_value'] = (df['tmp_r5']-df['r5'])**2 + (df['tmp_g5']-df['g5'])**2 + (df['tmp_b5']-df['b5'])**2
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

    def create_overall_database(self):
        initial_db = pd.read_excel("total_filtered_colors.xlsx")
        columns = ['hex1', 'hex1', 'hex1', 'hex1', 'hex1', 'r1', 'g1', 'b1', 'r2', 'g2', 'b2','r3', 'g3', 'b3','r4', 'g4', 'b4','r5', 'g5', 'b5', 'h1', 'l1', 's1', 'h2', 'l2', 's2', 'h3', 'l3', 's3', 'h4', 'l4', 's4', 'h5', 'l5', 's5', 'gs1', 'gs2', 'gs3', 'gs4', 'gs5','lightness']
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
            lightness_factor = l1+l2+l3+l4+l5

            new_db.loc[index] = [row['hex1'], row['hex2'],row['hex3'], row['hex4'], row['hex5'], r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4, b4, r5, g5, b5, h1, l1, s1, h2, l2, s2, h3, l3, s3, h4, l4, s4, h5, l5, s5, gs1, gs2, gs3, gs4, gs5, lightness_factor]

        new_db.to_excel('3000ColorValues.xlsx')

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
        sorted_based_on_color = self.search_nearest_color_rgb(color=self.color_to_search, df=self.color_db)
        sorted_based_on_color = sorted_based_on_color.head(self.result_count)

        if self.filter_string == "light":
            light_candidate = self.find_light_palettes(sorted_based_on_color)
            avilable = light_candidate.boolean.sum()
            print("Light palettes Numbers", avilable)
            if avilable!=0:
                pl = self.create_image_of_result(light_candidate, avilable)

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

    def find_between_range(self, df):
        # df = self.color_db
        df['avg_saturation'] = (df['s1']+df['s2']+df['s3']+df['s4']+df['s5']).divide(5)
        df['avg_lightness'] = (df['l1']+df['l2']+df['l3']+df['l4']+df['l5']).divide(5)
        # extreme_list = self.hls_range['blue']
        # df['boolean'] = (extreme_list[0] <= df['h5']) & (df['h5'] <= extreme_list[1]) & (extreme_list[2] <= df['l5']) & (df['l5'] <= extreme_list[3])
        # df['boolean'] = (df['s5']<40) & (df['s4']<40) & (df['s3']<40) & (df['s2']<40) & (df['s1']<40)
        df['boolean'] = (df['avg_saturation']<40) & (df['avg_lightness']<40)
        sorted = self.sort_based_on_two_columns('boolean', 'l5', df, False, True)
        self.create_image_of_result(sorted, 200)

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
        df['boolean'] = (df['s5'] < 45) & (df['s4'] < 45) & (df['s3'] < 45) & (df['s2'] < 45) & (df['s1'] < 45) & (df['l5'] < 50) & (df['l4'] < 50) & (df['l3'] < 50) & (df['l2'] < 50) & (df['l1'] < 50)
        muted_candidate = self.sort_based_on_two_columns('boolean', 'lightness_std', df, False, True)
        return muted_candidate

    def find_colorful_palettes(self, df):
        df['avg_saturation'] = (df['s1']+df['s2']+df['s3']+df['s4']+df['s5']).divide(5)
        df['avg_lightness'] = (df['l1']+df['l2']+df['l3']+df['l4']+df['l5']).divide(5)
        df['hue_std'] = df[['h1', 'h2', 'h3', 'h4', 'h5']].std(axis=1)
        df['boolean'] = (df['avg_saturation']>40) & (df['avg_lightness']>40)
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




if __name__ == "__main__":
    filename = "3000ColorValues.xlsx"
    sorting_obj = SortingDatabase()
    #
    # sorting_obj = SortingDatabase(filename="3000ColorValues.xlsx", c=(200, 0, 0),
    #                               filter_string="dark", number=100)
    # sorting_obj.handle_requests()

    # rgb = (236/255, 166/255, 174/255)
    # hls = colorsys.rgb_to_hls(rgb[0], rgb[1], rgb[2])
    # print(hls[0]*360, hls[1]*100, hls[2]*100)