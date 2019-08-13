from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import QDateTime, Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PIL import Image
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import QDateTime, Qt, QTimer, QThread, pyqtSignal, QRectF, QPoint, QSize
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
import cv2
from PIL import Image
import numpy as np


class WidgetGallery(QDialog):
    worker_thread_signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)

        parameter_groupbox = self.crete_parameter_groupbox()
        w = screen.size().width()

        parameter_groupbox.setFixedWidth(w // 5)

        image_groupbox = self.create_input_img_groupbox()
        seam_groupbox = self.create_seam_img_groupbox()
        ordinary_groupbox = self.create_second_img_groupbox()

        main_layout = QGridLayout()
        main_layout.addWidget(parameter_groupbox, 0, 0)
        main_layout.addWidget(image_groupbox, 0, 1)
        # main_layout.addWidget(seam_groupbox, 1, 1)
        # main_layout.addWidget(ordinary_groupbox, 2, 1)

        self.timer_id = -1
        styleComboBox = QComboBox()
        styleComboBox.addItems(QStyleFactory.keys())
        self.current_img_path = 'palettesimage.png'
        self.current_img = np.array(Image.open(self.current_img_path).convert('RGB'))
        self.setWindowTitle("ColorFilter")
        self.changeStyle('Fusion')
        self.render_image(self.current_img, self.input_scene)
        self.setLayout(main_layout)
        self.selected_category = "light"
        self.selected_color_value = np.array([0, 0, 0])
        self.worker_process = WorkerThread()
        self.connect_signals()


    def changeStyle(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))

    def connect_signals(self):
        self.search_colors_btn.clicked.connect(self.value_change)
        # self.low_avg_hue_spinbox.valueChanged.connect(self.value_change)
        # self.high_avg_hue_spinbox.valueChanged.connect(self.value_change)
        # self.low_avg_lightness_spinbox.valueChanged.connect(self.value_change)
        # self.high_avg_lightness_spinbox.valueChanged.connect(self.value_change)
        # self.low_avg_saturation_spinbox.valueChanged.connect(self.value_change)
        # self.high_avg_saturation_spinbox.valueChanged.connect(self.value_change)

        self.worker_process.worker_thread_signal.connect(self.worker_thread_complete)


    def crete_parameter_groupbox(self):
        parameter_groupbox = QGroupBox("Parameters")

        self.search_colors_btn = QPushButton('Search Palettes')
        selected_color_text = QLabel("Hue Range: 0-360 \nLightness Range: 0-100 \nSaturation Range:0-100")

        filter_group_box = QGroupBox("Average Values")
        vbox = QGridLayout()
        self.low_avg_hue_spinbox = QSpinBox()
        avg_hue_lbl = QLabel("Average Hue")
        self.high_avg_hue_spinbox = QSpinBox()
        self.high_avg_hue_spinbox.setMaximum(360)
        self.high_avg_hue_spinbox.setValue(360)

        self.low_avg_lightness_spinbox = QSpinBox()
        avg_lightness_lbl = QLabel("Average Lightness")
        self.high_avg_lightness_spinbox = QSpinBox()
        self.high_avg_lightness_spinbox.setValue(100)

        self.low_avg_saturation_spinbox = QSpinBox()
        avg_saturation_lbl = QLabel("Average Saturation")
        self.high_avg_saturation_spinbox = QSpinBox()
        self.high_avg_saturation_spinbox.setValue(100)

        vbox.addWidget(self.low_avg_hue_spinbox, 0,1)
        vbox.addWidget(avg_hue_lbl, 0,2)
        vbox.addWidget(self.high_avg_hue_spinbox, 0,3)
        vbox.addWidget(self.low_avg_lightness_spinbox, 1,1)
        vbox.addWidget(avg_lightness_lbl, 1,2)
        vbox.addWidget(self.high_avg_lightness_spinbox, 1,3)
        vbox.addWidget(self.low_avg_saturation_spinbox, 2,1)
        vbox.addWidget(avg_saturation_lbl, 2,2)
        vbox.addWidget(self.high_avg_saturation_spinbox, 2,3)

        individual_filter_group_box = QGroupBox("Individual Values")
        vbox_ind = QGridLayout()
        self.low_ind_hue_spinbox = QSpinBox()
        ind_hue_lbl = QLabel("Individual Hue")
        self.high_ind_hue_spinbox = QSpinBox()
        self.high_ind_hue_spinbox.setMaximum(360)
        self.high_ind_hue_spinbox.setValue(360)


        self.low_ind_lightness_spinbox = QSpinBox()
        ind_lightness_lbl = QLabel("Individual Lightness")
        self.high_ind_lightness_spinbox = QSpinBox()
        self.high_ind_lightness_spinbox.setValue(100)

        self.low_ind_saturation_spinbox = QSpinBox()
        ind_saturation_lbl = QLabel("Individual Saturation")
        self.high_ind_saturation_spinbox = QSpinBox()
        self.high_ind_saturation_spinbox.setValue(100)

        vbox_ind.addWidget(self.low_ind_hue_spinbox, 0, 1)
        vbox_ind.addWidget(ind_hue_lbl, 0, 2)
        vbox_ind.addWidget(self.high_ind_hue_spinbox, 0, 3)
        vbox_ind.addWidget(self.low_ind_lightness_spinbox, 1, 1)
        vbox_ind.addWidget(ind_lightness_lbl, 1, 2)
        vbox_ind.addWidget(self.high_ind_lightness_spinbox, 1, 3)
        vbox_ind.addWidget(self.low_ind_saturation_spinbox, 2, 1)
        vbox_ind.addWidget(ind_saturation_lbl, 2, 2)
        vbox_ind.addWidget(self.high_ind_saturation_spinbox, 2, 3)


        # vbox.addStretch(1)
        filter_group_box.setLayout(vbox)
        individual_filter_group_box.setLayout(vbox_ind)


        vbox_layout = QGridLayout()

        # vbox_layout.addWidget(self.file_open_btn)
        vbox_layout.addWidget(selected_color_text, 0 , 0, 2, 0)
        vbox_layout.addWidget(filter_group_box, 2, 0)

        vbox_layout.addWidget(individual_filter_group_box, 3, 0)
        vbox_layout.addWidget(self.search_colors_btn, 4, 0)

        parameter_groupbox.setLayout(vbox_layout)

        return parameter_groupbox

    def create_input_img_groupbox(self):
        img_groupbox = QGroupBox('Image')
        vbox_layout = QVBoxLayout()

        self.input_scene = QGraphicsScene()
        self.input_view = QGraphicsView(self.input_scene)
        self.input_view.mousePressEvent = self.get_coordinate


        vbox_layout.addWidget(self.input_view)
        img_groupbox.setLayout(vbox_layout)
        return img_groupbox

    def get_coordinate(self, event):
        pos = self.input_view.mapToScene(event.pos())
        x = int(pos.x())
        y = int(pos.y())
        self.y_value = y
        # print(x, y)
        if (0<x<self.current_img.shape[1]) and (0<y<self.current_img.shape[0]):
            self.selected_color_value = self.current_img[y, x]
            selected_color_img = np.zeros((200, 1000, 3), dtype='uint8')
            selected_color_img[:, :, :] = self.selected_color_value
            # self.render_img_toQlbl(selected_color_img, self.selected_color_lbl)
            # self.call_sorting_algorithm()

    def render_img_toQlbl(self, img, place_holder):
        pixmap = QPixmap(self.numpy_to_pixmap(img))
        # w = self.main_img_group_box.width()
        # mainsmaller_pixmap = pixmap.scaled(w-5, h-10,Qt.KeepAspectRatio, Qt.FastTransformation)
        place_holder.setPixmap(pixmap)

    def numpy_to_pixmap(self, img):
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return qImg

    def create_second_img_groupbox(self):
        img_groupbox = QGroupBox('Ordinary')
        vbox_layout = QVBoxLayout()

        self.ordinary_scene = QGraphicsScene()
        self.ordinary_view = QGraphicsView(self.ordinary_scene)
        # self.ordinary_view.mousePressEvent = self.get_coordinate

        vbox_layout.addWidget(self.ordinary_view)
        img_groupbox.setLayout(vbox_layout)
        return img_groupbox

    def create_seam_img_groupbox(self):
        img_groupbox = QGroupBox('Seam-Carving')
        vbox_layout = QVBoxLayout()

        self.seam_scene = QGraphicsScene()
        self.seam_view = QGraphicsView(self.seam_scene)
        # self.ordinary_view.mousePressEvent = self.get_coordinate

        vbox_layout.addWidget(self.seam_view)
        img_groupbox.setLayout(vbox_layout)
        return img_groupbox

    def openfile_dialog(self):
        filename = QFileDialog.getOpenFileName(self, "Select Image")
        if filename[0] != '':
            print(filename)
            self.current_img_path = filename[0]
            self.current_img = np.asarray(Image.open(self.current_img_path).convert('RGB'))
            self.render_image(self.current_img, self.input_scene)

    def convert_from_pil_to_numpy(self, img):
        return np.asarray(img)

    def numpy_to_pixmap(self, img):
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return qImg

    def render_image(self, img, place_holder):
        place_holder.clear()
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap(qImg)
        place_holder.addPixmap(pixmap)

    def value_change(self):
        if self.timer_id != -1:
            self.killTimer(self.timer_id)

        self.timer_id = self.startTimer(500)

    def timerEvent(self, event):
        self.killTimer(self.timer_id)
        self.timer_id = -1
        self.call_sorting_algorithm()

    def get_filter_string(self):

        if self.colorful_rbtn.isChecked():
            return "colorful_filter"

        if self.light_rbtn.isChecked():
            return "light_filter"

        if self.dark_rbtn.isChecked():
            return "dark_filter"

        if self.tone_rbtn.isChecked():
            return "tone_filters"

        if self.muted_rbtn.isChecked():
            return "muted_filter"

    def define_category_from_position(self, y_value) -> int:
        if y_value < 75:
            return 0
        elif 75<y_value<155:
            return 1
        elif 155<y_value<239:
            return 2
        elif 239<y_value<323:
            return 3
        elif 323<y_value<406:
            return 4
        elif 406<y_value<490:
            return 5
        elif 490<y_value<576:
            return 6
        elif 576<y_value<655:
            return 7
        elif 655<y_value<740:
            return 8
        elif y_value>740:
            return 9

    def set_status_to_thread(self):
        range_list = []
        range_list.append(self.low_avg_hue_spinbox.value())
        range_list.append(self.high_avg_hue_spinbox.value())
        range_list.append(self.low_avg_lightness_spinbox.value())
        range_list.append(self.high_avg_lightness_spinbox.value())
        range_list.append(self.low_avg_saturation_spinbox.value())
        range_list.append(self.high_avg_saturation_spinbox.value())

        range_list.append(self.low_ind_hue_spinbox.value())
        range_list.append(self.high_ind_hue_spinbox.value())
        range_list.append(self.low_ind_lightness_spinbox.value())
        range_list.append(self.high_ind_lightness_spinbox.value())
        range_list.append(self.low_ind_saturation_spinbox.value())
        range_list.append(self.high_ind_saturation_spinbox.value())
        print(range_list)

        self.worker_process.range_list = range_list


    def call_sorting_algorithm(self):

        self.set_status_to_thread()
        self.worker_process.start()
        # from SortingDatabase import SortingDatabase
        # sorting_obj = SortingDatabase(filename="3000ColorValues.xlsx", c=tuple(self.selected_color_value), filter_string=self.get_filter_string())
        # sorting_obj.handle_requests()

    @QtCore.pyqtSlot(list)
    def worker_thread_complete(self, returned_list):
        print(returned_list)
        print('finished')

class WorkerThread(QThread):

    worker_thread_signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super(WorkerThread, self).__init__(parent)
        self.range_list = []

    @QtCore.pyqtSlot()
    def run(self):
        return_list = []
        # print('still in thread')
        from SortingDatabase import SortingDatabase
        # print("Searcing for Color :", self.selected_color)
        sorting_obj = SortingDatabase(filename="filters_aaded.xlsx", c=tuple((0, 0,0)), palette_filter="none", number=1, color_category="none")
        # output_result = sorting_obj.handle_requests()
        sorting_obj.find_between_range(self.range_list)
        return_list.append(1)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    screen = app.primaryScreen()
    h = screen.size().height()
    w = screen.size().width()

    gallery = WidgetGallery()
    gallery.setFixedSize(int(w - w * 0.025), int(h - h * 0.1))
    gallery.setWindowFlag(QtCore.Qt.WindowMinMaxButtonsHint)
    gallery.show()
    sys.exit(app.exec_())