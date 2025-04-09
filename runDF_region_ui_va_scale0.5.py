"""with GUI"""
"""ori name runDF : run_demo_functions"""
""" ---autosave ver--- 檢查用,每一frame都匯處存"""
"""
圖片有水平翻轉:
image = flip(image, axis=1)  # axis=1 表示水平翻轉
"""
from sys import argv, exit
from PySide6 import QtWidgets, QtCore
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import QFile, QIODevice, QTimer, Qt, Slot, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtUiTools import QUiLoader
# import cv2
from cv2 import imshow, resize, line, rectangle, circle, putText, calcHist, cvtColor, destroyAllWindows, namedWindow, \
    setMouseCallback, imwrite
from cv2 import EVENT_MOUSEMOVE, EVENT_LBUTTONDOWN, COLOR_RGB2GRAY, COLOR_GRAY2RGB, COLOR_BGR2GRAY, \
    FONT_HERSHEY_SIMPLEX, COLOR_BGR2RGB
import numpy as np  # temp
from numpy import zeros, uint8, uint32, flip, array, array_equal, stack, frombuffer
from functools import partial
from datetime import datetime
from configparser import ConfigParser  # read setting
# import socket
from socket import socket, AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR
# import os
from os import path, getcwd, makedirs
from threading import Thread

# np image (Y,X)

# set: 下面 self.self.image_in_H, self.self.image_in_W = self.imgSHeight, self.imgSWidth # set the chart page img size

# noIRdevice = 1

channels = 3
his_width = 600
his_height = 200


class Worker(QtCore.QRunnable):
    def __init__(self, func):
        super(Worker, self).__init__()
        self.func = func

    @QtCore.Slot()
    def run(self):
        self.func()


def ensure_save_folder():
    # 获取当前目录路径
    save_folder = path.join(getcwd(), "save")

    # 检查是否存在，不存在则创建
    if not path.exists(save_folder):
        makedirs(save_folder)
        print(f"Created folder: {save_folder}")
    print(f"File will be saved in: {save_folder}")

    # # ---autosave ver---
    # save_folder = path.join(getcwd(), "autosave")
    # if not path.exists(save_folder):
    #     makedirs(save_folder)


class HisThread(QThread):
    result_ready = Signal(object)
    receive_value = Signal(object)  # 接收主线程的值

    def __init__(self):
        super().__init__()
        self.current_value = None
        self.receive_value.connect(self.update_value)  # 连接信号到槽函数
        self.move = 5
        self.init_figure()

    def init_figure(self):
        """初始化 Matplotlib 圖形和軸，只執行一次"""
        # 定义上下左右的边距
        margin_top = 20
        margin_bottom = 20
        margin_left = 25
        margin_right = 20
        self.his_height_in = his_height - margin_top - margin_bottom
        self.his_width_in = his_width - margin_left - margin_right

        # 创建一个空白黑色图像用于绘制曲线（增加边距）
        self.hist_img_i = zeros((self.his_height_in + margin_top + margin_bottom,
                                 self.his_width_in + margin_left + margin_right, 3), dtype=uint8)
        self.hist_img_i[:] = (0, 0, 0)  # 背景设为黑色

        # 定义绘图区域的大小
        plot_width = self.his_width_in
        plot_height = self.his_height_in

        # 绘制曲线图
        self.bin_width = plot_width // 256  # 每个点的间隔

        # 绘制坐标轴
        line(self.hist_img_i,
             (margin_left, margin_top + plot_height),
             (margin_left + plot_width, margin_top + plot_height),
             (255, 255, 255), 1)  # X 轴
        line(self.hist_img_i,
             (margin_left, margin_top),
             (margin_left, margin_top + plot_height),
             (255, 255, 255), 1)  # Y 轴

        # 添加刻度和标签
        for i in range(0, 257, 64):  # X 轴刻度
            x_pos = margin_left + i * self.bin_width
            line(self.hist_img_i,
                 (x_pos, margin_top + plot_height - 2),
                 (x_pos, margin_top + plot_height - 7),
                 (255, 255, 255), 1)
            putText(self.hist_img_i, str(i),
                    (x_pos - 15, margin_top + plot_height + 15),
                    FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        for j in range(50, 301, 50):  # Y 轴刻度 #0就不用了
            y_pos = margin_top + plot_height - int(j * plot_height / 300)
            line(self.hist_img_i,
                 (margin_left, y_pos),
                 (margin_left + 5, y_pos),
                 (255, 255, 255), 1)
            # putText(self.hist_img_i, str(j),
            #         (margin_left - 30, y_pos + 5),
            #         FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            #
        # 保存边距信息，供更新图像时使用
        self.margin_top = margin_top
        self.margin_left = margin_left
        self.plot_width = plot_width
        self.plot_height = plot_height

    def update_value(self, value):
        self.current_value = value
        # print('his_thread')

        image_forhis = self.current_value
        hist = calcHist([image_forhis], [0], None, [256], [0, 256])
        # print('hist.sum', hist.sum())
        histMax = hist.max()
        # print(hist.max())
        hist = (hist / histMax) * self.plot_height
        # hist = (hist / hist.max()) * max

        hist_img = self.hist_img_i.copy()
        # Draw bars for histogram
        for i in range(256):
            x1 = self.margin_left + i * self.bin_width
            y1 = self.margin_top + self.plot_height - int(hist[i][0])
            x2 = x1 + self.bin_width
            y2 = self.margin_top + self.plot_height

            # Draw rectangle for each bin
            rectangle(hist_img, (x1, y1), (x2, y2), (255, 255, 255), -1)  # Filled rectangle

        hist_img = cvtColor(hist_img, COLOR_RGB2GRAY)
        # imshow('hist_img', hist_img)
        # print(hist_img.shape)
        # 将绘制好的图像通过信号发送回主线程
        self.result_ready.emit(hist_img)


class MyClass(QtCore.QObject):
    def __init__(self, window):
        super(MyClass, self).__init__()

        config = ConfigParser()
        config.read('config.cfg')

        self.imgC_height, self.imgC_width = 1, 1  # initial
        self.color_ch = 'bColorC_RGB'  # initial, 直接跟物件同名稱，名稱通用好處理
        self.LeaveEmpty = 0  # initial 當抽取channel的時候原本其他顏色留空白，圖片會保持原始尺寸

        self.update_image_start = 0

        self.window = window

        # self.coordinate_absolute = 1 #加上margin之後為了方便一律absolute
        # 單個視窗顯示展開圖，click，顯示的座標
        # absolute = 1 => 對應到原圖的實際座標
        # absolute = 0 => 在這個切塊(原始切塊大小)的座標
        self.region_str = 'top_left'  # just initial

        # --1.  real size
        self.imgWidth, self.imgHeight = 4096, 3072  # *6.4 , *6.4
        self.imgWidthD2, self.imgHeightD2 = 4096 // 2, 3072 // 2  # *6.4 , *6.4
        # self.imgSWidth, self.imgSHeight = 640, 480
        # --2. deal in program (S-Show)
        # 目前如果input img 長寬不等於 UI object minimum size, 就會在UI顯示部正常 (後面會做 self.img_stream.setPixmap(
        #                     QPixmap.fromImage(qt_image)))
        self.imgSWidth, self.imgSHeight = 640, 480
        # --3. scale
        self.imgScaleW = self.imgSWidth / self.imgWidthD2
        self.imgScaleH = self.imgSHeight / self.imgHeightD2

        self.startAndRuning = 0

        # tabwidget
        self.current_tab_idx = 0
        self.current_tab_name = "Histogram"
        self.effects_to_tab_idx = {"Histogram": 0}

        # region size # 提取不同區域的方塊 #方塊長
        get_H_real, get_W_real = int(config.get('Settings', 'get_H_real')), int(config.get('Settings', 'get_W_real'))
        # print(type(get_H_real))
        # =>改成先使用原始尺寸 (原本在這步就resize了)
        self.get_H, self.get_W = get_H_real, get_W_real
        self.margin = int(config.get('Settings', 'margin'))  # 取框框,往內移動 (也就是最外邊會有margin) # set 10 pixel
        # print(type(self.margin))

        # 5 self.regions, region all show in window 放大幾倍
        self.scaleR = int(config.get('Settings', 'show_region_scale'))

        self.is_first_update_frame = True

        self.image_in_H, self.image_in_W = 1, 1  # set the chart page img size
        # chart
        self.size_of_chart = 200  # 曲線圖大小
        self.chartgap = 10

        self.scaleC = int(config.get('Settings', 'show_one_region_scale'))  # image in chart 縮放倍數
        self.coordinateC = (-1, -1)  # 表示尚未有值
        self.coordinateC_text = ''
        self.coordinateC_select_text = ''
        self.coordinateC_select = (-1, -1)  # 表示尚未有值

        self.coordinateR_click_text = ''
        self.coordinateR_click = (-1, -1)  # 表示尚未有值

        self.vertical_chart = zeros((2, 2), dtype=uint8)
        self.horizontal_chart = zeros((2, 2), dtype=uint8)
        self.image_with_charts = zeros((2, 2), dtype=uint8)
        self.region5_image = zeros((2, 2), dtype=uint8)

        # initial
        self.Cam_img = zeros((12582912,), dtype=uint32)

        self.set_widget()
        # connect
        self.window.treeWidget.itemClicked.connect(self.connect_dashboard_tab)
        self.connect_buttons()
        # connect?
        self.TStatus = self.window.findChild(QtWidgets.QTextBrowser, 'TextStatus')
        # text in ColorChannel/
        # self.TStatus = self.window.findChild(QtWidgets.QPlainTextEdit, 'TStatus')

        self.window.tabWidget.tabBar().setVisible(False)

        mar = self.margin  # margin
        self.regions = {  # top_left top_right bottom_left bottom_right center null(後面處理)
            "top_left": (0 + mar, 0 + mar, self.get_W + mar, self.get_H + mar),
            "top_right": (self.imgWidthD2 - self.get_W - mar, 0 + mar, self.imgWidthD2 - mar, self.get_H + mar),
            "bottom_left": (0 + mar, self.imgHeightD2 - self.get_H - mar, self.get_W + mar, self.imgHeightD2 - mar),
            "bottom_right": (
                self.imgWidthD2 - self.get_W - mar, self.imgHeightD2 - self.get_H - mar,
                self.imgWidthD2 - mar, self.imgHeightD2 - mar),
            "center": (
                self.imgWidthD2 // 2 - self.get_W // 2, self.imgHeightD2 // 2 - self.get_H // 2,
                self.imgWidthD2 // 2 + self.get_W // 2, self.imgHeightD2 // 2 + self.get_H // 2
            )
        }
        # print(self.regions)

        self.extracted_regions = {}
        self.list_regions = [['TL', "top_left"], ['TR', 'top_right'], ['C', 'center'], ['BL', 'bottom_left'],
                             ['BR', 'bottom_right']]
        # about read or save file
        ensure_save_folder()  # ready save folder
        self.cansave_region5, self.cansave_img, self.canread_Cam_img = 0, 0, 1

        self.channel_bgr = None
        self.image_his = zeros((2, 2), dtype=uint8)
        self.update_image_value = 0
        # 初始化子线程
        self.his_thread = HisThread()
        self.his_thread.result_ready.connect(self.process_result)  # connect to histogram thread, get signal
        self.his_thread.start()

        self.tcpServer_ready = 0

        # initial
        self.on_but_clicked_color_channel(self.color_ch)

    def setup_label_and_layout(self, parent_widget_name, Qlabel_name, align):
        parent_widget = self.window.findChild(QWidget, parent_widget_name)
        qlabel = QLabel(parent_widget)
        qlabel.setObjectName(Qlabel_name)
        qlabel.setAlignment(align)
        layout = QVBoxLayout(parent_widget)
        layout.addWidget(qlabel)
        parent_widget.setLayout(layout)
        return qlabel

    def set_widget(self):
        """設置LWIR QLabel和stream的layout"""
        self.img_stream = self.setup_label_and_layout("w_forStream", "img_stream", Qt.AlignCenter)

        """設置histogram QLabel和stream的layout"""
        self.stm_his_widget = self.setup_label_and_layout("w_forHis", "stm_his_widget", Qt.AlignLeft)

    def on_but_clicked_LeaveEmpty(self):
        print(f"self.LeaveEmpty:{self.LeaveEmpty}")
        if self.LeaveEmpty:
            self.button_leaveEmpty.setStyleSheet("")
        else:
            # apply LeaveEmpty
            # print('self.LeaveEmpty=', self.LeaveEmpty, 'set but false')
            self.button_leaveEmpty.setStyleSheet("background-color: gray; color: white;")

        self.LeaveEmpty = 1 - self.LeaveEmpty  # 0變1 1變0

        print(f"now self.LeaveEmpty:{self.LeaveEmpty}")

    def on_but_clicked_color_channel(self, clicked_button_id):
        # print('clicked', clicked_button_id)
        self.color_ch = clicked_button_id
        # 遍历颜色通道按钮
        for button_id in self.color_channel_buttons:
            button = self.window.findChild(QtWidgets.QPushButton, button_id)
            if button:
                if button_id == clicked_button_id:
                    button.setStyleSheet("background-color: #935116; color: white;")
                    # setStyleSheet('QPushButton {background-color: #A3C1DA; color: red;}')

                    # button.setStyleSheet("background-color: yellow;")
                    # button.setEnabled(False)
                else:
                    # 其他按钮恢复默认状态
                    button.setStyleSheet("")
                    # button.setEnabled(True)

    def connect_buttons(self):  # but connect
        # 第一組 button
        button_ids = ['B_stop', 'B_start',
                      'B_saveImg', 'B_saveRaw', 'B_saveROI']
        button_callbacks = [
            self.on_but_clicked_Stop,
            self.on_but_clicked_Start,
            self.on_but_clicked_SaveImg,
            partial(self.do_SaveRaw, 1),
            self.on_but_clicked_SaveROI
        ]
        for button_id, callback in zip(button_ids, button_callbacks):
            button = self.window.findChild(QtWidgets.QPushButton, button_id)
            if button:
                button.clicked.connect(callback)

        button_id = 'B_leaveEmpty'
        self.window.findChild(QtWidgets.QPushButton, button_id)
        self.button_leaveEmpty = self.window.findChild(QtWidgets.QPushButton, button_id)
        self.button_leaveEmpty.clicked.connect(self.on_but_clicked_LeaveEmpty)

        # 第二組 button
        self.color_channel_buttons = ['bColorC_RGB', 'bColorC_R', 'bColorC_G1', 'bColorC_G2', 'bColorC_B']
        # 绑定颜色按钮的点击事件
        for button_id in self.color_channel_buttons:
            button = self.window.findChild(QtWidgets.QPushButton, button_id)
            if button:
                button.clicked.connect(partial(self.on_but_clicked_color_channel, button_id))

    @Slot()
    def connect_dashboard_tab(self, position, column):
        # get item name from the tree on the left bar
        item_name = position.text(column)
        # print(self.effects_to_tab_idx)
        if item_name in self.effects_to_tab_idx:
            # print(item_name)
            self.window.tabWidget.setCurrentIndex(self.effects_to_tab_idx[item_name])
            self.current_tab_idx = self.effects_to_tab_idx[item_name]
            self.current_tab_name = item_name

    def on_but_clicked_Start(self):
        # SHOW STREAM
        # print("on_but_clicked_Start")
        self.startAndRuning = 1
        self.update_image_start = 1
        # self.on_but_clicked_color_channel('bColorC_RGB')  # set defaut mode - rgb

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(40)  # (ms) 目前這支要>=30 #可以100

        self.tcpServer_init()

    def on_but_clicked_Stop(self):
        print("Stop or Close ...")
        # print(f'ori self.startAndRuning:{self.startAndRuning}')
        # if self.startAndRuning == 1:
        self.update_image_start = 0
        self.tcpServer_stop()
        self.tcpServer_ready = 0

        # self.timer.stop()
        self.startAndRuning = 0
        self.img_stream.setPixmap(QPixmap())
        self.stm_his_widget.setPixmap(QPixmap())

        # 不需要，因為可能會需要跟之前依樣
        # self.on_but_clicked_color_channel('bColorC_RGB')  # 復原 # !!! 這一步函示名稱最好修改
        destroyAllWindows()
        print("... Stop or Close done")

    def on_but_clicked_SaveImg(self):  # save, 要水平翻轉
        bgr = self.channel_bgr
        # 改成更簡單的辨認方式
        # target_shape = (3072, 4096)
        # all_shapes_match = all(channel.shape == target_shape for channel in bgr)
        # if all_shapes_match:
        while not self.cansave_img:
            pass
        # print("SaveImg and save")
        img = stack((bgr[0], bgr[1], bgr[2]), axis=-1).astype(uint8)
        now = datetime.now()
        formatted_time = now.strftime('%Y%m%d-%H%M%S')
        fname = f"./save/{formatted_time}_RGB.tiff"
        imwrite(fname, img)

        print("RGB IMG saved: ", fname)
        self.TStatus.setPlainText("RGB IMG saved: " + fname)

    def do_SaveRaw(self, mode):
        print("Save Raw ...")
        while not self.canread_Cam_img:
            pass
        # self.canread_Cam_img

        now = datetime.now()
        formatted_time = now.strftime('%Y%m%d-%H%M%S')
        fdir = './save/'
        # ---autosave ver---
        # if mode == 1:
        #     fdir = './save/'
        # else:
        #     fdir = './autosave/'
        fname = fdir + f'{formatted_time}_raw.bin'

        text = "raw IMG saved: " + fname
        self.Cam_img.tofile(fname)
        print('... ' + text)
        if mode == 1:
            self.TStatus.setPlainText(text)

    def on_but_clicked_SaveROI(self):
        self.TStatus.setPlainText("Processing ROI")

        while not self.cansave_region5:
            pass
        now = datetime.now()
        formatted_time = now.strftime('%Y%m%d-%H%M%S')
        color = self.color_ch.replace("bColorC_", '')
        fname1 = f"./save/{formatted_time}_ROI_{color}"
        for i, plc in enumerate(self.list_regions):
            # print(plc[0], plc[1])
            fname2 = f"_{i + 1}_{plc[0]}.tiff"
            fname = fname1 + fname2
            # print(fname)
            imwrite(fname, self.extracted_regions[plc[1]])

        # print("can save")
        self.TStatus.setPlainText("5 region save: " + fname1)
        # .text_edit.setPlainText("This is a test text for QPlainTextEdit!")

    def process_result(self, result):
        # print('process_result')
        # 处理子线程返回的结果
        self.image_his = result

    def region_deal(self, extracted_regions):
        win_H, win_W = self.get_H * 3, self.get_W * 3
        combined_image = zeros((win_H, win_W), dtype=uint8)

        # 放置每個方塊
        combined_image[0:self.get_H, 0:self.get_W] = extracted_regions['top_left']  # 左上角
        # imshow("test", extracted_regions['top_left'])  # test
        # print(extracted_regions['top_right'].shape)
        combined_image[0:self.get_H, self.get_W * 2:self.get_W * 3] = extracted_regions['top_right']  # 右上角
        combined_image[self.get_H * 2:self.get_H * 3, 0:self.get_W] = extracted_regions[
            'bottom_left']  # 左下角
        combined_image[self.get_H * 2:self.get_H * 3, self.get_W * 2:self.get_W * 3] = extracted_regions[
            'bottom_right']  # 右下角
        combined_image[self.get_H:self.get_H * 2, self.get_W:self.get_W * 2] = extracted_regions[
            'center']  # 正中
        # print("----")
        # print(combined_image.shape)
        # print((int(win_H * self.scaleR), int(win_W * self.scaleR)))

        self.region5_image = resize(combined_image, (int(win_H * self.scaleR), int(win_W * self.scaleR)))

        namedWindow('ROI')  # windows B
        setMouseCallback("ROI", self.click_coord_img_region_choose)  # windows B
        imshow("ROI", self.region5_image)  # windows B
        # print(self.region5_image.shape)
        # print("乘", self.region5_image.shape[0] * self.region5_image.shape[1])
        # print(self.get_H, self.get_W, self.scaleR)
        # print("乘", self.get_H * self.get_W * self.scaleR* self.scaleR*3*3)
        # # 以上會相等

        # 銜接到 image_to_chart
        image_to_chart = extracted_regions[self.region_str]
        # image with chart / image_to_chart

        self.image_in_H = image_to_chart.shape[0] * self.scaleC
        self.image_in_W = image_to_chart.shape[1] * self.scaleC
        if self.is_first_update_frame:
            self.get_canvas()  # 调用另一个函数
            self.is_first_update_frame = False  # 设置标志为 False，后续不再调用 get_canvas

        image_to_chart = resize(image_to_chart, (self.image_in_H, self.image_in_W))
        self.image_with_charts[:self.image_in_H, :self.image_in_W] = image_to_chart  # 原图放置在左上角

        display_image = self.image_with_charts.copy()
        # display_image = resize(display_image,
        #                            (self.imgC_height, self.imgC_width))
        namedWindow("One Region with Charts")  # windows A
        setMouseCallback("One Region with Charts", self.mouse_coord_img_chart)  # windows A

        x_value, y_value = self.coordinateC_select[0], self.coordinateC_select[1]

        listA = [[self.image_in_W + 20, self.image_in_H + 40],
                 [self.image_in_W + 20, self.image_in_H + 60],
                 [self.image_in_W + 20, self.image_in_H + 80],
                 [self.image_in_W + 20, self.image_in_H + 100]]
        # print(listA)
        listA = [[x for x in A] for A in listA]
        # print(listA)

        # (Union[ndarray, Any], str, Sequence[int], int, float, Sequence[float], int)
        putText(display_image, "Mouse:", listA[0], FONT_HERSHEY_SIMPLEX, 0.4, [255], 1)
        putText(display_image, self.coordinateC_text, listA[1],
                FONT_HERSHEY_SIMPLEX, 0.4, [255], 1)
        putText(display_image, "Select:", listA[2],
                FONT_HERSHEY_SIMPLEX, 0.4, [255], 1)
        putText(display_image, self.coordinateC_select_text, listA[3],
                FONT_HERSHEY_SIMPLEX, 0.4, [255], 1)

        if x_value == -1:  # 沒值
            vertical_line = array([-1])
            horizontal_line = array([-1])
        else:  # 有值
            vertical_line = image_to_chart[:, x_value]  # x=320 的纵向数据
            horizontal_line = image_to_chart[y_value, :]  # y=240 的横向数据
            circle(display_image, (x_value, y_value), 5, [0], -1)
            circle(display_image, (x_value, y_value), 2, [255], -1)

            # 垂直水平線
            line(display_image, (x_value, 0), (x_value, self.image_in_W), [255], 1)
            line(display_image, (0, y_value), (self.image_in_H, y_value), [255], 1)

        self.draw_chart(vertical_line, horizontal_line)
        imshow("One Region with Charts", display_image)  # windows A
        # waitKey(1)

    def LeaveEmpty_do(self):  # self.LeaveEmpty==1
        get_zero_ori_size = zeros((self.imgHeight, self.imgWidth), dtype=uint8)
        img_keepSize = get_zero_ori_size  # 3072, 4096

        get_img_channel = self.channel_img_dict[self.color_ch]

        if self.color_ch == 'bColorC_B':
            img_keepSize[1::2, 1::2] = get_img_channel
            img_keepSize_rgb = np.stack((img_keepSize, get_zero_ori_size, get_zero_ori_size),
                                        axis=-1).astype(np.uint8)
        elif self.color_ch == 'bColorC_G1':
            img_keepSize[1::2, ::2] = get_img_channel
            img_keepSize_rgb = np.stack((get_zero_ori_size, img_keepSize, get_zero_ori_size),
                                        axis=-1).astype(np.uint8)
        elif self.color_ch == 'bColorC_G2':
            img_keepSize[::2, 1::2] = get_img_channel
            img_keepSize_rgb = np.stack((get_zero_ori_size, img_keepSize, get_zero_ori_size),
                                        axis=-1).astype(np.uint8)

        elif self.color_ch == 'bColorC_R':
            img_keepSize[::2, ::2] = get_img_channel
            img_keepSize_rgb = np.stack((get_zero_ori_size, get_zero_ori_size, img_keepSize),
                                        axis=-1).astype(np.uint8)
        # imwrite(f'./save/{self.color_ch}.tiff', img_keepSize_rgb)

        # 'bColorC_G1': green1_img_keepSize,
        # 'bColorC_G2': green2_img_keepSize,
        # 'bColorC_B': blue_img_keepSize,
        # 'bColorC_R': red_img_keepSize
        return img_keepSize_rgb

    def update_image(self):  # /upI/
        # print("update_image")
        if self.update_image_start:
            # try:
            # print("update_image")
            # temp take dif img
            while not self.canread_Cam_img:
                # print("waiting for img")
                pass

            # === deal image ===
            image = self.Cam_img.reshape((self.imgHeight, self.imgWidth))
            image = flip(image, axis=1)  # 水平翻轉

            # -- 原始影像 ARGB 32bit --
            # image = image.astype(uint8)

            # 3072, 4096
            red_channel = (image >> 16) & 0xFF  # 取出 Red 通道
            green_channel = (image >> 8) & 0xFF  # 取出 Green 通道
            blue_channel = image & 0xFF  # 取出 Blue 通道
            # print("1:", blue_channel.shape)
            self.cansave_img = 0
            self.channel_bgr = [blue_channel, green_channel, red_channel]
            self.cansave_img = 1

            red_img = red_channel[::2, ::2]  # (1536, 2048) # separate color channel
            green1_img = green_channel[1::2, ::2]
            green2_img = green_channel[::2, 1::2]
            blue_img = blue_channel[1::2, 1::2]

            self.channel_img_dict = {  # self.LeaveEmpty==0
                'bColorC_B': blue_img,
                'bColorC_G1': green1_img,
                'bColorC_G2': green2_img,
                'bColorC_R': red_img
            }

            if self.color_ch == 'bColorC_RGB':
                image_now = stack((blue_channel, green_channel, red_channel), axis=-1).astype(uint8)
                image_now = resize(image_now, (self.imgWidthD2, self.imgHeightD2))
                image_gray = cvtColor(image_now, COLOR_BGR2GRAY)
                image_resize = resize(image_now, (self.imgSWidth, self.imgSHeight))  # 直接用城跟de-bayer 之後依樣的尺寸 (長寬變成原本的一半)
                # -- ver 原始影像 ARGB 32bit -- (???)
                image_rgb = cvtColor(image_resize, COLOR_BGR2RGB)
                self.update_image_value = image_gray  # 計算灰度直方圖
            else:
                # print(f"else")
                channel = self.channel_img_dict[self.color_ch]
                image_now = channel.astype(uint8)
                image_resize = resize(image_now, (self.imgSWidth, self.imgSHeight))
                # -- ver de-bayer後的8bit影像 --
                self.update_image_value = image_now  # 計算灰度直方圖
                if self.LeaveEmpty:
                    image_rgb_get = self.LeaveEmpty_do()  # rgb image # (3072, 4096, 3)
                    image_resize = resize(image_rgb_get,
                                          (self.imgSWidth, self.imgSHeight))  # 直接用城跟de-bayer 之後依樣的尺寸 (長寬變成原本的一半)
                # -- ver de-bayer後的8bit影像 --
                image_rgb = cvtColor(image_resize, COLOR_BGR2RGB)

            # print('~', self.update_image_value.shape)

            self.his_thread.receive_value.emit(self.update_image_value)  # 送去處理histogram

            image_rgb_his = cvtColor(self.image_his, COLOR_BGR2RGB)

            bytes_per_line = channels * self.imgSWidth
            bytes_per_line_his = channels * his_width

            # print("region:", self.get_W, self.get_H)
            # print("region:", self.imgSWidth, self.imgSHeight)

            # print(self.regions)

            image_draww = image_rgb.copy()

            self.cansave_region5 = 0
            self.extracted_regions = {}
            for name, (x1, y1, x2, y2) in self.regions.items():
                # print("shape:", len(image_now.shape))
                if len(image_now.shape) == 3:  # apply color 之後都會是3
                    self.extracted_regions[name] = cvtColor(image_now, COLOR_RGB2GRAY)[y1:y2, x1:x2]  # 截取區域
                else:
                    self.extracted_regions[name] = image_now[y1:y2, x1:x2]  # 截取區域
                color = (255, 0, 0) if name == "center" else (0, 255, 0)  # 中心框為藍色，其餘為綠色
                # print(self.extracted_regions[name].shape)
                # 在原圖畫上取出region的框框
                # 畫畫這邊, 要根據scale resize
                x1d, y1d = x1 * self.imgScaleW, y1 * self.imgScaleH
                x2d, y2d = x2 * self.imgScaleW, y2 * self.imgScaleH
                xyd = [x1d, y1d, x2d, y2d]
                xyd = [round(x) for x in xyd]
                rectangle(image_draww, (xyd[0], xyd[1]), (xyd[2], xyd[3]), color,
                          1)  # 原本 (x2,y2)都有 -1，為了讓畫的線可以看的到。現在有最外面的margin所以畫的線不會看不到，就不-1了

            self.cansave_region5 = 1
            # 創建 region null, 後面選區塊，點到空白區域的時候就顯示全黑
            self.extracted_regions['null'] = zeros((self.get_H, self.get_W), dtype=uint8)

            # put to UI
            # image_rgb_his = resize(image_rgb_his, (100, 500))
            # print("when set", image_rgb_his.shape)
            # print(image_draww.shape)
            qt_image = QImage(image_draww.data, self.imgSWidth, self.imgSHeight, bytes_per_line,
                              QImage.Format_RGB888)
            qt_image_his = QImage(image_rgb_his.data, his_width, his_height, bytes_per_line_his,
                                  QImage.Format_RGB888)
            self.img_stream.setPixmap(QPixmap.fromImage(qt_image))
            self.stm_his_widget.setPixmap(QPixmap.fromImage(qt_image_his))

            self.region_deal(self.extracted_regions)

            # except error as e:
            #     print(f"OpenCV error: {e}")
            # except Exception as e:
            #     print(f"Unexpected error: {e}")

    def get_canvas(self):  # 有圖片 跟可以展開
        now_image_in_H = self.image_in_H
        now_image_in_W = self.image_in_W
        canvas_height = now_image_in_H + self.size_of_chart + self.chartgap
        canvas_width = now_image_in_W + self.size_of_chart + self.chartgap
        # print(f"canvas_height: {canvas_height}, canvas_width: {canvas_width}")
        # 创建显示区域
        self.image_with_charts = zeros((canvas_height, canvas_width),
                                       dtype=uint8)  # 比原图大，留空白区域放图表

        # 绘制纵向数值曲线图
        self.vertical_chart = zeros((now_image_in_H, self.size_of_chart), dtype=uint8)  # 高度等于原图，宽度为 200
        self.horizontal_chart = zeros((self.size_of_chart, now_image_in_W), dtype=uint8)  # 宽度等于原图，高度为 200

        # 添加纵轴单位线
        for value in [0, 100, 200, 255]:
            y_position = self.size_of_chart - int(value * self.size_of_chart / 255)
            line(self.vertical_chart, (y_position, 0), (y_position, now_image_in_H), [128], 1)  # 单位线
            putText(self.vertical_chart, str(value), (y_position - 15, 20), FONT_HERSHEY_SIMPLEX, 0.5, [128],
                    1)
        # 添加横轴单位线
        for value in [0, 100, 200, 255]:
            y_position = self.size_of_chart - int(value * self.size_of_chart / 255)
            line(self.horizontal_chart, (0, y_position), (now_image_in_W, y_position), [128], 1)  # 单位线
            putText(self.horizontal_chart, str(value), (5, y_position - 5), FONT_HERSHEY_SIMPLEX, 0.5, [128],
                    1)
        self.image_with_charts[:now_image_in_H,
        now_image_in_W + 10:now_image_in_W + 10 + self.size_of_chart] = self.vertical_chart
        self.image_with_charts[now_image_in_H + self.chartgap:now_image_in_H + self.chartgap + self.size_of_chart,
        :now_image_in_W] = self.horizontal_chart

    def draw_chart(self, vertical_line, horizontal_line):  # 畫chart的部分 (曲線圖)
        # print("Drawing chart")
        now_image_in_H = self.image_in_H
        now_image_in_W = self.image_in_W
        vertical_chart_draw = self.vertical_chart.copy()
        horizontal_chart_draw = self.horizontal_chart.copy()
        nodraw = array([-1])
        if not array_equal(nodraw, vertical_line):  # == 代表清空
            for i in range(now_image_in_H - 1):
                # 在纵向图中绘制平滑曲线
                y1 = int(vertical_line[i] * self.size_of_chart / 255)
                y2 = int(vertical_line[i + 1] * self.size_of_chart / 255)
                line(vertical_chart_draw, (self.size_of_chart - y1, i), (self.size_of_chart - y2, i + 1), [255], 1)

            for i in range(now_image_in_W - 1):
                # 在横向图中绘制平滑曲线
                x1 = int(horizontal_line[i] * self.size_of_chart / 255)
                x2 = int(horizontal_line[i + 1] * self.size_of_chart / 255)
                line(horizontal_chart_draw, (i, self.size_of_chart - x1), (i + 1, self.size_of_chart - x2), [255],
                     1)
        # else:
        #     print("clean char")

        self.image_with_charts[:now_image_in_H,
        now_image_in_W + 10:now_image_in_W + 10 + self.size_of_chart] = vertical_chart_draw
        self.image_with_charts[now_image_in_H + self.chartgap:now_image_in_H + self.chartgap + self.size_of_chart,
        :now_image_in_W] = horizontal_chart_draw

    def coordinate_absolute_update(self, x, y, move_click):
        # print("\n* in func coordinate_absolute_update")
        x += self.regions[self.region_str][0]
        y += self.regions[self.region_str][1]

        if move_click == 0:  # move_click=0 -> move ; 1 -> click
            self.coordinateC_text = f"X: {int(x * 2)}, Y: {int(y * 2)}"  # *2: 因為圖片是原本的1/2
        else:
            self.coordinateC_select_text = f"X: {int(x * 2)}, Y: {int(y * 2)}"

        return x, y

    def mouse_coord_img_chart(self, event, x1, y1, flags, param):  # 就是要這些input,不動 #在image with chart 的img 選的座標

        # print(f'x1={x1}, y1={y1}')
        # print(f'x1={x1}, y1={y1},self.image_in_W={self.image_in_W},self.scaleC={self.scaleC}')
        # image_coo = resize(self.image_with_charts, (self.image_in_W , self.image_in_H ))
        if x1 < self.image_in_W and y1 < self.image_in_H and self.region_str != 'null':
            now_x1, now_y1 = (x1 / self.scaleC, y1 / self.scaleC)  # 從顯示的縮放比例復原到原始尺寸
            if event == EVENT_MOUSEMOVE:
                now_x1, now_y1 = self.coordinate_absolute_update(now_x1, now_y1, 0)
                self.coordinateC = (x1, y1)
                self.coordinateC_text = f"X: {int(now_x1 * 2)}, Y: {int(now_y1 * 2)}"
            if event == EVENT_LBUTTONDOWN:
                # print(f'click: x1={x1}, y1={y1}')
                now_x1, now_y1 = self.coordinate_absolute_update(now_x1, now_y1, 1)
                self.coordinateC_select = (x1, y1)
                # print(f"image with chart, Mouse clicked at X: {now_x1}, Y: {now_y1}")
                self.coordinateC_select_text = f"X: {int(now_x1 * 2)}, Y: {int(now_y1 * 2)}"

    def click_coord_img_region_choose(self, event, x2, y2, flags, param):  # 就是要這些input,不動 # choose which region
        image_coo = self.region5_image
        if event == EVENT_LBUTTONDOWN:
            # 清空目前
            # self.coordinateC_text = ""
            # self.coordinateC_select_text = ""
            # -- 如果目前座標跟畫圖要清空 --
            # self.coordinateC_select = (-1,-1)
            # clean_line = array([-1])
            # self.draw_chart(clean_line, clean_line)

            self.coordinateR_click = (x2, y2)
            # print(f"region, Mouse clicked at X: {x2}, Y: {y2}")
            if len(image_coo.shape) == 2:  # 灰度图
                value = image_coo[y2, x2]
                self.coordinateR_click_text = f"X: {x2}, Y: {y2}, {value}"
            else:  # 彩色图
                b, g, r = image_coo[y2, x2]
                self.coordinateR_click_text = f"X: {x2}, Y: {y2}, B: {b}, G: {g}, R: {r}"

            y, x = y2 / self.scaleR, x2 / self.scaleR
            if 0 <= y < self.get_H and 0 <= x < self.get_W:
                self.region_str = 'top_left'
            elif 0 <= y < self.get_H and self.get_W * 2 <= x < self.get_W * 3:
                self.region_str = 'top_right'
            elif self.get_H * 2 <= y < self.get_H * 3 and 0 <= x < self.get_W:
                self.region_str = 'bottom_left'
            elif self.get_H * 2 <= y < self.get_H * 3 and self.get_W * 2 <= x < self.get_W * 3:
                self.region_str = 'bottom_right'
            elif self.get_H <= y < self.get_H * 2 and self.get_W <= x < self.get_W * 2:
                self.region_str = 'center'
            else:
                self.region_str = 'null'

            print("* when choose region, chosse:", self.region_str)
            # print("ori select x y :", self.coordinateC_select)

            self.coordinateC_text = ''
            if self.region_str != 'null' and self.coordinateC_select[0] != -1:
                now_x, now_y = self.coordinate_absolute_update(
                    self.coordinateC_select[0] / self.scaleC, self.coordinateC_select[1] / self.scaleC, 1)
                self.coordinateC_select_text = f"X: {int(now_x)}, Y: {int(now_y)}"
            else:
                self.coordinateC_select_text = ''

    def tcpServer_init(self):
        # temp
        # print("tcpServer_init")
        # HOST = '192.168.1.1'
        # PORT = 5001
        #
        # self.tcpBunch = 7300
        # # cTcpBunch = tcpBunch * 100
        #
        # self.s = socket(AF_INET, SOCK_STREAM)
        # self.s.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        # self.s.bind((HOST, PORT))
        # self.s.listen(5)
        #
        # print('server start at: %s:%s' % (HOST, PORT))
        # print('wait for connection...')

        self.tcpServer_ready = 1

        self.server_thread = Thread(target=self.tcpServer_update, daemon=True)
        self.server_thread.start()

    def tcpServer_update(self):
        # print("tcpServer_update")

        while self.tcpServer_ready:
            # # tcp real use
            # # print("accept")
            # conn, addr = self.s.accept()
            # # print('connected by ' + str(addr))
            # # writeFileName = 'C:/Users/00457/。F/UI_F/LWIR_UI/DEMO_python/takeImg/'
            # totalLength = 0
            #
            # # writeFileName +=  datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S-%f') + '.bin'
            # # writeFileName += 'img.bin'
            #
            # # f = open(writeFileName, "wb")
            # data_buffer = bytearray()
            # while True:
            #     indata = conn.recv(self.tcpBunch)
            #     # print('recved: %d bytes' %(totalLength))
            #
            #     if len(indata) == 0:  # connection closed
            #         conn.close()
            #         # print('client closed connection.')
            #         break
            #     totalLength += len(indata)
            #     # f.write(indata)
            #     data_buffer.extend(indata)
            #     # self.canread_Cam_img = 0
            #     # if totalLength == 12582912:
            #
            #     # self.canread_Cam_img = 1
            # self.canread_Cam_img = 0
            # self.Cam_img = indata
            # self.canread_Cam_img = 1
            # a = indata
            # print(type(a))
            #
            # now_img = frombuffer(data_buffer, dtype=uint32)
            now_img = np.fromfile("test_img/0328/2025-03-28--10-01-59-816709.bin", dtype=uint32)  # for test
            if len(now_img) == 12582912:  # here
                # print("settt")
                self.Cam_img = now_img
                # # ---autosave ver---
                # self.do_SaveRaw(2)

            # print(type(self.Cam_img))
            # print('recved: %d bytes' % (totalLength))
            # print('recived.')

    def tcpServer_stop(self):
        print("tcpServer stop ...")
        self.tcpServer_ready = 0
        # self.s.close() #temp
        print("... tcpServer stop done")

    def show(self):
        self.window.show()


if __name__ == "__main__":
    ui_file_path = "window.ui"
    loader = QUiLoader()
    app = QtWidgets.QApplication(argv)

    ui_file = QFile(ui_file_path)
    if not ui_file.open(QIODevice.ReadOnly):
        print(f"Cannot open {ui_file_path}: {ui_file.errorString()}")
        exit(-1)

    window = loader.load(ui_file, None)
    ui_file.close()

    if not window:
        print(loader.errorString())
        exit(-1)

    myclass = MyClass(window)
    myclass.show()

    app.aboutToQuit.connect(myclass.tcpServer_stop)
    app.aboutToQuit.connect(myclass.on_but_clicked_Stop)  # stopOrCloseWindow=1: stop

    exit(app.exec())
