from tkinter import *
from tkinter.messagebox import *
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import numpy as np
import io
from PIL import EpsImagePlugin
import keras
import math
import cv2
from scipy.ndimage.measurements import center_of_mass
import os

class Paint(Frame):
    def __init__(self, parent, model):
        Frame.__init__(self, parent)
        self.parent = parent
        self.brush_size = 10
        self.brush_color = "black"
        self.color = "black"
        self.setUI()
        self.model = model

    def draw(self, event):
        self.canv.create_oval(event.x - self.brush_size,
                              event.y - self.brush_size,
                              event.x + self.brush_size,
                              event.y + self.brush_size,
                              fill=self.color, outline=self.color)

    def set_brush_size(self, new_size):
        self.brush_size = new_size

    def __get_image(self):
        ps = self.canv.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        return img

    def __fill_bg(self):
        img_np = np.full((self.canv.winfo_height(), self.canv.winfo_width()), 255)
        img = Image.fromarray(img_np)
        self.image_on_canvas = ImageTk.PhotoImage(img)
        self.image_id = self.canv.create_image(0, 0, anchor=NW, image=self.image_on_canvas)
        self.canv.itemconfig(self.image_id)

    def getBestShift(self, img):
        cy, cx = center_of_mass(img)

        rows, cols = img.shape
        shiftx = np.round(cols / 2.0 - cx).astype(int)
        shifty = np.round(rows / 2.0 - cy).astype(int)

        return shiftx, shifty

    def shift(self, img, sx, sy):
        rows, cols = img.shape
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        shifted = cv2.warpAffine(img, M, (cols, rows))
        return shifted

    def predict_label(self):
        img = self.__get_image()
        tmp_path = './tmp_image.bmp'
        img.save(tmp_path)
        img = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        gray = 255 - img
        # применяем пороговую обработку

        # удаляем нулевые строки и столбцы
        while np.sum(gray[0]) == 0:
            gray = gray[1:]
        while np.sum(gray[:, 0]) == 0:
            gray = np.delete(gray, 0, 1)
        while np.sum(gray[-1]) == 0:
            gray = gray[:-1]
        while np.sum(gray[:, -1]) == 0:
            gray = np.delete(gray, -1, 1)
        rows, cols = gray.shape

        # изменяем размер, чтобы помещалось в box 20x20 пикселей
        if rows > cols:
            factor = 20.0 / rows
            rows = 20
            cols = int(round(cols * factor))
            gray = cv2.resize(gray, (cols, rows))
        else:
            factor = 20.0 / cols
            cols = 20
            rows = int(round(rows * factor))
            gray = cv2.resize(gray, (cols, rows))

        # расширяем до размера 28x28
        colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
        rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
        gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

        # сдвигаем центр масс
        shiftx, shifty = self.getBestShift(gray)
        shifted = self.shift(gray, shiftx, shifty)
        gray = shifted

        img = gray / 255.0
        img = np.array(img).reshape(-1, 28, 28, 1)
        answer = np.argmax(self.model.predict(img, verbose=0))
        print(self.model.predict(img, verbose=0))
        self.pred_lab.config(text=f'Предсказанное число = {answer}')


    #def predict_label(self):
    #    img = self.__get_image()
    #    img = img.resize((28, 28), Image.BICUBIC)
    #    img_np = np.array(img, dtype='float64')
    #    if len(img_np.shape) > 2:
    #        img_np = np.mean(img_np, axis=2)
    #    img = np.expand_dims(img_np, axis=0)
    #    answer = np.argmax(self.model.predict(img, verbose=0))
    #    print(self.model.predict(img, verbose=0))
    #    self.pred_lab.config(text=f'Предсказанное число = {answer}')

    def load_image(self):
        filetypes = (
            ('All files', '*.*'),
        )
        filename = fd.askopenfilename(
            title='Выбор изображения',
            initialdir='/',
            filetypes=filetypes)

        try:
            img = Image.open(filename)
            img = img.resize((self.canv.winfo_width(), self.canv.winfo_height()), Image.BICUBIC)
            img_np = np.asarray(img, dtype='float64')
            if len(img_np.shape) > 2:
                img_np = np.mean(img_np, axis=2)

            img = Image.fromarray(img_np)
            self.image_on_canvas = ImageTk.PhotoImage(img)
            self.image_id = self.canv.create_image(0, 0, anchor=NW, image=self.image_on_canvas)
            self.canv.itemconfig(self.image_id)

        except BaseException as e:
            print(e)
            showerror(
                title='Ошибка',
                message='Проблема с файлом'
            )

    def invert_image(self):
        img = self.__get_image()
        img = img.resize((self.canv.winfo_width(), self.canv.winfo_height()), Image.BICUBIC)
        img_np = np.array(img, dtype='float64')
        img_np = 255 - img_np
        if len(img_np.shape) > 2:
            img_np = np.mean(img_np, axis=2)
        img = Image.fromarray(img_np)
        self.image_on_canvas = ImageTk.PhotoImage(img)
        self.image_id = self.canv.create_image(0, 0, anchor=NW, image=self.image_on_canvas)
        self.canv.itemconfig(self.image_id)

    def setUI(self):
        # Устанавливаем название окна
        self.parent.title("MNIST CLASSIFIER")
        # Размещаем активные элементы на родительском окне
        self.pack(fill=BOTH, expand=1)

        self.columnconfigure(6, weight=1)
        self.rowconfigure(2, weight=1)

        # Создаем холст с чёрным фоном
        self.canv = Canvas(self, bg="white")

        # Приклепляем канвас методом grid. Он будет находится в 3м ряду, первой колонке,
        # и будет занимать 7 колонок, задаем отступы по X и Y в 5 пикселей, и
        # заставляем растягиваться при растягивании всего окна

        self.canv.grid(row=2, column=0, columnspan=5, padx=5, pady=5, sticky=N)

        # задаем реакцию холста на нажатие левой кнопки мыши
        self.canv.bind("<B1-Motion>", self.draw)

        self.__fill_bg()

        # Создаем метку для кнопок изменения размера кисти
        size_lab = Label(self, text="Размер кисти: ")
        size_lab.grid(row=1, column=0, padx=5)
        two_btn = Button(self, text="2x", width=15, command=lambda: self.set_brush_size(2))
        two_btn.grid(row=1, column=1)

        five_btn = Button(self, text="5x", width=15, command=lambda: self.set_brush_size(5))
        five_btn.grid(row=1, column=2)

        seven_btn = Button(self, text="7x", width=15, command=lambda: self.set_brush_size(7))
        seven_btn.grid(row=1, column=3)

        ten_btn = Button(self, text="10x", width=15, command=lambda: self.set_brush_size(10))
        ten_btn.grid(row=1, column=4)

        twenty_btn = Button(self, text="20x", width=15, command=lambda: self.set_brush_size(20))
        twenty_btn.grid(row=1, column=5)

        fifty_btn = Button(self, text="50x", width=15, command=lambda: self.set_brush_size(50))
        fifty_btn.grid(row=1, column=6, sticky=W)

        clear_btn = Button(self, text="Очистить", width=15, command=self.__fill_bg)
        clear_btn.grid(row=0, column=1, sticky=W)

        load_btn = Button(self, text="Загрузить", width=15, command=self.load_image)
        load_btn.grid(row=0, column=2, sticky=W)

        inv_btn = Button(self, text="Инвертировать", width=15, command=self.invert_image)
        inv_btn.grid(row=0, column=3, sticky=W)

        pred_btn = Button(self, text="Предсказать", width=15, command=self.predict_label)
        pred_btn.grid(row=2, column=5, sticky=W)

        self.pred_lab = Label(self, text="Предсказанное число: ")
        self.pred_lab.grid(row=3, column=5, padx=5)


def main():
    # here your path to GhostScript
    EpsImagePlugin.gs_windows_binary = r'C:\Program Files\gs\gs10.02.0\bin\gswin64c.exe'
    model = keras.models.load_model('./model_conv.h5')
    root = Tk()
    root.geometry("800x300+300+300")
    root.resizable(False, False)
    app = Paint(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()