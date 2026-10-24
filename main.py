"""
Распознавание нарисованных цифр (MNIST).

Установка:
    pip install tensorflow numpy pillow

Запуск:
    python mnist_draw.py

При первом запуске модель обучится на MNIST (пара минут) и сохранится
в файл mnist_model.keras. Дальше будет загружаться сразу.

Логика превращения рисунка в картинку 28x28:
  1. Рисуем на холсте 280x280 (кисть толстая).
  2. Холст делится на сетку 28x28, размер клетки 10x10 пикселей.
  3. Клетка считается закрашенной, если максимальный пиксель в ней > 0
     (max-pooling) -> получаются "клетки", из которых собираются линии.
  4. (по желанию) Центрирование цифры в рамке 20x20, как в оригинальном MNIST.
  5. (по желанию) Сглаживание -- гауссово размытие 3x3, т.к. в MNIST края
     цифр мягкие, а после max-pooling они получаются "ступенчатыми".
"""
import os
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

MODEL_PATH = "mnist_model.keras"
CANVAS = 280          # размер холста в пикселях
GRID = 28             # размер сетки MNIST
CELL = CANVAS // GRID # 10 пикселей на клетку
BRUSH = 22            # толщина кисти


# ----------------------------------------------------------------------
# 1. Модель
# ----------------------------------------------------------------------
def build_and_train():
    from tensorflow import keras
    from tensorflow.keras import layers

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # нормализация 0..255 -> 0..1 и добавление канала
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]

    model = keras.Sequential([
        layers.Input((28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),  # вероятности для цифр 0..9
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=4, batch_size=128,
              validation_split=0.1, verbose=2)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Точность на тесте: {acc * 100:.2f}%")
    model.save(MODEL_PATH)
    return model


def load_model():
    from tensorflow import keras
    if os.path.exists(MODEL_PATH):
        return keras.models.load_model(MODEL_PATH)
    return build_and_train()


# ----------------------------------------------------------------------
# 2. Предобработка рисунка
# ----------------------------------------------------------------------
def smooth(img):
    """Сглаживание: свёртка с гауссовым ядром 3x3."""
    k = np.array([[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]], dtype="float32")
    k /= k.sum()
    p = np.pad(img, 1)
    out = np.zeros_like(img)
    for dy in range(3):
        for dx in range(3):
            out += k[dy, dx] * p[dy:dy + img.shape[0], dx:dx + img.shape[1]]
    return out


def center_digit(pil_img):
    """Вписать цифру в рамку 200x200 (т.е. 20x20 клеток) и поставить в центр."""
    arr = np.array(pil_img)
    ys, xs = np.where(arr > 0)
    if len(ys) == 0:
        return pil_img
    crop = pil_img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    w, h = crop.size
    scale = 200 / max(w, h)
    crop = crop.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                       Image.LANCZOS)
    out = Image.new("L", (CANVAS, CANVAS), 0)
    out.paste(crop, ((CANVAS - crop.size[0]) // 2,
                     (CANVAS - crop.size[1]) // 2))
    return out


def to_mnist(pil_img, use_center=True, use_smooth=True):
    """Холст 280x280 -> массив 28x28 в диапазоне 0..1."""
    if use_center:
        pil_img = center_digit(pil_img)
    arr = np.array(pil_img, dtype="float32") / 255.0

    # клетка = максимум пикселя внутри блока 10x10
    grid = arr.reshape(GRID, CELL, GRID, CELL).max(axis=(1, 3))

    if use_smooth:
        grid = smooth(grid)
        if grid.max() > 0:
            grid = grid / grid.max()
    return grid


# ----------------------------------------------------------------------
# 3. Интерфейс
# ----------------------------------------------------------------------
class App:
    def __init__(self, root, model):
        self.model = model
        self.root = root
        root.title("MNIST: нарисуй цифру")

        self.img = Image.new("L", (CANVAS, CANVAS), 0)
        self.draw = ImageDraw.Draw(self.img)
        self.last = None

        left = tk.Frame(root)
        left.grid(row=0, column=0, padx=10, pady=10)
        right = tk.Frame(root)
        right.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        self.canvas = tk.Canvas(left, width=CANVAS, height=CANVAS,
                                bg="black", cursor="cross")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.var_smooth = tk.BooleanVar(value=True)
        self.var_center = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="Сглаживание", variable=self.var_smooth,
                       command=self.predict).pack(anchor="w")
        tk.Checkbutton(left, text="Центрировать цифру", variable=self.var_center,
                       command=self.predict).pack(anchor="w")
        btns = tk.Frame(left)
        btns.pack(pady=5)
        tk.Button(btns, text="Распознать", command=self.predict).pack(side="left", padx=3)
        tk.Button(btns, text="Очистить", command=self.clear).pack(side="left", padx=3)

        tk.Label(right, text="Что видит модель (28x28):").pack()
        self.preview = tk.Label(right, bg="black")
        self.preview.pack(pady=5)

        self.result = tk.Label(right, text="—", font=("Arial", 22, "bold"))
        self.result.pack(pady=5)

        self.bars = tk.Canvas(right, width=260, height=230)
        self.bars.pack()
        self.show_preview(np.zeros((GRID, GRID)))

    # --- рисование ---
    def on_move(self, e):
        r = BRUSH // 2
        self.canvas.create_oval(e.x - r, e.y - r, e.x + r, e.y + r,
                                fill="white", outline="white")
        self.draw.ellipse((e.x - r, e.y - r, e.x + r, e.y + r), fill=255)
        if self.last:
            self.draw.line((*self.last, e.x, e.y), fill=255, width=BRUSH)
        self.last = (e.x, e.y)

    def on_release(self, _):
        self.last = None
        self.predict()

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, CANVAS, CANVAS), fill=0)
        self.result.config(text="—")
        self.bars.delete("all")
        self.show_preview(np.zeros((GRID, GRID)))

    # --- распознавание ---
    def predict(self):
        grid = to_mnist(self.img, self.var_center.get(), self.var_smooth.get())
        self.show_preview(grid)
        if grid.max() == 0:
            return
        probs = self.model.predict(grid[None, ..., None], verbose=0)[0]
        best = int(np.argmax(probs))
        self.result.config(text=f"Это {best}  ({probs[best] * 100:.1f}%)")
        self.draw_bars(probs, best)

    def show_preview(self, grid):
        im = Image.fromarray((grid * 255).astype("uint8")).resize(
            (168, 168), Image.NEAREST)
        self.tk_img = ImageTk.PhotoImage(im)
        self.preview.config(image=self.tk_img)

    def draw_bars(self, probs, best):
        self.bars.delete("all")
        for i, p in enumerate(probs):
            y = i * 22 + 4
            color = "#2e8b57" if i == best else "#9aa5b1"
            self.bars.create_text(10, y + 9, text=str(i), font=("Arial", 11, "bold"))
            self.bars.create_rectangle(25, y, 25 + 170 * p, y + 18,
                                       fill=color, outline="")
            self.bars.create_text(205, y + 9, text=f"{p * 100:.1f}%",
                                  anchor="w", font=("Arial", 10))


if __name__ == "__main__":
    model = load_model()
    root = tk.Tk()
    App(root, model)
    root.mainloop()