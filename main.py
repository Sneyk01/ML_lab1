import ctypes
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageGrab
from PIL import ImageTk
from PIL import Image


class Pixel:
    def __init__(self, x_coord, y_coord):
        self.x = x_coord
        self.y = y_coord

    @staticmethod
    def is_object(color: (int, int, int)):
        return color == (0, 0, 0)


class PaintApp:
    def __init__(self, root):
        self.root = root
        self.canvas_width = 320
        self.canvas_height = 320
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg="white", bd=1, relief=tk.SUNKEN)
        self.canvas.pack(side=tk.LEFT, expand=False)
        self.setup_navbar()
        self.setup_tools()
        self.setup_events()
        self.prev_x = None
        self.prev_y = None

    def setup_navbar(self):
        self.navbar = tk.Menu(self.root)
        self.root.config(menu=self.navbar)

        # File menu
        self.file_menu = tk.Menu(self.navbar, tearoff=False)
        # self.navbar.add_cascade(label="File", menu=self.file_menu)
        self.navbar.add_command(label="Save Snapshot", command=self.take_snapshot)
        self.navbar.add_command(label="Save weight", command=self.save_weight)
        self.navbar.add_command(label="Load weight", command=self.load_weight)
        self.navbar.add_command(label="Load image", command=self.load_img)
        # self.file_menu.add_separator()
        self.navbar.add_command(label="Exit", command=self.root.quit)

        # Edit menu
        # self.edit_menu = tk.Menu(self.navbar, tearoff=False)
        # self.navbar.add_cascade(label="Edit", menu=self.edit_menu)
        # self.edit_menu.add_command(label="Undo", command=self.undo)

    def setup_tools(self):
        self.selected_tool = "pen"
        self.colors = ["black", "red", "green", "blue", "yellow", "orange", "purple", "white"]
        self.selected_color = self.colors[0]
        self.brush_sizes = [2, 4, 6, 8]
        self.selected_size = self.brush_sizes[1]
        self.pen_types = ["line", "round", "square", "arrow", "diamond"]
        self.selected_pen_type = self.pen_types[1]

        self.tool_frame = ttk.LabelFrame(self.root, text="Tools")
        self.tool_frame.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.Y)

        self.brush_size_label = ttk.Label(self.tool_frame, text="Brush Size:")
        self.brush_size_label.pack(side=tk.TOP, padx=5, pady=5)

        self.brush_size_combobox = ttk.Combobox(self.tool_frame, values=self.brush_sizes, state="readonly")
        self.brush_size_combobox.current(1)
        self.brush_size_combobox.pack(side=tk.TOP, padx=5, pady=5)
        self.brush_size_combobox.bind("<<ComboboxSelected>>", lambda event: self.select_size(int(self.brush_size_combobox.get())))

        self.color_label = ttk.Label(self.tool_frame, text="Color:")
        self.color_label.pack(side=tk.TOP, padx=5, pady=5)

        self.color_combobox = ttk.Combobox(self.tool_frame, values=self.colors, state="readonly")
        self.color_combobox.current(0)
        self.color_combobox.pack(side=tk.TOP, padx=5, pady=5)
        self.color_combobox.bind("<<ComboboxSelected>>", lambda event: self.select_color(self.color_combobox.get()))

        self.pen_type_label = ttk.Label(self.tool_frame, text="Pen Type:")
        self.pen_type_label.pack(side=tk.TOP, padx=5, pady=5)

        self.pen_type_combobox = ttk.Combobox(self.tool_frame, values=self.pen_types, state="readonly")
        self.pen_type_combobox.current(1)
        self.pen_type_combobox.pack(side=tk.TOP, padx=5, pady=5)
        self.pen_type_combobox.bind("<<ComboboxSelected>>", lambda event: self.select_pen_type(self.pen_type_combobox.get()))

        self.clear_button = ttk.Button(self.tool_frame, text="Clear Canvas", command=self.clear_canvas)
        self.clear_button.pack(side=tk.TOP, padx=5, pady=5)

        self.recognize_button = ttk.Button(self.tool_frame, text="Recognize img", command=self.recognize_img)
        self.recognize_button.pack(side=tk.TOP, padx=5, pady=5)

        self.save_button = ttk.Button(self.tool_frame, text="Save img", command=self.save_img)
        self.save_button.pack(side=tk.TOP, padx=5, pady=5)

        self.center_button = ttk.Button(self.tool_frame, text="Center img", command=self.center_photo)
        self.center_button.pack(side=tk.TOP, padx=5, pady=5)

        # Metrics frame
        self.metrics_frame = ttk.LabelFrame(self.root, text="Metrics")
        self.metrics_frame.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.Y)

        self.accuracy = ttk.Label(self.metrics_frame, text="Accuracy:None")
        self.accuracy.pack(side=tk.TOP, padx=5, pady=5)

        self.precision = ttk.Label(self.metrics_frame, text="Precision:None")
        self.precision.pack(side=tk.TOP, padx=5, pady=5)

        self.recall = ttk.Label(self.metrics_frame, text="Recall:None")
        self.recall.pack(side=tk.TOP, padx=5, pady=5)

        self.loss = ttk.Label(self.metrics_frame, text="Loss:None")
        self.loss.pack(side=tk.TOP, padx=5, pady=5)

        # ML frame
        self.ml_frame = ttk.LabelFrame(self.root, text="ML")
        self.ml_frame.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.Y)

        self.era_box_label = ttk.Label(self.ml_frame, text="Number of epochs:")
        self.era_box_label.pack(side=tk.TOP, padx=5, pady=5)

        self.era_box = ttk.Spinbox(self.ml_frame, from_=10, to=120)
        self.era_box.set(100)
        self.era_box.pack(side=tk.TOP, padx=5, pady=5)

        self.learning_speed_label = ttk.Label(self.ml_frame, text="Learning speed:")
        self.learning_speed_label.pack(side=tk.TOP, padx=5, pady=5)

        self.learning_speed_box = ttk.Spinbox(self.ml_frame, from_=10, to=120)
        self.learning_speed_box.set(100)
        self.learning_speed_box.pack(side=tk.TOP, padx=5, pady=5)

        self.start_ml = ttk.Button(self.ml_frame, text="Start ml", command=self.save_img)
        self.start_ml.pack(side=tk.TOP, padx=5, pady=5)

    def setup_events(self):
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.release)

    def select_size(self, size):
        self.selected_size = size

    def select_color(self, color):
        self.selected_color = color

    def select_pen_type(self, pen_type):
        self.selected_pen_type = pen_type

    def recognize_img(self):
        return

    def save_img(self):
        self.take_snapshot()

    def save_weight(self):
        return

    def load_weight(self):
        return

    def load_img(self):
        img_path = filedialog.askopenfilename()
        if img_path != "":
            global imageCanvas
            imageCanvas = ImageTk.PhotoImage(file=img_path)
            self.clear_canvas()
            item = self.canvas.create_image((3, 3), image=imageCanvas, anchor='nw')
            # self.canvas.itemconfigure(item, image=imageCanvas)

    def draw(self, event):
        if self.selected_tool == "pen":
            if self.prev_x is not None and self.prev_y is not None:
                if self.selected_pen_type == "line":
                    self.canvas.create_line(self.prev_x, self.prev_y, event.x, event.y, fill=self.selected_color,
                                            width=self.selected_size, smooth=True)
                elif self.selected_pen_type == "round":
                    x1 = event.x - self.selected_size
                    y1 = event.y - self.selected_size
                    x2 = event.x + self.selected_size
                    y2 = event.y + self.selected_size
                    self.canvas.create_oval(x1, y1, x2, y2, fill=self.selected_color, outline=self.selected_color)
                elif self.selected_pen_type == "square":
                    x1 = event.x - self.selected_size
                    y1 = event.y - self.selected_size
                    x2 = event.x + self.selected_size
                    y2 = event.y + self.selected_size
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.selected_color, outline=self.selected_color)
                elif self.selected_pen_type == "arrow":
                    x1 = event.x - self.selected_size
                    y1 = event.y - self.selected_size
                    x2 = event.x + self.selected_size
                    y2 = event.y + self.selected_size
                    self.canvas.create_polygon(x1, y1, x1, y2, event.x, y2, fill=self.selected_color,
                                               outline=self.selected_color)
                elif self.selected_pen_type == "diamond":
                    x1 = event.x - self.selected_size
                    y1 = event.y
                    x2 = event.x
                    y2 = event.y - self.selected_size
                    x3 = event.x + self.selected_size
                    y3 = event.y
                    x4 = event.x
                    y4 = event.y + self.selected_size
                    self.canvas.create_polygon(x1, y1, x2, y2, x3, y3, x4, y4, fill=self.selected_color,
                                               outline=self.selected_color)
            self.prev_x = event.x
            self.prev_y = event.y

    def release(self, event):
        self.prev_x = None
        self.prev_y = None

    def clear_canvas(self):
        self.canvas.delete("all")

    def take_snapshot(self):
        # self.canvas.postscript(file="snapshot.eps")
        # img = Image.open("snapshot.eps")
        # img.save("temp.png", 'png')
        x = root.winfo_rootx()+self.canvas.winfo_x() + 3
        y = root.winfo_rooty()+self.canvas.winfo_y() + 3
        x1 = x + self.canvas.winfo_height() - 6
        y1 = y + self.canvas.winfo_width() - 6
        ImageGrab.grab().crop((x, y, x1, y1)).resize((320, 320)).save('temp.jpg', 'JPEG')

    def center_photo(self):
        self.take_snapshot()

        global imageCanvas
        im = Image.open('temp.jpg')
        px = im.load()
        width, height = im.size

        upper_edge = Pixel(width, height)
        down_edge = Pixel(0, 0)

        for y in range(height):
            for x in range(width):
                # Search upper edge
                if Pixel.is_object(px[x, y]) and upper_edge.x > x:
                    upper_edge.x = x
                if Pixel.is_object(px[x, y]) and upper_edge.y > y:
                    upper_edge.y = y

                # Search down edge
                if Pixel.is_object(px[x, y]) and down_edge.x < x:
                    down_edge.x = x
                if Pixel.is_object(px[x, y]) and down_edge.y < y:
                    down_edge.y = y

        print(upper_edge.x, upper_edge.y)
        print(down_edge.x, down_edge.y)

        object_w = down_edge.x+1 - upper_edge.x
        object_h = down_edge.y+1 - upper_edge.y
        print(object_w, object_h)

        # Calc coord for new img
        img_center = int(320 / 2)
        new_upper_edge = Pixel(img_center - int(object_w / 2), img_center - int(object_h / 2))

        new_img = Image.new('RGB', (320, 320), color=(255, 255, 255))
        y_counter = 0
        for y in range(upper_edge.y, down_edge.y+1):
            x_counter = 0
            for x in range(upper_edge.x, down_edge.x+1):
                new_img.putpixel((new_upper_edge.x+x_counter, new_upper_edge.y+y_counter), px[x, y])
                x_counter += 1
            y_counter += 1

        new_img.save('temp_center.jpg', 'JPEG')
        new_img.resize((64, 64)).save('temp_center_resize.jpg', 'JPEG')

        imageCanvas = ImageTk.PhotoImage(file='temp_center.jpg')
        self.clear_canvas()
        item = self.canvas.create_image((3, 3), image=imageCanvas, anchor='nw')

    def undo(self):
        items = self.canvas.find_all()
        if items:
            self.canvas.delete(items[-1])


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Paint Application")
    root.resizable(width=False, height=False)
    app = PaintApp(root)
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    root.mainloop()
