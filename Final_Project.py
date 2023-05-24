import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model
import h5py
import time


class App(tk.Tk):
    
    def __init__(self):
        super().__init__()
        self.geometry('1000x470')
        self.title('Image Prediction')
        self.configure(bg='#E6E6FA')
        self.resizable(width=False, height=False)
        
        # Khai báo các widget
        self.label_0 =tk.Label(self, text = 'Diabetes Dectection Through Iris of Eye',relief=tk.FLAT, bg='#E6E6FA', font=("Arial",20,"bold") )
        self.label_0.place(x=120, y = 10)
       
        # thực hiện thay đổi màu của label
        self.colors = ['red', 'green', 'blue', 'purple']  # danh sách các màu cần thay đổi
        self.index = 0  # chỉ số của màu hiện tại trong danh sách colors
        self.change_color()

        self.canvas0 = tk.Canvas(self, width=80, height=80, bg='#E6E6FA')
        self.canvas0.place(x=20, y=10)
        self.image3 = Image.open('HCMUTE-01.png').resize((80, 80))
        img3 = ImageTk.PhotoImage(self.image3)
        self.canvas0.create_image(0,0,anchor='nw',image=img3)
        self.canvas0.image = img3

        self.canvas1 = tk.Canvas(self, width=400, height=300, bg='white',relief=tk.SUNKEN)
        self.canvas1.place(x=20, y=100)
        
        self.canvas2 = tk.Canvas(self, width=400, height=340, bg='white',relief=tk.SUNKEN)
        self.canvas2.place(x= 575, y=100)
        self.image2 = Image.open('3.png').resize((400, 340))
        img2 = ImageTk.PhotoImage(self.image2)
        self.canvas2.create_image(0,0,anchor='nw',image=img2)
        self.canvas2.image = img2
        
        self.btn_open = ttk.Button(self, text='Open', command=self.open_img)
        self.btn_open.place(x=450, y=130)
        
        self.btn_predict = ttk.Button(self, text='Predict', command=self.predict, state='disabled')
        self.btn_predict.place(x=450, y=170)
        
        self.btn_close = ttk.Button(self, text='Close', command=self.close_img, state='disabled')
        self.btn_close.place(x=450, y=210)

        self.progress_bar = ttk.Progressbar(self, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.place(x=20, y=420)
        self.progress_bar["maximum"] = 100
        self.progress_bar["value"] = 0

        self.label_1 =tk.Label(self, text = '',relief=tk.SUNKEN,bg='white' )
        self.label_1.place(x=445, y = 265,width= 20, height = 20)
        self.label1 =tk.Label(self, text = 'Diabetes',bg='#E6E6FA',font=("Bold"))
        self.label1.place(x=470, y = 260,width= 100)

        self.label_2 =tk.Label(self, text = '',relief=tk.SUNKEN,bg='white' )
        self.label_2.place(x=445, y = 305,width= 20, height = 20)
        self.label2 =tk.Label(self, text = 'Healthy',bg='#E6E6FA',font=("Bold" ))
        self.label2.place(x=470, y = 300,width= 100)

        self.image = None  # Lưu trữ ảnh hiện trại
    
    def change_color(self):
        self.label_0.config(fg=self.colors[self.index])  # thay đổi màu của chữ trong label
        self.index = (self.index + 1) % len(self.colors)  # tăng chỉ số của màu lên 1 và lặp lại từ đầu nếu quá giới hạn
        self.label_0.after(1000, self.change_color)  # thực hiện hàm change_color sau 1 giây

    def set_progress(self, value):
        self.progress_bar["value"] = value
        self.master.update_idletasks()

    # thực hiện chức năng cho open button    
    def open_img(self):
        ftypes = [ ('All files', '*jpeg *.png *.jpg')]
        filename = filedialog.askopenfilename(title='open', filetypes = ftypes)
        self.image = Image.open(filename).resize((400, 300))
        img = ImageTk.PhotoImage(self.image)
        self.canvas1.create_image(0,0,anchor='nw',image=img)
        self.canvas1.image = img
    # cập nhật cho giá trị của thanh progress bar active
        for i in range(101):
            self.progress_bar["value"] = i
            self.update()
            time.sleep(0.005)

        # Cho phép nút Predict và Close
        self.btn_predict['state'] = 'normal'
        self.btn_close['state'] = 'normal'

    # thực hiện chức năng cho predict buttom   
    def predict(self):
        if self.image is None:
            self.result_label.config(text="Please select an image first.")
            return

        # Đọc mô hình đã được huấn luyện từ tệp diabetes.h5
        with h5py.File('diabetes.h5', 'r') as f:
            model = load_model(f)

        # Chuyển đổi ảnh thành một mảng numpy và chuẩn hóa giá trị pixel
        img = np.asarray(self.image)
        img = img.astype('float32')
        img = img/255
        img = img.reshape(1, 300, 400, 3)

        # Thực hiện dự đoán và hiển thị kết quả
        pred = model.predict(img)
        predicted_class = np.argmax(pred, axis=1)

        if predicted_class == 1:
            self.image1 = Image.open('1.png').resize((400, 340))
            img1 = ImageTk.PhotoImage(self.image1)
            self.canvas2.create_image(0,0,anchor='nw',image=img1)
            self.canvas2.image = img1
            self.label_1.configure(bg="red")
            self.label_2.configure(bg="white")
        elif predicted_class == 2:
            self.image1 = Image.open('2.png').resize((400, 340))
            img1 = ImageTk.PhotoImage(self.image1)
            self.canvas2.create_image(0,0,anchor='nw',image=img1)
            self.canvas2.image = img1
            self.label_1.configure(bg="white")
            self.label_2.configure(bg="green")
    
    def close_img(self):
        # Xoá ảnh và kết quả trên canvas1 và canvas2
        self.canvas1.delete('all')
        self.canvas2.delete('all')
        self.image2 = Image.open('3.png').resize((400, 340))
        img2 = ImageTk.PhotoImage(self.image2)
        self.canvas2.create_image(0,0,anchor='nw',image=img2)
        self.canvas2.image = img2
        self.image = None
     
        for i in range(100,-1,-1):
            self.progress_bar["value"] = i
            self.update()
            time.sleep(0.001)

        self.label_1.configure(bg="white")
        self.label_2.configure(bg="white")

        # Vô hiệu hóa nút Predict và Close
        self.btn_predict['state'] = 'disabled'
        self.btn_close['state'] = 'disabled'
        
    def run(self):
        self.mainloop()

if __name__ == '__main__':
    app = App()
    app.run()