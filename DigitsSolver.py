import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

CNN = False

class DigitRecognizerApp:
    def __init__(self, model, master=None):
        self.master = master
        self.master.title("Digit Recognizer")
        self.model = model
        
        # Obszar rysowania (czarne tło)
        self.canvas = tk.Canvas(self.master, width=280, height=280, bg="black")
        self.canvas.pack()
        
        # Rysowanie na płótnie
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.update_prediction)

        # Przycisk czyszczenia
        self.clear_button = tk.Button(self.master, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)

        # Przycisk zapisu
        self.save_button = tk.Button(self.master, text="Save as PNG", command=self.save_canvas)
        self.save_button.pack(side=tk.LEFT)

        # Wyświetlanie przewidywania
        self.prediction_label = tk.Label(self.master, text="Prediction: None", font=("Arial", 16))
        self.prediction_label.pack(side=tk.LEFT)

        # Obraz do rysowania (czarne tło)
        self.image = Image.new("L", (28, 28), color="black")
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x, y = event.x, event.y
        r = 3  # Cieńszy promień pędzla (zmniejszenie wartości)
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        # Skala 1:10 dla konwersji współrzędnych
        self.draw.ellipse([x / 10 - 1, y / 10 - 1, x / 10 + 1, y / 10 + 1], fill="white")  # Skala 1:10

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), color="black")
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="Prediction: None")

    def save_canvas(self):
        filepath = "digit.png"
        self.image.save(filepath)
        messagebox.showinfo("Saved", f"Image saved as {filepath}")

    def preprocess_image(self):
        # Przekształcenie obrazu (skalowanie do 28x28 i normalizacja)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalizacja dla modelu
        ])
        if CNN == True:
            return transform(self.image).unsqueeze(0)
        else:
            return transform(self.image).view(-1, 784)

    def update_prediction(self, event=None):
        input_tensor = self.preprocess_image()
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_label = torch.argmax(output).item()
        self.prediction_label.config(text=f"Prediction: {predicted_label}")

def load_model_CNN(filepath):
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()

            # Warstwa konwolucyjna 1
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
            self.batch_norm1 = nn.BatchNorm2d(32)
            self.dropout1 = nn.Dropout2d(0.25)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPooling o rozmiarze 2x2

            # Warstwa konwolucyjna 2
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.batch_norm2 = nn.BatchNorm2d(64)
            self.dropout2 = nn.Dropout2d(0.25)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPooling o rozmiarze 2x2

            # Warstwa konwolucyjna 3
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.batch_norm3 = nn.BatchNorm2d(128)
            self.dropout3 = nn.Dropout2d(0.25)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPooling o rozmiarze 2x2

            # Warstwa w pełni połączona (FC)
            self.fc1 = nn.Linear(128 * 3 * 3, 512)  # Po trzech operacjach poolingowych obraz zostanie zmniejszony do 3x3
            self.fc2 = nn.Linear(512, 10)  # 10 klas wyjściowych (cyfry 0-9)

        def forward(self, x):
            # Warstwa konwolucyjna 1 -> BatchNorm -> ReLU -> Dropout -> MaxPool
            x = F.relu(self.batch_norm1(self.conv1(x)))
            x = self.dropout1(x)
            x = self.pool1(x)

            # Warstwa konwolucyjna 2 -> BatchNorm -> ReLU -> Dropout -> MaxPool
            x = F.relu(self.batch_norm2(self.conv2(x)))
            x = self.dropout2(x)
            x = self.pool2(x)

            # Warstwa konwolucyjna 3 -> BatchNorm -> ReLU -> Dropout -> MaxPool
            x = F.relu(self.batch_norm3(self.conv3(x)))
            x = self.dropout3(x)
            x = self.pool3(x)

            # Spłaszczanie obrazu przed warstwą w pełni połączoną
            x = x.view(-1, 128 * 3 * 3)  # Spłaszczanie: 128 filtrów 3x3

            # Warstwa w pełni połączona (FC)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    model = CNNModel()
    model.load_state_dict(torch.load(filepath, weights_only=True))
    model.eval()
    return model

def load_model_MLP(filepath):
    class MLPShallow(nn.Module):
        def __init__(self):
            super(MLPShallow, self).__init__()

            self.input_layer = nn.Linear(784, 256)
            self.batch_norm_1 = nn.BatchNorm1d(256)  # BatchNorm po warstwie wejściowej
            self.dropout_1 = nn.Dropout(0.2)  # Dropout z prawdopodobieństwem 20%

            self.hidden_layer_1 = nn.Linear(256, 256)
            self.batch_norm_2 = nn.BatchNorm1d(256)
            self.dropout_2 = nn.Dropout(0.2)

            self.hidden_layer_2 = nn.Linear(256, 128)
            self.batch_norm_3 = nn.BatchNorm1d(128)
            self.dropout_3 = nn.Dropout(0.2)

            self.hidden_layer_3 = nn.Linear(128, 64)
            self.batch_norm_4 = nn.BatchNorm1d(64)
            self.dropout_4 = nn.Dropout(0.2)

            self.hidden_layer_4 = nn.Linear(64, 32)
            self.batch_norm_5 = nn.BatchNorm1d(32)
            self.dropout_5 = nn.Dropout(0.2)

            self.hidden_layer_5 = nn.Linear(32, 16)
            self.batch_norm_6 = nn.BatchNorm1d(16)
            self.dropout_6 = nn.Dropout(0.2)

            self.output_layer = nn.Linear(16, 10)

        def forward(self, x):
            x = F.relu(self.batch_norm_1(self.input_layer(x)))
            x = self.dropout_1(x)

            x = F.relu(self.batch_norm_2(self.hidden_layer_1(x)))
            x = self.dropout_2(x)

            x = F.relu(self.batch_norm_3(self.hidden_layer_2(x)))
            x = self.dropout_3(x)

            x = F.relu(self.batch_norm_4(self.hidden_layer_3(x)))
            x = self.dropout_4(x)

            x = F.relu(self.batch_norm_5(self.hidden_layer_4(x)))
            x = self.dropout_5(x)

            x = F.relu(self.batch_norm_6(self.hidden_layer_5(x)))
            x = self.dropout_6(x)

            x = self.output_layer(x)
            return x

    model = MLPShallow()
    model.load_state_dict(torch.load(filepath, weights_only=True))
    model.eval()
    return model

if __name__ == "__main__":
    CNN = True
    model_path = "cnn.pth"  # Upewnij się, że plik modelu istnieje
    model = load_model_CNN(model_path)

    root = tk.Tk()
    app = DigitRecognizerApp(model, master=root)
    root.mainloop()