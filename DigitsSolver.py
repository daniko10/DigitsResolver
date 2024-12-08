import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F




from tkinter import filedialog


class DigitRecognizerApp:
    def __init__(self, models, master=None):
        self.master = master
        self.master.title("Rozpoznawacz cyfr")
        self.master.resizable(False, False)
        self.models = models
        self.current_model_type = tk.StringVar(value="CNN")  # Domyślny model to CNN
        self.model = self.models[self.current_model_type.get()]

        # Główna ramka
        self.frame = ttk.Frame(self.master, padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Obszar rysowania
        self.canvas = tk.Canvas(self.frame, width=280, height=280, bg="black", relief=tk.RIDGE, bd=2)
        self.canvas.grid(row=0, column=0, rowspan=4, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.update_prediction)

        # Przyciski akcji
        self.clear_button = ttk.Button(self.frame, text="Wyczyść", command=self.clear_canvas)
        self.clear_button.grid(row=0, column=1, sticky="ew", padx=10, pady=5)

        self.save_button = ttk.Button(self.frame, text="Zapisz jako PNG", command=self.save_canvas)
        self.save_button.grid(row=1, column=1, sticky="ew", padx=10, pady=5)

        self.upload_button = ttk.Button(self.frame, text="Wczytaj obraz (PNG)", command=self.upload_image)
        self.upload_button.grid(row=2, column=1, sticky="ew", padx=10, pady=5)

        # Opcje wyboru modelu
        self.label_model = ttk.Label(self.frame, text="Wybierz model:", font=("Arial", 12))
        self.label_model.grid(row=3, column=1, pady=5, sticky="w")

        self.cnn_checkbox = ttk.Checkbutton(
            self.frame,
            text="CNN",
            variable=self.current_model_type,
            onvalue="CNN",
            command=self.update_model
        )
        self.cnn_checkbox.grid(row=4, column=1, sticky="w")

        self.mlp_checkbox = ttk.Checkbutton(
            self.frame,
            text="MLP",
            variable=self.current_model_type,
            onvalue="MLP",
            command=self.update_model
        )
        self.mlp_checkbox.grid(row=5, column=1, sticky="w")

        # Wyświetlanie przewidywania
        self.prediction_label = ttk.Label(self.frame, text="Prediction: None", font=("Arial", 16))
        self.prediction_label.grid(row=6, column=0, columnspan=2, pady=10)

        # Obraz do rysowania
        self.image = Image.new("L", (28, 28), color="black")
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x, y = event.x, event.y
        r = 3  # Rozmiar pędzla
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x / 10 - 1, y / 10 - 1, x / 10 + 1, y / 10 + 1], fill="white")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), color="black")
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="Prediction: None")

    def save_canvas(self):
        filepath = "digit.png"
        self.image.save(filepath)

    def preprocess_image(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        if self.current_model_type.get() == "CNN":
            return transform(self.image).unsqueeze(0)  # CNN wymaga dodatkowego wymiaru
        else:
            return transform(self.image).view(-1, 784)  # MLP wymaga wektora

    def update_prediction(self, event=None):
        input_tensor = self.preprocess_image()
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_label = torch.argmax(output).item()
        self.prediction_label.config(text=f"Rozpoznana cyfra: {predicted_label}")

    def update_model(self):
        # Aktualizacja modelu na podstawie wyboru
        self.model = self.models[self.current_model_type.get()]
        self.clear_canvas()  # Czyszczenie płótna przy zmianie modelu

    def upload_image(self):
        try:
            # Wybór pliku z obrazem (tylko PNG)
            file_path = filedialog.askopenfilename(
                title="Wybierz obraz PNG",
                filetypes=[("PNG files", "*.png")]  # Akceptujemy tylko pliki PNG
            )
            if not file_path:  # Jeśli użytkownik nie wybierze pliku
                print("Nie wybrano pliku.")
                return

            # Sprawdzenie, czy plik ma poprawne rozszerzenie (dodatkowa weryfikacja)
            if not file_path.lower().endswith(".png"):
                messagebox.showerror("Invalid File", "Tylko pliki PNG są obsługiwane.")
                return

            # Załaduj obraz jako skala szarości
            img = Image.open(file_path).convert("L")
            img = img.resize((28, 28))  # Dopasowanie do rozmiaru MNIST
            self.image = img
            self.draw = ImageDraw.Draw(self.image)

            # Wyświetlenie obrazu na płótnie
            self.canvas.delete("all")
            resized_img = img.resize((280, 280))  # Skalowanie do rozmiaru płótna
            tk_img = ImageTk.PhotoImage(resized_img)
            self.canvas.image = tk_img  # Zachowanie referencji do obrazu
            self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

            # Przeprowadzenie predykcji
            self.update_prediction()
        except Exception as e:
            messagebox.showerror("Error", f"Nie udało się załadować obrazu: {e}")


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
        def init(self):
            super(MLPShallow, self).init()
            self.input_layer = nn.Linear(28 * 28, 512)  # 784 wejścia -> 512 neuronów
            self.hidden_layer_1 = nn.Linear(512, 256)  # 512 -> 256
            self.hidden_layer_2 = nn.Linear(256, 128)  # 256 -> 128
            self.hidden_layer_3 = nn.Linear(128, 64)   # 128 -> 64
            self.output_layer = nn.Linear(64, 10)     # 64 -> 10 (klasy cyfr)

            # Regularizacja
            self.dropout1 = nn.Dropout(0.2)
            self.dropout2 = nn.Dropout(0.2)

        def forward(self, x):
            x = x.view(-1, 28 * 28)  # Spłaszczenie obrazu

            # Warstwy w pełni połączone + ReLU + dropout
            x = F.relu(self.input_layer(x))
            x = self.dropout1(x)

            x = F.relu(self.hidden_layer_1(x))
            x = self.dropout1(x)

            x = F.relu(self.hidden_layer_2(x))
            x = self.dropout2(x)

            x = F.relu(self.hidden_layer_3(x))
            x = self.dropout2(x)

            # Warstwa wyjściowa
            x = self.output_layer(x)
            return x

    model = MLPShallow()
    model.load_state_dict(torch.load(filepath, weights_only=True))
    model.eval()
    return model

if __name__ == "__main__":


    # Wczytaj modele
    cnn_model = load_model_CNN("cnn.pth")
    mlp_model = load_model_MLP("mlp_shallow.pth")

    # Słownik modeli
    models = {
        "CNN": cnn_model,
        "MLP": mlp_model
    }

    # Uruchom aplikację
    root = tk.Tk()
    app = DigitRecognizerApp(models=models, master=root)
    root.mainloop()