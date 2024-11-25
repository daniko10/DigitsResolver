import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
from torchvision import transforms
import torch

# Klasa GUI
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
        r = 1  # Cieńszy promień pędzla (zmniejszenie wartości)
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
        return transform(self.image).view(-1, 784)

    def update_prediction(self, event=None):
        input_tensor = self.preprocess_image()
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_label = torch.argmax(output, dim=1).item()
        self.prediction_label.config(text=f"Prediction: {predicted_label}")

# Wczytywanie modelu
def load_model(filepath):
    class MLPShallow(torch.nn.Module):
        def __init__(self):
            super(MLPShallow, self).__init__()
            self.input_layer = torch.nn.Linear(784, 256)
            self.hidden_layer_1 = torch.nn.Linear(256, 256)
            self.hidden_layer_2 = torch.nn.Linear(256, 128)
            self.hidden_layer_3 = torch.nn.Linear(128, 64)
            self.hidden_layer_4 = torch.nn.Linear(64, 32)
            self.hidden_layer_5 = torch.nn.Linear(32, 16)
            self.output_layer = torch.nn.Linear(16, 10)

        def forward(self, x):
            x = torch.nn.functional.relu(self.input_layer(x))
            x = torch.nn.functional.relu(self.hidden_layer_1(x))
            x = torch.nn.functional.relu(self.hidden_layer_2(x))
            x = torch.nn.functional.relu(self.hidden_layer_3(x))
            x = torch.nn.functional.relu(self.hidden_layer_4(x))
            x = torch.nn.functional.relu(self.hidden_layer_5(x))
            x = self.output_layer(x)
            return x

    model = MLPShallow()
    model.load_state_dict(torch.load(filepath, weights_only=True))
    model.eval()
    return model

if __name__ == "__main__":
    model_path = "mlp_shallow.pth"  # Upewnij się, że plik modelu istnieje
    model = load_model(model_path)

    root = tk.Tk()
    app = DigitRecognizerApp(model, master=root)
    root.mainloop()
