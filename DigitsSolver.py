import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from tkinter import filedialog, messagebox
import os
import re
import importlib.util

class DigitRecognizerApp:
    def __init__(self, master=None):
        self.master = master
        self.master.title("Rozpoznawacz cyfr")
        self.master.resizable(False, False)

        self.model = None
        self.model_path = None

        self.frame = ttk.Frame(self.master, padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.frame, width=280, height=280, bg="black", relief=tk.RIDGE, bd=2)
        self.canvas.grid(row=0, column=0, rowspan=4, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.update_prediction)

        self.clear_button = ttk.Button(self.frame, text="Wyczyść", command=self.clear_canvas)
        self.clear_button.grid(row=0, column=1, sticky="ew", padx=10, pady=5)

        self.save_button = ttk.Button(self.frame, text="Zapisz jako PNG", command=self.save_canvas)
        self.save_button.grid(row=1, column=1, sticky="ew", padx=10, pady=5)

        self.upload_button = ttk.Button(self.frame, text="Wczytaj (PNG)", command=self.upload_images_or_folders)
        self.upload_button.grid(row=2, column=1, sticky="ew", padx=10, pady=5)

        self.load_model_button = ttk.Button(self.frame, text="Wczytaj model", command=self.load_model)
        self.load_model_button.grid(row=3, column=1, sticky="ew", padx=10, pady=5)

        self.prediction_label = ttk.Label(self.frame, text="Rozpoznana cyfra: brak", font=("Arial", 16))
        self.prediction_label.grid(row=4, column=0, columnspan=2, pady=10)

        self.image = Image.new("L", (28, 28), color="black")
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x, y = event.x, event.y
        r = 3
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x / 10 - 1, y / 10 - 1, x / 10 + 1, y / 10 + 1], fill="white")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), color="black")
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="Rozpoznana cyfra: brak")

    def save_canvas(self):
        filepath = "digit.png"
        self.image.save(filepath)

    def preprocess_image(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        return transform(self.image).unsqueeze(0)

    def update_prediction(self, event=None):
        if not self.model:
            self.prediction_label.config(text="Model nie załadowany")
            return

        input_tensor = self.preprocess_image()
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_label = torch.argmax(output).item()
        self.prediction_label.config(text=f"Rozpoznana cyfra: {predicted_label}")

        return predicted_label
        
    def load_model(self):
        model_path = filedialog.askopenfilename(title="Wybierz model (.pth)", filetypes=[("PyTorch Model", "*.pth")])

        if model_path:
            try:
                self.model = load_model_with_script(model_path)
                self.model_path = model_path
                model_name = os.path.basename(model_path)
                self.load_model_button.config(text=f"Załadowano: {model_name}")
                messagebox.showinfo("Sukces", "Model wczytany pomyślnie!")
            except Exception as e:
                self.model_path = None
                self.load_model_button.config(text="Wczytaj model")
                messagebox.showerror("Błąd", f"Nie udało się wczytać modelu: {e}")

    def upload_images_or_folders(self):
        try:
            choice = messagebox.askyesno("Wybór", "Czy chcesz wybrać foldery z obrazami?\n Tak dla 'folderów', Nie dla 'plików'")

            correct_predictions = 0
            total_predictions = 0

            if choice:
                folder_path = filedialog.askdirectory(title="Wybierz folder z obrazami (oznaczony jako 0, 1, 2, ...)")
                if not folder_path:
                    print("Nie wybrano folderu.")
                    return

                folder_name = os.path.basename(folder_path)
                if not folder_name.isdigit():
                    messagebox.showerror("Błąd", f"Nazwa folderu '{folder_name}' musi być liczbą (np. 0, 1, 2).")
                    return

                true_label = int(folder_name)

                correct_predictions = 0
                total_predictions = 0

                for file in os.listdir(folder_path):
                    if not file.lower().endswith(".png"):
                        continue

                    file_path = os.path.join(folder_path, file)

                    img = Image.open(file_path).convert("L")
                    img = img.resize((28, 28))

                    self.image = img
                    self.draw = ImageDraw.Draw(self.image)

                    prediction = self.update_prediction()

                    if prediction == true_label:
                        correct_predictions += 1
                    total_predictions += 1
            else:
                file_path = filedialog.askopenfilename(
                title="Wybierz obraz PNG",
                filetypes=[("PNG files", "*.png")]
                )
                if not file_path:
                    print("Nie wybrano pliku.")
                    return

                if not file_path.lower().endswith(".png"):
                    messagebox.showerror("Invalid File", "Tylko pliki PNG są obsługiwane.")
                    return

                img = Image.open(file_path).convert("L")
                img = img.resize((28, 28))
                self.image = img
                self.draw = ImageDraw.Draw(self.image)

                self.canvas.delete("all")
                resized_img = img.resize((280, 280))
                tk_img = ImageTk.PhotoImage(resized_img)
                self.canvas.image = tk_img
                self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

                prediction = self.update_prediction()

            if total_predictions > 0:
                accuracy = (correct_predictions / total_predictions) * 100
                messagebox.showinfo("Wynik", f"Dokładność modelu: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")

        except Exception as e:
            messagebox.showerror("Error", f"Nie udało się przetworzyć obrazów: {e}")


def load_model_with_script(model_path):
    model_dir, model_file = os.path.split(model_path)
    model_name, _ = os.path.splitext(model_file)
    loader_script = os.path.join(model_dir, f"{model_name}_loader.py")

    if not os.path.exists(loader_script):
        raise FileNotFoundError(f"Nie znaleziono skryptu ładowania: {loader_script}")

    spec = importlib.util.spec_from_file_location("model_loader", loader_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "load_model"):
        raise AttributeError(f"Skrypt {loader_script} musi mieć funkcję 'load_model'.")

    return module.load_model(model_path)


if __name__ == "__main__":

    root = tk.Tk()
    app = DigitRecognizerApp(master=root)
    root.mainloop()
