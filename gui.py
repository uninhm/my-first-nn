import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NN:
    def __init__(self, filename="version1_params.npz"):
        data = np.load(filename, allow_pickle=True)
        self.weights = [w for w in data["weights"]]
        self.biases  = [b for b in data["biases"]]

    def eval(self, x):
        a = x
        for i in range(len(self.weights)):
            a = sigmoid(self.weights[i] @ a + self.biases[i])
        return a

# =========================
# GUI
# =========================

CANVAS_SIZE = 280
GRID_SIZE = 28
CELL = CANVAS_SIZE // GRID_SIZE

class DigitGUI:
    def __init__(self, root):
        self.root = root
        root.title("MNIST Digit Recognizer")

        self.model = NN()

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
        self.canvas.grid(row=0, column=0, columnspan=4)

        self.image = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)

        ttk.Button(root, text="Predict", command=self.predict).grid(row=1, column=0)
        ttk.Button(root, text="Clear", command=self.clear).grid(row=1, column=1)
        ttk.Button(root, text="Quit", command=root.quit).grid(row=1, column=2)

        self.pred_label = ttk.Label(root, text="Prediction: ?")
        self.pred_label.grid(row=1, column=3)

    def draw(self, event):
        x, y = event.x // CELL, event.y // CELL
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            self.image[y, x] = 1.0
            self.canvas.create_rectangle(
                x * CELL, y * CELL,
                (x + 1) * CELL, (y + 1) * CELL,
                fill="white", outline=""
            )
            for i in (-1, 0, 1):
                if 0 <= x + i < GRID_SIZE:
                    self.canvas.create_rectangle(
                        x * CELL + i * CELL, y * CELL,
                        (x + 1) * CELL + i * CELL, (y + 1) * CELL,
                        fill="white", outline=""
                    )
                    self.image[y, x + i] = 1.0
                if 0 <= y + i < GRID_SIZE:
                    self.canvas.create_rectangle(
                        x * CELL, y * CELL + i * CELL,
                        (x + 1) * CELL, (y + 1) * CELL + i * CELL,
                        fill="white", outline=""
                    )
                    self.image[y + i, x] = 1.0


    def clear(self):
        self.canvas.delete("all")
        self.image[:] = 0
        self.pred_label.config(text="Prediction: ?")

    def predict(self):
        x = self.image.reshape(-1)
        probs = self.model.eval(x)
        pred = np.argmax(probs)

        self.pred_label.config(text=f"Prediction: {pred}")

        self.show_probs(probs)

    def show_probs(self, probs):
        plt.figure(figsize=(6, 4))
        plt.bar(range(10), probs)
        plt.xticks(range(10))
        plt.ylim(0, 1)
        plt.xlabel("Digit")
        plt.ylabel("Probability")
        plt.title("Prediction probabilities")
        plt.show()

# =========================
# Run
# =========================

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitGUI(root)
    root.mainloop()

