# My first neural network

In this project I'm doing the "Hello World" of neural networks, which is to classify handwritten digits.
For context, I watched [3Blue1Brown's series on neural networks](https://www.3blue1brown.com/?v=neural-networks) and I implemented "version1" from what I understood.
It didn't work first try. I had to add a minus sign that I had missed, and some array initializations, but most of the logic, much to my surprise, was right.
Further versions will implement better strategies explained in [Michael Nielsen's book](https://neuralnetworksanddeeplearning.com/chap3.html).

## How to run

The project is managed with [uv](https://github.com/astral-sh/uv), so you can install it first if you want.
If you don't, the dependencies are:
- `numpy` (needed)
- `matplotlib` (gui)
- `scikit-learn` (dataset download)

If you want to run the GUI, your Python distribution must provide `tkinter`.

### Run the GUI to test the model
```bash
uv run gui.py
```
or
```bash
python3 gui.py
```

### Train the model
Warning: The resulting parameters won't be saved if you don't edit the file.
```bash
uv run download_dataset.py
uv run version1.py
```
or
```bash
python3 download_dataset.py
python3 version1.py
```
