import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def cwt(coloumncount, rowcount, a, da, dt, f0, y):
    w0 = 2*np.pi* f0
    Ndata = len(y)
    pi = np.pi
    db = (Ndata - 1) * dt / coloumncount
    cwtre = np.zeros((coloumncount, rowcount))
    cwtim = np.zeros((coloumncount, rowcount))
    cwt = np.zeros((coloumncount, rowcount))
    db = (Ndata-1)* dt / coloumncount
    for i in range(rowcount):
        b = 0.0
        for j in range(coloumncount):
            t = 0.0
            cwtre_sum = 0.0
            cwtim_sum = 0.0
            for k in range(Ndata):
                rem = (1 / np.sqrt(a)) * (1 / np.power(pi, 0.25)) * np.exp(-((t - b) / a) ** 2 / 2.0) * np.cos(w0 * (t - b) / a)
                imm = (1 / np.sqrt(a)) * (-1 / np.power(pi, 0.25)) * np.exp(-((t - b) / a) ** 2 / 2.0) * np.sin(w0 * (t - b) / a)
                cwtre_sum += y[k] * rem
                cwtim_sum += y[k] * imm
                t += dt
            cwtre[j, i] = cwtre_sum
            cwtim[j, i] = cwtim_sum
            cwt[j, i] = np.sqrt(cwtre[j, i] ** 2 + cwtim[j, i] ** 2)
            b += db
        a += da
    return cwt.T

def plot_signal(data):
    fig_data = go.Figure()
    fig_data.add_trace(go.Scatter(x=data[0], y=data[1], mode='lines', name='Signal'))
    fig_data.update_layout(
        title="Signal vs Time",
        xaxis_title="Time (s)",
        yaxis_title="Signal",
        template="plotly"
    )
    fig_data.show()

def plot_cwt(coloumncount, rowcount, a, da, dt, f0, y):
    X, Y = np.meshgrid(np.arange(rowcount), np.arange(coloumncount))
    Z = cwt(coloumncount, rowcount, a, da, dt, f0, y)
    fig_cwt = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig_cwt.update_layout(title='Continuous Wavelet Transform (CWT) 3D Plot',
                          scene=dict(xaxis_title='Scale',
                                     yaxis_title='Freq',
                                     zaxis_title='Magnitude'),
                          autosize=True)
    fig_cwt.show()

def open_file():
    filename = askopenfilename(filetypes=[("Text files", "*.txt")])
    if filename:
        data = pd.read_csv(filename, sep='\t', header=None)
        plot_signal(data)
        return data
    return None

def run_cwt():
    data = open_file()
    if data is not None:
        # Get user parameters from input fields
        coloumncount = int(coloumncount_var.get())
        rowcount = int(rowcount_var.get())
        a = float(a_var.get())
        da = float(da_var.get())
        dt = float(dt_var.get())
        f0 = float(f0_var.get())

        plot_cwt(coloumncount, rowcount, a, da, dt, f0, data[1])

# Tkinter GUI setup
root = Tk()
root.title("CWT 3D Plot with Tkinter")

frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0, sticky=(N, W, E, S))

# Add input fields
ttk.Label(frame, text="Sumbu Vertikal (coloumncount):").grid(row=0, column=0, padx=5, pady=5)
coloumncount_var = StringVar(value='100')
ttk.Entry(frame, textvariable=coloumncount_var).grid(row=0, column=1)

ttk.Label(frame, text="Sumbu Horizontal (rowcount):").grid(row=1, column=0, padx=5, pady=5)
rowcount_var = StringVar(value='100')
ttk.Entry(frame, textvariable=rowcount_var).grid(row=1, column=1)

ttk.Label(frame, text="Initial Value (a):").grid(row=2, column=0, padx=5, pady=5)
a_var = StringVar(value='0.0001')
ttk.Entry(frame, textvariable=a_var).grid(row=2, column=1)

ttk.Label(frame, text="Delta Skala (da):").grid(row=3, column=0, padx=5, pady=5)
da_var = StringVar(value='0.0009')
ttk.Entry(frame, textvariable=da_var).grid(row=3, column=1)

ttk.Label(frame, text="dt:").grid(row=4, column=0, padx=5, pady=5)
dt_var = StringVar(value='0.000125')
ttk.Entry(frame, textvariable=dt_var).grid(row=4, column=1)

ttk.Label(frame, text="Frekuensi (f0):").grid(row=5, column=0, padx=5, pady=5)
f0_var = StringVar(value='0.849')
ttk.Entry(frame, textvariable=f0_var).grid(row=5, column=1)

# Add buttons
ttk.Button(frame, text="Open File & Plot CWT", command=run_cwt).grid(row=6, column=0, columnspan=2, pady=10)

root.mainloop()
