import pandas as pd 
import numpy as np 
import plotly.graph_objects as go
import streamlit as st  
from numba import jit
from plotly.subplots import make_subplots

#Perhitungan Fungsi CWT 
@jit(nopython=True)
def cwt(coloumncount, rowcount, a, da, dt, f0, y):
    w0 = 2*np.pi* f0
    Ndata = len(y)
    pi = np.pi

    db = (Ndata - 1) * dt / coloumncount
    cwtre = np.zeros((coloumncount, rowcount))
    cwtim = np.zeros((coloumncount, rowcount))
    cwt = np.zeros((coloumncount, rowcount))

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
    return cwt


#Perhitungan Fungsi CWT
@jit(nopython=True)
def discrete_forrier_transform(N, fs, signal):
    
    #Initial Array
    X_real = np.zeros(10000)
    X_imj = np.zeros(10000)
    MagDFT = np.zeros(10000)

    #DFT
    for k in range (N):
        for n in range (N):
            X_real[k] += (signal[n])*np.cos(2*np.pi*k*n/N)
            X_imj[k] -= (signal[n])*np.sin(2*np.pi*k*n/N)

    for k in range (N):
        MagDFT[k] = np.sqrt(np.square(X_real[k])+np.square(X_imj[k]))
    return MagDFT

def create_plotly_figure(title, xaxis, yaxis, time, signal, name, mode):
    # Create a Plotly figure
    fig = go.Figure()
    color= ['blue', 'red', 'green']
    for i in range(len(signal)):
        fig.add_trace(go.Scatter(x=time, y=signal[i], mode=mode, name=name[i], line=dict(color=color[i])))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        width=800,
        height=500,
        xaxis=dict(showline=True, showgrid=True),
        yaxis=dict(showline=True, showgrid=True)
    )

    st.plotly_chart(fig)

st.title("CWT 3D Plot from Scratch Code")

st.sidebar.title("Parameter")

#Pemilihan Dataset 
select_data = st.sidebar.selectbox("Select Data", ["dataset PCG 1", "dataset PCG 2", "ETS Carotid Pulse", "ETS ECG", "ETS PCG"])
if select_data == "dataset PCG 1":
    data = pd.read_csv("dataset_PCG.txt", sep='\t', header=None)
elif select_data == "dataset PCG 2":
    data = pd.read_csv("dataset_PCG 2.txt", sep='\t', header=None)
elif select_data == "ETS Carotid Pulse":
    data = pd.read_csv("Print_16_CP_RV.txt", sep='\t', header=None)
elif select_data == "ETS ECG":
    data = pd.read_csv("Print_16_v1_ECG_RV.txt", sep='\t', header=None)
elif select_data == "ETS PCG":
    data = pd.read_csv("Print_16_v2_PCG_RV.txt", sep='\t', header=None)

st.sidebar.subheader("Skala Sumbu")
coloumncount=st.sidebar.number_input("Sumbu Vertikal", value=100)
rowcount=st.sidebar.number_input("Sumbu Horizontal", value=100)

st.sidebar.subheader("Kondisi Awal Skala")
# Input untuk nilai awal skala
a = st.sidebar.number_input("Initial Value (x 10^(-4))", value=1)
a = round(a * 1e-4, 4)  

# Input untuk Delta Skala
da = st.sidebar.number_input("Delta Skala (x 10^(-4))", value=9)
da = round(da * 1e-4, 4) 

# Input untuk nilai dt
dt = st.sidebar.number_input("dt (x 10^(-6))", value=1)
dt = round(dt * 1e-3, 6)  

# Sidebar Input untuk Frekuensi
st.sidebar.subheader("Frekuensi")

# Input untuk frekuensi awal
f0 = st.sidebar.number_input("Frekuensi Awal (Hz)", value=1000)
f0 = round(f0 * 1e-3, 3)  
w0 = 2*np.pi* f0

# Extract time (x-axis) and signal (y-axis), convert to numpy arrays
x = data[0].values
y = data[1].values

# Create a plotly scatter plot
fig_data = go.Figure()
fig_data.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Signal'))

# Customize the layout
fig_data.update_layout(
    title="Signal vs Time",
    xaxis_title="Time (s)",
    yaxis_title="Signal",
    template="plotly"
)
st.header("Plot Original Siginal Input")
st.plotly_chart(fig_data)
st.markdown("""---""")
if st.button("Start Compute"):
    #a = 0.0001
    #da = 0.0001
    #dt = 1/8000
    #f0 = 0.849

    # Membuat DataFrame untuk parameter CWT
    data_param = {
        "Parameter": ["Skala Awal (a)", "Delta Skala (da)", "dt", "Frekuensi Awal (f0)", "Frekuensi Sudut (w0)"],
        "Nilai": [f"{a:.4f}", f"{da:.4f}", f"{dt:.6f}", f"{f0:.3f} kHz", f"{w0:.6f} rad/s"]
    }
    df_param = pd.DataFrame(data_param)
    st.header("Parameter Perhitungan CWT")
    st.table(df_param)
    st.markdown("""---""")
    #Komputasi CWT
    X, Y = np.meshgrid(np.arange(coloumncount), np.arange(rowcount))
    Z = cwt(coloumncount, rowcount, a, da, dt, f0, y).T #transpose CWT

    # Membuat subplot dengan 2 kolom
    fig = make_subplots(
        rows=1, cols=2,  # 1 baris, 2 kolom
        subplot_titles=("3D Surface Plot", "2D Contour Plot"),
        specs=[[{'type': 'surface'}, {'type': 'contour'}]],  # Tipe untuk masing-masing subplot
        column_widths=[0.5, 0.5]  # Lebar relatif kolom
    )
    # Plot 3D Surface
    fig.add_trace(
        go.Surface(
            z=Z, 
            x=X, 
            y=Y, 
            colorscale='Jet',
            contours_z=dict(
                show=True,  # Menampilkan kontur di sumbu Z
                usecolormap=True,  # Menggunakan skala warna
                highlightcolor="Forestgreen",  # Warna highlight
                project_z=True  # Proyeksikan kontur ke sumbu Z
            )
        ),
        row=1, col=1  # Ditempatkan di baris 1 kolom 1
    )
    # Data untuk plot 2D Contour
    y = np.arange(coloumncount + 1)
    x = np.arange(rowcount + 1)
    # Plot 2D Contour
    fig.add_trace(
        go.Contour(
            z=Z,
            y=y,
            x=x,
            colorscale='Jet',
            contours_coloring="heatmap",
            line_smoothing=0.85,
            colorbar=dict(title="Magnitude"),
        ),
        row=1, col=2  # Ditempatkan di baris 1 kolom 2
    )
    # Memperbarui layout plot keseluruhan
    fig.update_layout(
        title_text="3D Surface and 2D Contour Plot", 
        height=600,  # Tinggi figure
        width=1000,  # Lebar figure
        scene=dict(
            xaxis_title='Scale Time (S)',  # Label sumbu x untuk 3D plot
            yaxis_title='Scale Freq (Hz)',   # Label sumbu y untuk 3D plot
            zaxis_title='Magnitude'  # Label sumbu z untuk 3D plot
        ),
        autosize=True,
        xaxis=dict(
            title='Scale Time (s)',  # Label sumbu x pada 2D Contour
        ),
        yaxis=dict(
            title='Scale Frequency (Hz)',  # Label sumbu y pada 2D Contour
        )
        )
    # Menampilkan plot
    st.header("Plot CWT")
    with st.spinner("Menghitung CWT ..."):
        st.plotly_chart(fig)
    st.markdown("""---""")
    #Perhitungan Skala
    a_awal = 0.0001  # initial scale
    time_max = np.max(data[0]) 

    # Initialize arrays for frequency and time scaling
    fk = np.zeros(rowcount)  # Frequency scale
    tk = np.zeros(coloumncount)  # Time scale

    # Calculate frequency scale fk[i]
    for i in range(rowcount):
        fk[i] = f0 / (a_awal + (i * da))

    # Calculate time scale tk[j]
    for j in range(coloumncount):
        tk[j] = (time_max / coloumncount) * j

    # Create two DataFrames: one for time and one for frequency
    df_time_scale = pd.DataFrame({
        'Time After Scaling (S)': tk})
    df_frequency_scale = pd.DataFrame({
        'Frequency After Scaling (Hz)': fk})

    # Display the tables side by side in Streamlit
    st.header("Perbandingan Skala")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Time Scale (S)")
        st.dataframe(df_time_scale)
    with col2:
        st.subheader("Frequency Scale (Hz)")
        st.dataframe(df_frequency_scale)
    st.markdown("""---""")

    st.header("Frequency Domain")
    N = len(data[1])
    MagDFT_full = discrete_forrier_transform(N, 1000, data[1].values)
    k =  np.arange (0, N, 1, dtype=int)
    n = np.arange (0, N, 1, dtype=int)
    create_plotly_figure('DFT Original Signal','Frequency (Hz)','Magnitude',k*1000/N, [MagDFT_full[k]], ['Original Signal'],'markers + lines')
    # Menampilkan footer di tengah menggunakan Markdown
    st.markdown("<br><br><br>", unsafe_allow_html=True)  # Spacer untuk memisahkan footer dari konten utama
    st.write("### Firza Aji Zunaarta (5023211002)")
    st.write("### Continuous Wavelength Transform - Analisis Sinyal Nonstasioner")
    st.write("### 2024")
    
    # Untuk memastikan footer berada di bagian bawah, Anda bisa menambahkan spacer jika perlu
    st.markdown("<br><br><br>", unsafe_allow_html=True)