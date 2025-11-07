import os
import numpy as np
from scipy import io as sio
from self_py_fun.HW8Fun import produce_trun_mean_cov, plot_trunc_mean, plot_trunc_cov

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
SAVE_DIR = os.path.join(PROJECT_DIR, "K114")

print("PROJECT_DIR =", PROJECT_DIR)
print("DATA_DIR    =", DATA_DIR, "exists:", os.path.exists(DATA_DIR))
print("DATA_DIR list:", os.listdir(DATA_DIR))

# auto-pick the K114 .mat file if present
cands = [p for p in os.listdir(DATA_DIR) if p.lower().endswith(".mat")]
if cands:
    # prefer a file containing 'k114'
    k114 = [p for p in cands if "k114" in p.lower()]
    mat_name = k114[0] if k114 else cands[0]
print("USING mat_name:", mat_name)

subject_name = "K114"
mat_name = "K114_001_BCI_TRN_Truncated_Data_0.5_6.mat"
E_val = 16
L = 25
time_index = np.linspace(0, 1000, L)
electrode_name_ls = [f"Ch{i+1}" for i in range(E_val)]

os.makedirs(SAVE_DIR, exist_ok=True)

mat_path = os.path.join(DATA_DIR, mat_name)
print("mat_path:", mat_path)
if not os.path.exists(mat_path):
    raise FileNotFoundError(f"Cannot find file: {mat_path}")

eeg_trunc_obj = sio.loadmat(mat_path)
eeg_trunc_signal = eeg_trunc_obj["Signal"]
eeg_trunc_type = np.squeeze(eeg_trunc_obj["Type"])

signal_tar_mean, signal_ntar_mean, signal_tar_cov, signal_ntar_cov, signal_all_cov = (
    produce_trun_mean_cov(eeg_trunc_signal, eeg_trunc_type, E_val=E_val, L=L)
)

plot_trunc_mean(
    signal_tar_mean,
    signal_ntar_mean,
    subject_name,
    time_index,
    E_val,
    electrode_name_ls,
    save_dir=SAVE_DIR,
)
plot_trunc_cov(
    signal_tar_cov,
    "Target",
    time_index,
    subject_name,
    E_val,
    electrode_name_ls,
    save_dir=SAVE_DIR,
)
plot_trunc_cov(
    signal_ntar_cov,
    "Non-Target",
    time_index,
    subject_name,
    E_val,
    electrode_name_ls,
    save_dir=SAVE_DIR,
)
plot_trunc_cov(
    signal_all_cov,
    "All",
    time_index,
    subject_name,
    E_val,
    electrode_name_ls,
    save_dir=SAVE_DIR,
)

print("Done! Figures saved in:", SAVE_DIR)
