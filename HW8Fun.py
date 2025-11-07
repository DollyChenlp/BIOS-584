import os
import numpy as np

def produce_trun_mean_cov(input_signal, input_type, E_val, L=25):

    N, F = input_signal.shape
    data = input_signal.reshape(N, E_val, L)  # (N, E, L)

    mask_tar  = (input_type ==  1)
    mask_ntar = (input_type == -1)

    signal_tar_mean  = np.mean(data[mask_tar],  axis=0)
    signal_ntar_mean = np.mean(data[mask_ntar], axis=0)

    signal_tar_cov  = np.array([np.cov(data[mask_tar,  i, :],  rowvar=False) for i in range(E_val)])
    signal_ntar_cov = np.array([np.cov(data[mask_ntar, i, :],  rowvar=False) for i in range(E_val)])
    signal_all_cov  = np.array([np.cov(data[:,        i, :],   rowvar=False) for i in range(E_val)])

    return [signal_tar_mean, signal_ntar_mean, signal_tar_cov, signal_ntar_cov, signal_all_cov]

def plot_trunc_mean(
    eeg_tar_mean, eeg_ntar_mean,
    subject_name, time_index, E_val, electrode_name_ls,
    y_limit=np.array([-5, 8]), fig_size=(12, 12), save_dir="K114"
):
    import matplotlib.pyplot as plt
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(4, 4, figsize=fig_size)
    axes = axes.flatten()
    for i in range(E_val):
        ax = axes[i]
        ax.plot(time_index, eeg_tar_mean[i],  color='red',  label='Target')
        ax.plot(time_index, eeg_ntar_mean[i], color='blue', label='Non-Target')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(electrode_name_ls[i])
        ax.set_ylim(y_limit)
        ax.legend(loc='upper right')

    plt.suptitle(f'Subject: {subject_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Mean.png"), dpi=150)
    plt.close()


def plot_trunc_cov(
    eeg_cov, cov_type, time_index, subject_name, E_val, electrode_name_ls,
    fig_size=(14, 12), save_dir="K114"
):
    import matplotlib.pyplot as plt
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(4, 4, figsize=fig_size)
    axes = axes.flatten()
    X, Y = np.meshgrid(time_index, time_index)

    for i in range(E_val):
        ax = axes[i]
        c = ax.contourf(X, Y, eeg_cov[i])
        fig.colorbar(c, ax=ax)
        ax.set_title(electrode_name_ls[i])
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Time (ms)')

    plt.suptitle(f'Subject: {subject_name} - {cov_type}')
    plt.tight_layout()

    name_map = {
        "Target": "Covariance_Target.png",
        "Non-Target": "Covariance_Non-Target.png",
        "All": "Covariance_All.png",
    }
    out_name = name_map.get(cov_type, f"Covariance_{cov_type}.png")
    plt.savefig(os.path.join(save_dir, out_name), dpi=150)
    plt.close()
