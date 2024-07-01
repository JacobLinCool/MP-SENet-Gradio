import torch
import torch.utils.data


def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):

    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    mag = torch.abs(stft_spec)
    pha = torch.angle(stft_spec)
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)

    return mag, pha, com


def mag_pha_istft(
    mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True
):
    # Magnitude Decompression
    mag = torch.pow(mag, (1.0 / compress_factor))
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(
        com,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
    )

    return wav
