
import scipy
import numpy as np
from scipy.fftpack import dct
# from tools.spectral import stft, powspec
# from tools.preprocessing import pre_emphasis
# from tools.cepstral import *
# from tools.exceptions import ParameterError, ErrorMsgs
# from fbanks.gammatone_fbanks import gammatone_filter_banks
# from fbanks.mel_fbanks import mel_filter_banks
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import librosa.display
import numpy as np
import scipy.ndimage
from spafe.utils.spectral import rfft
# from .exceptions import ParameterError, ErrorMsgs
"""
Exception classes for Spafe
"""

ErrorMsgs = {
    "low_freq": "minimal frequency cannot be less than zero.",
    "high_freq":
    "maximum frequency cannot be greater than half sampling frequency.",
    "nfft": "size of the FFT must be an integer.",
    "nfilts": "number of filters must be bigger than number of cepstrums",
    "win_len_win_hop_comparison": "window's length has to be larger than the window's hop"
}


class SpafeError(Exception):
    """
    The root spafe exception class
    """


class ParameterError(SpafeError):
    """
    Exception class for mal-formed inputs
    """


def assert_function_availability(hasattr_output):
    # raise assertion error if function is not availible
    if not hasattr_output:
        raise AssertionError
def compute_stft(x, win, hop):
    """
    Compute the short time Fourrier transform of an audio signal x.

    Args:
        x   (array) : audio signal in the time domain
        win   (int) : window to be used for the STFT
        hop   (int) : hop-size

    Returns:
        X : 2d array of the STFT coefficients of x
    """
    # length of the audio signal
    sig_len = x.size

    # length of the window = fft size
    win_len = win.size

    # number of steps to take
    num_steps = (np.ceil((sig_len - win_len) / hop) + 1).astype(int)

    # init STFT coefficients
    X = np.zeros((win_len, num_steps), dtype=complex)

    # normalizing factor
    nf = np.sqrt(win_len)

    for k in range(num_steps - 1):
        d = x[k * hop:k * hop + win_len] * win
        X[:, k] = np.fft.fft(d) / nf

    # the last window may partially overlap with the signal
    d = x[num_steps * hop:]
    X[:, k] = np.fft.fft(d * win[:d.size], n=win_len) / nf
    return X
def normalize_window(win, hop):
    """
    Normalize the window according to the provided hop-size so that the STFT is
    a tight frame.

    Args:
        win   (int) : window to be used for the STFT
        hop   (int) : hop-size
    """
    N = win.size
    K = int(N / hop)
    win2 = win * win
    z = 1 * win2
    k = 1
    ind1 = N - hop
    ind2 = hop
    while (k < K):
        z[0:ind1] += win2[ind2:N]
        z[ind2:N] += win2[0:ind1]
        ind1 -= hop
        ind2 += hop
        k += 1
    win2 = win / np.sqrt(z)
    return win2

def pre_process_x(sig, fs=16000, win_type="hann", win_len=0.025, win_hop=0.01):
    """
    Prepare window and pad signal audio
    """
    # convert integer to double
    # sig = np.double(sig) / 2.**15

    # STFT parameters
    # convert win_len and win_hop from seconds to samples
    win_length = int(win_len * fs)
    hop_length = int(win_hop * fs)

    # compute window
    window = np.hanning(win_length)
    if win_type == "hamm":
        window = np.hamming(win_length)

    # normalization step to ensure that the STFT is self-inverting (or a Parseval frame)
    normalized_window = normalize_window(win=window, hop=hop_length)

    # Compute the STFT
    # zero pad to ensure that there are no partial overlap windows in the STFT computation
    sig = np.pad(sig, (window.size + hop_length, window.size + hop_length),
                 'constant',
                 constant_values=(0, 0))
    return sig, normalized_window, hop_length

def stft(sig, fs=16000, win_type="hann", win_len=0.025, win_hop=0.01):
    """
    Compute the short time Fourrier transform of an audio signal x.

    Args:
        x   (array) : audio signal in the time domain
        win   (int) : window to be used for the STFT
        hop   (int) : hop-size

    Returns:
        X : 2d array of the STFT coefficients of x
    """
    sig, normalized_window, hop_length = pre_process_x(sig,
                                                       fs=fs,
                                                       win_type=win_type,
                                                       win_len=win_len,
                                                       win_hop=win_hop)

    X = compute_stft(x=sig, win=normalized_window, hop=hop_length)
    return X, sig
def powspec(sig,
            fs=16000,
            nfft=512,
            win_type="hann",
            win_len=0.025,
            win_hop=0.01,
            dither=1):
    """
     compute the powerspectrum and frame energy of the input signal.
     basically outputs a power spectrogram

     each column represents a power spectrum for a given frame
     each row represents a frequency

     default values:
         fs = 8000Hz
         wintime = 25ms (200 samps)
         steptime = 10ms (80 samps)
         which means use 256 point fft
         hamming window

     $Header: /Users/dpwe/matlab/rastamat/RCS/powspec.m,v 1.3 2012/09/03 14:02:01 dpwe Exp dpwe $

     for fs = 8000
         NFFT = 256;
         NOVERLAP = 120;
         SAMPRATE = 8000;
         WINDOW = hamming(200);
    """
    # convert win_len and win_hop from seconds to samples
    win_length = int(win_len * fs)
    hop_length = int(win_hop * fs)
    fft_length = int(np.power(2, np.ceil(np.log2(win_len * fs))))

    # compute stft
    X, _ = stft(sig=sig,
                fs=fs,
                win_type=win_type,
                win_len=win_len,
                win_hop=win_hop)

    pow_X = np.abs(X)**2
    if dither:
        pow_X = pow_X + win_length

    e = np.log(np.sum(pow_X, axis=0))
    return pow_X, e
def zero_handling(x):
    """
    handle the issue with zero values if they are exposed to become an argument
    for any log function.

    Args:
        x (array): input vector.

    Returns:
        vector with zeros substituted with epsilon values.
    """
    return np.where(x == 0, np.finfo(float).eps, x)


def pre_emphasis(sig, pre_emph_coeff=0.97):
    """
    perform preemphasis on the input signal.

    Args:
        sig   (array) : signal to filter.
        coeff (float) : preemphasis coefficient. 0 is no filter, default is 0.95.

    Returns:
        the filtered signal.
    """
    return np.append(sig[0], sig[1:] - pre_emph_coeff * sig[:-1])
def medium_time_power_calculation(power_stft_signal, M=2):
    medium_time_power = np.zeros_like(power_stft_signal)
    power_stft_signal = np.pad(power_stft_signal, [(M, M), (0, 0)], 'constant')
    for i in range(medium_time_power.shape[0]):
        medium_time_power[i, :] = sum([
            1 / float(2 * M + 1) * power_stft_signal[i + k - M, :]
            for k in range(2 * M + 1)
        ])
    return medium_time_power


def asymmetric_lawpass_filtering(rectified_signal, lm_a=0.999, lm_b=0.5):
    floor_level = np.zeros_like(rectified_signal)
    floor_level[0, ] = 0.9 * rectified_signal[0, ]

    for m in range(floor_level.shape[0]):
        x = lm_a * floor_level[m - 1, :] + (1 - lm_a) * rectified_signal[m, :]
        y = lm_b * floor_level[m - 1, :] + (1 - lm_b) * rectified_signal[m, :]
        floor_level[m, :] = np.where(
            rectified_signal[m, ] >= floor_level[m - 1, :], x, y)
    return floor_level


def temporal_masking(rectified_signal, lam_t=0.85, myu_t=0.2):
    # rectified_signal[m, l]
    temporal_masked_signal = np.zeros_like(rectified_signal)
    online_peak_power = np.zeros_like(rectified_signal)

    temporal_masked_signal[0, :] = rectified_signal[0, ]
    online_peak_power[0, :] = rectified_signal[0, :]

    for m in range(1, rectified_signal.shape[0]):
        online_peak_power[m, :] = np.maximum(
            lam_t * online_peak_power[m - 1, :], rectified_signal[m, :])
        temporal_masked_signal[m, :] = np.where(
            rectified_signal[m, :] >= lam_t * online_peak_power[m - 1, :],
            rectified_signal[m, :], myu_t * online_peak_power[m - 1, :])

    return temporal_masked_signal


def weight_smoothing(final_output, medium_time_power, N=4, L=128):

    spectral_weight_smoothing = np.zeros_like(final_output)
    for m in range(final_output.shape[0]):
        for l in range(final_output.shape[1]):
            l_1 = max(l - N, 1)
            l_2 = min(l + N, L)
            spectral_weight_smoothing[m, l] = (1 / float(l_2 - l_1 + 1)) * \
                sum([(final_output[m, l_] / medium_time_power[m, l_])
                     for l_ in range(l_1, l_2)])
    return spectral_weight_smoothing


def mean_power_normalization(transfer_function,
                             final_output,
                             lam_myu=0.999,
                             L=80,
                             k=1):
    myu = np.zeros(shape=(transfer_function.shape[0]))
    myu[0] = 0.0001
    normalized_power = np.zeros_like(transfer_function)
    for m in range(1, transfer_function.shape[0]):
        myu[m] = lam_myu * myu[m - 1] + \
            (1 - lam_myu) / L * \
            sum([transfer_function[m, s] for s in range(0, L - 1)])
    normalized_power = k * transfer_function / myu[:, None]
    # log_normalized_power = 10*np.log(normalized_power)
    # log_normalized_power = librosa.amplitude_to_db(transfer_function,ref=myu[:, None])
    return normalized_power
    # return log_normalized_power


def medium_time_processing(power_stft_signal, nfilts=22):
    # calculate medium time power
    medium_time_power = medium_time_power_calculation(power_stft_signal)
    lower_envelope = asymmetric_lawpass_filtering(medium_time_power, 0.999,
                                                  0.5)
    subtracted_lower_envelope = medium_time_power - lower_envelope

    # half waverectification
    threshold = 0
    rectified_signal = np.where(subtracted_lower_envelope < threshold,
                                np.zeros_like(subtracted_lower_envelope),
                                subtracted_lower_envelope)

    floor_level = asymmetric_lawpass_filtering(rectified_signal)
    temporal_masked_signal = temporal_masking(rectified_signal)

    # switch excitation or non-excitation
    c = 2
    F = np.where(medium_time_power >= c * lower_envelope,
                 temporal_masked_signal, floor_level)

    # weight smoothing
    spectral_weight_smoothing = weight_smoothing(F,
                                                 medium_time_power,
                                                 L=nfilts)
    return spectral_weight_smoothing, F


def spncc(sig,
         fs=16000,
         num_ceps=80,
         pre_emph=1,
         pre_emph_coeff=0.97,
         power=2,
         win_len=0.025,
         win_hop=0.01,
         win_type="hamming",
         nfilts=100,#24,
         nfft=512,
         low_freq=None,
         high_freq=None,
         scale="constant",
         dct_type=2,
         use_energy=False,
         dither=1,
         lifter=22,
         normalize=1):
    """
    Compute the power-normalized cepstral coefficients (SPNCC features) from an audio signal.
    Args:
        sig            (array) : a mono audio signal (Nx1) from which to compute features.
        fs               (int) : the sampling frequency of the signal we are working with.
                                 Default is 16000.
        num_ceps       (float) : number of cepstra to return.
                                 Default is 13.
        pre_emph         (int) : apply pre-emphasis if 1.
                                 Default is 1.
        pre_emph_coeff (float) : apply pre-emphasis filter [1 -pre_emph] (0 = none).
                                 Default is 0.97.
        power            (int) : spectrum power.
                                 Default is 2.
        win_len        (float) : window length in sec.
                                 Default is 0.025.
        win_hop        (float) : step between successive windows in sec.
                                 Default is 0.01.
        win_type       (float) : window type to apply for the windowing.
                                 Default is "hamming".
        nfilts           (int) : the number of filters in the filterbank.
                                 Default is 40.
        nfft             (int) : number of FFT points.
                                 Default is 512.
        low_freq         (int) : lowest band edge of mel filters (Hz).
                                 Default is 0.
        high_freq        (int) : highest band edge of mel filters (Hz).
                                 Default is samplerate / 2 = 8000.
        scale           (str)  : choose if max bins amplitudes ascend, descend or are constant (=1).
                                 Default is "constant".
        dct_type         (int) : type of DCT used - 1 or 2 (or 3 for HTK or 4 for feac).
                                 Default is 2.
        use_energy       (int) : overwrite C0 with true log energy
                                 Default is 0.
        dither           (int) : 1 = add offset to spectrum as if dither noise.
                                 Default is 0.
        lifter           (int) : apply liftering if value > 0.
                                 Default is 22.
        normalize        (int) : apply normalization if 1.
                                 Default is 0.
    Returns:
        (array) : 2d array of PNCC features (num_frames x num_ceps)
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # run checks
    if low_freq < 0:
        raise ParameterError(ErrorMsgs["low_freq"])
    if high_freq > (fs / 2):
        raise ParameterError(ErrorMsgs["high_freq"])
    if nfilts < num_ceps:
        print("nfilts",nfilts)
        print("num_ceps", num_ceps)
        raise ParameterError(ErrorMsgs["nfilts"])

    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=pre_emph_coeff)

    # # -> STFT()
    # stf_trafo, _ = stft(sig, fs)
    #
    # #  -> |.|^2
    # spectrum_power = np.abs(stf_trafo)**power
    #
    #
    #
    # # -> x Filterbanks
    # mel_filter = mel_filter_banks(nfilts=nfilts,
    #                               nfft=nfft,
    #                               fs=fs,
    #                               low_freq=low_freq,
    #                               high_freq=high_freq,
    #                               scale=scale)
    #
    # P = np.dot(a=spectrum_power[:, :mel_filter.shape[1]],
    #            b=mel_filter.T)

    # # medium_time_processing
    # S, F = medium_time_processing(P, nfilts=nfilts)
    #
    # # time-freq normalization
    # T = P * S
    #=======================================7==============================================
    # S = librosa.feature.melspectrogram(sig, sr=fs, power=1, n_fft=512, hop_length=160, n_mels=80)
    # ========================================FBank1==============================================
    S = librosa.feature.melspectrogram(sig, sr=fs, power=1, n_fft=512, hop_length=160, n_mels=80)
    # V = S ** (1 / 15)
    logmelspec = librosa.amplitude_to_db(S)
    # ========================================6==============================================
    # S = librosa.feature.melspectrogram(sig, sr=fs, power=1, n_fft=512, hop_length=160, n_mels=80)
    # # -> mean power normalization
    # U = mean_power_normalization(S,S ,L=nfilts)
    # # # -> power law non linearity
    # V = U ** (1 / 3)
    # pcen_S = librosa.pcen(S,sr=fs, hop_length=160, gain=0.98, bias=2, power=0.5,
    #                       time_constant=0.4, eps=1e-06, b=None, max_size=1, ref=None, axis=- 1,
    #                       max_axis=None, zi=None, return_zf=False)
    # ==========================================MFCC2=============================================
    # mfccs = librosa.feature.mfcc(y=sig, sr=16000, n_mfcc=80)

    # ========================================4、5==============================================
    # S = librosa.feature.melspectrogram(sig, sr=fs, power=1, n_fft=512, hop_length=160, n_mels=80)
    # # -> mean power normalization
    # U = mean_power_normalization(S,S ,L=nfilts)
    # logmean = librosa.amplitude_to_db(U)
    #=======================================3==============================================
    # S = librosa.feature.melspectrogram(sig, sr=fs, power=1, n_fft=512, hop_length=160, n_mels=80)
    # # # -> mean power normalization
    # U = mean_power_normalization(S, S, L=nfilts)
    # # # -> power law non linearity
    # V = U ** (1 / 15)
    # ======================================================================================
    # # pcen_S = librosa.pcen(V).T
    #
    # # DCT(.)
    # pnccs = scipy.fftpack.dct(V)[:, :num_ceps]
    # ==========================================================================================
    # use energy for 1st features column
    if use_energy:
        pspectrum, logE = powspec(sig,
                                  fs=fs,
                                  win_len=win_len,
                                  win_hop=win_hop,
                                  dither=dither)

        # bug: pnccs[:, 0] = logE

    # liftering
    # if lifter > 0:
    #     pnccs = lifter_ceps(pnccs, lifter)
    #
    # # normalization
    # if normalize:
    #     # pnccs = cmvn(cms(pcen_S))
    #     pnccs = cmn(mfccs)
    return logmelspec