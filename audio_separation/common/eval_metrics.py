import torch
import librosa
import numpy as np

HOP_LENGTH = 512
RECONSTRUCTED_SIGNAL_LENGTH = 16000
EPS = 1e-13
NAME_OF_ALL_QUALITY_METRICS = ['env', 'si_sdr', 'si_sir', 'si_sar', 'sd_sdr', 'snr', 'srr', 'si_sdri', 'sd_sdri', 'snri',
                               "si_siri", "si_sari", "sdr", "sir", "sar"]


def scale_bss_eval_helper(references, estimate, idx, compute_sir_sar=True):
    """
    Helper for scale_bss_eval to avoid infinite recursion loop.
    """
    source = references[..., idx]
    source_energy = (source ** 2).sum()

    alpha = (
        source @ estimate / source_energy
    )

    e_true = source
    e_res = estimate - e_true

    signal = (e_true ** 2).sum()
    noise = (e_res ** 2).sum()

    snr = 10 * np.log10(signal / noise)

    e_true = source * alpha
    e_res = estimate - e_true

    signal = (e_true ** 2).sum()
    noise = (e_res ** 2).sum()

    si_sdr = 10 * np.log10(signal / noise)

    srr = -10 * np.log10((1 - (1/alpha)) ** 2)
    sd_sdr = snr + 10 * np.log10(alpha ** 2)

    si_sir = np.nan
    si_sar = np.nan

    if compute_sir_sar:
        references_projection = references.T @ references

        references_onto_residual = np.dot(references.transpose(), e_res)
        b = np.linalg.solve(references_projection, references_onto_residual) + EPS

        e_interf = np.dot(references, b)
        e_artif = e_res - e_interf + EPS

        si_sir = 10 * np.log10(signal / (e_interf ** 2).sum())
        si_sar = 10 * np.log10(signal / (e_artif ** 2).sum())

    return si_sdr, si_sir, si_sar, sd_sdr, snr, srr


def scale_bss_eval(references, estimate, mixture, idx,
                   compute_sir_sar=True):
    """
    Computes metrics for references[idx] relative to the
    chosen estimates. This only works for mono audio. Each
    channel should be done independently when calling this
    function. Lovingly borrowed from Gordon Wichern and
    Jonathan Le Roux at Mitsubishi Electric Research Labs.

    This returns 9 numbers (in this order):

    - SI-SDR: Scale-invariant source-to-distortion ratio. Higher is better.
    - SI-SIR: Scale-invariant source-to-interference ratio. Higher is better.
    - SI-SAR: Scale-invariant source-to-artifact ratio. Higher is better.
    - SD-SDR: Scale-dependent source-to-distortion ratio. Higher is better.
    - SNR: Signal-to-noise ratio. Higher is better.
    - SRR: The source-to-rescaled-source ratio. This corresponds to
      a term that punishes the estimate if its scale is off relative
      to the reference. This is an unnumbered equation in [1], but
      is the term on page 2, second column, second to last line:
      ||s - alpha*s||**2. s here is factored out. Higher is better.
    - SI-SDRi: Improvement in SI-SDR over using the mixture as the estimate.
    - SD-SDRi: Improvement in SD-SDR over using the mixture as the estimate.
    - SNRi: Improvement in SNR over using the mixture as the estimate.

    References:

    [1] Le Roux, J., Wisdom, S., Erdogan, H., & Hershey, J. R.
        (2019, May). SDR–half-baked or well done?. In ICASSP 2019-2019 IEEE
        International Conference on Acoustics, Speech and Signal
        Processing (ICASSP) (pp. 626-630). IEEE.

    Args:
        references (np.ndarray): object containing the
          references data. Of shape (n_samples, n_sources).

        estimate (np.ndarray): object containing the
          estimate data. Of shape (n_samples, 1).

        mixture (np.ndarray): objct containingthe
          mixture data. Of shape (n_samples, 1).

        idx (int): Which reference to compute metrics against.

        compute_sir_sar (bool, optional): Whether or not to compute SIR/SAR
          metrics, which can be computationally expensive and may not be
          relevant for your evaluation. Defaults to True

    Returns:
        tuple: SI-SDR, SI-SIR, SI-SAR, SD-SDR, SNR, SRR, SI-SDRi, SD-SDRi, SNRi
    """
    si_sdr, si_sir, si_sar, sd_sdr, snr, srr = scale_bss_eval_helper(
        references, estimate, idx, compute_sir_sar=compute_sir_sar)
    mix_metrics = scale_bss_eval_helper(
        references, mixture, idx, compute_sir_sar=compute_sir_sar)

    si_sdri = si_sdr - mix_metrics[0]
    si_siri = si_sir - mix_metrics[1]
    si_sari = si_sar - mix_metrics[2]
    sd_sdri = sd_sdr - mix_metrics[3]
    snri = snr - mix_metrics[4]

    return si_sdr, si_sir, si_sar, sd_sdr, snr, srr, si_sdri, sd_sdri, snri, si_siri, si_sari


def evaluate_helper(references, estimates, mixture, compute_sir_sar=True):
    """
    Implements evaluation using new BSSEval metrics [1]. This computes every
    metric described in [1], including:

    - SI-SDR: Scale-invariant source-to-distortion ratio. Higher is better.
    - SI-SIR: Scale-invariant source-to-interference ratio. Higher is better.
    - SI-SAR: Scale-invariant source-to-artifact ratio. Higher is better.
    - SD-SDR: Scale-dependent source-to-distortion ratio. Higher is better.
    - SNR: Signal-to-noise ratio. Higher is better.
    - SRR: The source-to-rescaled-source ratio. This corresponds to
      a term that punishes the estimate if its scale is off relative
      to the reference. This is an unnumbered equation in [1], but
      is the term on page 2, second column, second to last line:
      ||s - alpha*s||**2. s is factored out. Higher is better.
    - SI-SDRi: Improvement in SI-SDR over using the mixture as the estimate. Higher
      is better.
    - SD-SDRi: Improvement in SD-SDR over using the mixture as the estimate. Higher
      is better.
    - SNRi: Improvement in SNR over using the mixture as the estimate. Higher is
      better.

    Note:

    If `compute_sir_sar = False`, then you'll get `np.nan` for SI-SIR and
    SI-SAR!

    References:

    [1] Le Roux, J., Wisdom, S., Erdogan, H., & Hershey, J. R.
    (2019, May). SDR–half-baked or well done?. In ICASSP 2019-2019 IEEE
    International Conference on Acoustics, Speech and Signal
    Processing (ICASSP) (pp. 626-630). IEEE.
    """
    _SISDR, _SISIR, _SISAR, _SDSDR, _SNR, _SRR, _SISDRi, _SDSDRi, _SNRi, _SISIRi, _SISARi = (
        scale_bss_eval(
            references[..., 0, :], estimates[..., 0, 0], mixture[..., 0], 0, compute_sir_sar=compute_sir_sar
        )
    )

    score = {
        'si_sdr':  _SISDR, 'si_sir': _SISIR, 'si_sar': _SISAR, 'sd_sdr': _SDSDR, 'snr': _SNR, 'srr': _SRR,
        'si_sdri': _SISDRi, 'sd_sdri': _SDSDRi, 'snri': _SNRi, "si_siri": _SISIRi, "si_sari": _SISARi,
    }
    return score


def preprocess(true_signal, estimated_signal, mixed_signal, is_mono=True):
    """
    Implements preprocess by stacking the audio_data inside each AudioSignal
    object in both self.true_sources_list and self.estimated_sources_list.

    Returns:
        tuple: Tuple containing reference and estimate arrays.
    """
    references = np.stack(
        [x for x in true_signal],
        axis=-1
    )
    references = references.transpose(1, 0, 2)  # --> time_series x num_channels x num_sources
    references -= references.mean(axis=0)

    estimates = np.stack(
        [x for x in estimated_signal],
        axis=-1
    )
    estimates = estimates.transpose(1, 0, 2)
    estimates -= estimates.mean(axis=0)

    assert len(mixed_signal) == 1, "some bug somewhere and list of mixed signals != 1"
    mixture = mixed_signal[0].transpose(1, 0) - mixed_signal[0].transpose(1, 0).mean(axis=0)
    if is_mono:
        mixture = np.mean(mixture, axis=1, keepdims=True)

    return references, estimates, mixture


def evaluate(true_signal, estimated_signal, mixed_signal, compute_sir_sar=True):
    """
    This function encapsulates the main functionality of all evaluation classes.
    It performs the following steps, some of which must be implemented in subclasses
    of EvaluationBase.
        1. Preprocesses the data somehow into numpy arrays that get passed into your
           evaluation function.
        2. Gets all possible candidates that will be evaluated in your evaluation function.
        3. For each candidate, runs the evaluation function (must be implemented in subclass).
        4. Finds the results from the best candidate.
        5. Returns a dictionary containing those results.
    Steps 1 and 3 must be implemented by the subclass while the others are implemented
    by EvaluationBase.

    Returns:
        A dictionary containing the scores for each source for the best candidate.
    """
    # INPUT
    # 1. true_signal, estimated_signal = [np.array(1 x 16000), ..., np.array(1 x 16000)]_length=1
    # 2. mixed_signal = [np.array(2 x 16000), ..., np.array(2 x 16000)]_length=1
    # OUTPUT
    # 1. references, estimates = np.array(16000 x 1 x 1)
    # 2. mixed_signal = np.array(16000 x 1)"
    references, estimates, mixture = preprocess(true_signal, estimated_signal, mixed_signal)

    scores = evaluate_helper(references, estimates, mixture, compute_sir_sar=compute_sir_sar)

    return scores


def istft(mag_l, phase_l, mag_r=None, phase_r=None):
    """
    computes inverse STFT of a monaural or a binaural spectrogram
    :param mag_l: magnitude of left binaural channel or single mono channel
    :param phase_l: phase of left binaural channel or single mono channel
    :param mag_r: magnitude of right binaural channel
    :param phase_r: phase of right binaural channel
    :return:
        signal: reconstructed waveform
    """
    spec_l_complex = mag_l * np.exp(1j * phase_l)
    spec_l_signal = librosa.istft(spec_l_complex, hop_length=HOP_LENGTH, length=RECONSTRUCTED_SIGNAL_LENGTH)
    signal = [spec_l_signal]
    if mag_r is not None:
        assert phase_r is not None
        spec_r_complex = mag_r * np.exp(1j * phase_r)
        spec_r_signal = librosa.istft(spec_r_complex, hop_length=HOP_LENGTH, length=RECONSTRUCTED_SIGNAL_LENGTH)
        signal.append(spec_r_signal)

    return signal


# implementation of all helper functions other than istft copied from https://github.com/nussl
# (had some issues while directly installing nussl on my machine)
def compute_waveform_quality(pred_n_gt_spects, eval_metrics_to_compute):
    """
    computes waveform-level quality metrics, like SI-SDR (currently works for just 1 eval process)
    :param pred_n_gt_spects: dictionary of predicted and ground-truth spectrogram magnitudes and phases
    :param eval_metrics_to_compute: waveform-level metrics to be computed
    :return:
        metrics: dictionary of waveform-level metric names and values for both 'mono' and 'monoFromMem'
    """
    mixed_bin_audio_mag = pred_n_gt_spects["mixed_bin_audio_mag"]
    mixed_bin_audio_phase = pred_n_gt_spects["mixed_bin_audio_phase"]
    gt_mono_mag = pred_n_gt_spects["gt_mono_mag"]
    gt_mono_phase = pred_n_gt_spects["gt_mono_phase"]
    pred_mono = pred_n_gt_spects["pred_mono"]
    pred_monoFromMem = pred_n_gt_spects["pred_monoFromMem"]

    # get mixed signal
    mixed_signal_lst = istft(mixed_bin_audio_mag[0, :, :, 0],
                             mixed_bin_audio_phase[0, :, :, 0],
                             mag_r=mixed_bin_audio_mag[0, :, :, 1],
                             phase_r=mixed_bin_audio_phase[0, :, :, 1])
    mixed_signal_lst = [np.array(mixed_signal_lst)]

    # get gt mono signal
    gt_mono_signal_lst = istft(gt_mono_mag[0, :, :, 0],
                               gt_mono_phase[0, :, :, 0])
    gt_mono_signal_lst = [np.array(gt_mono_signal_lst)]

    # get predicted mono signal
    pred_mono_signal_lst = istft(pred_mono[0, :, :, 0],
                                 gt_mono_phase[0, :, :, 0])
    pred_mono_signal_lst = [np.array(pred_mono_signal_lst)]

    # get predicted monoFromMem signal
    pred_monoFromMem_signal_lst = istft(pred_monoFromMem[0, :, :, 0],
                                        gt_mono_phase[0, :, :, 0])
    pred_monoFromMem_signal_lst = [np.array(pred_monoFromMem_signal_lst)]

    # compute waveform-level metrics
    mono_metrics = evaluate(gt_mono_signal_lst, pred_mono_signal_lst, mixed_signal_lst)
    monoFromMem_metrics = evaluate(gt_mono_signal_lst, pred_monoFromMem_signal_lst, mixed_signal_lst)

    metrics = {"mono": {}, "monoFromMem": {}}
    for metric in eval_metrics_to_compute:
        assert metric in NAME_OF_ALL_QUALITY_METRICS, print("doesn't support computation of this metric")
        metrics["mono"][metric] = mono_metrics[metric]
        metrics["monoFromMem"][metric] = monoFromMem_metrics[metric]

    return metrics


def STFT_L2_distance(mixed_audio, pred_binSepMasks, gt_bin_comps, pred_mono, gt_mono_comps):
    # gt binaural spectrogram
    gt_bin_mag_l = gt_bin_comps[..., 0].clone()
    gt_bin_phase_l = gt_bin_comps[..., 1].clone()
    gt_bin_realImg_l = torch.cat([(gt_bin_mag_l * torch.cos(gt_bin_phase_l)).unsqueeze(0),
                                  (gt_bin_mag_l * torch.sin(gt_bin_phase_l)).unsqueeze(0)], dim=0)
    gt_bin_realImg_l = gt_bin_realImg_l.permute(1, 0, 2, 3).unsqueeze(1).contiguous().view(gt_bin_realImg_l.size(1),
                                                                                           1, -1)

    gt_bin_mag_r = gt_bin_comps[..., 2].clone()
    gt_bin_phase_r = gt_bin_comps[..., 3].clone()
    gt_bin_realImg_r = torch.cat([(gt_bin_mag_r * torch.cos(gt_bin_phase_r)).unsqueeze(0),
                                  (gt_bin_mag_r * torch.sin(gt_bin_phase_r)).unsqueeze(0)], dim=0)
    gt_bin_realImg_r = gt_bin_realImg_r.permute(1, 0, 2, 3).unsqueeze(1).contiguous().view(gt_bin_realImg_r.size(1),
                                                                                           1, -1)

    # predicted binaural spectrogram
    mixed_audio = torch.exp(mixed_audio) - 1
    pred_bin = mixed_audio * pred_binSepMasks

    pred_bin_mag_l = pred_bin[..., 0]
    pred_bin_realImg_l = torch.cat([(pred_bin_mag_l * torch.cos(gt_bin_phase_l)).unsqueeze(0),
                                    (pred_bin_mag_l * torch.sin(gt_bin_phase_l)).unsqueeze(0)], dim=0)
    pred_bin_realImg_l = pred_bin_realImg_l.permute(1, 0, 2, 3).unsqueeze(1).contiguous().view(pred_bin_realImg_l.size(1),
                                                                                               1, -1)

    pred_bin_mag_r = pred_bin[..., 1]
    pred_bin_realImg_r = torch.cat([(pred_bin_mag_r * torch.cos(gt_bin_phase_r)).unsqueeze(0),
                                    (pred_bin_mag_r * torch.sin(gt_bin_phase_r)).unsqueeze(0)], dim=0)
    pred_bin_realImg_r = pred_bin_realImg_r.permute(1, 0, 2, 3).unsqueeze(1).contiguous().view(pred_bin_realImg_r.size(1),
                                                                                               1, -1)

    assert gt_bin_realImg_l.size() == gt_bin_realImg_r.size() == pred_bin_realImg_l.size() == pred_bin_realImg_r.size(),\
        "shapes of realImg spec. of binaural gt and preds are different, gt_ch_l: {}, pred_ch_l: {}, gt_ch_r: {}, pred_ch_r: {}"\
        .format(gt_bin_realImg_l.size(), pred_bin_realImg_l.size(), gt_bin_realImg_r.size(), pred_bin_realImg_r.size())

    bin_stft_l2_dist = torch.mean(torch.pow((gt_bin_realImg_l - pred_bin_realImg_l), 2), dim=2).cpu() +\
                       torch.mean(torch.pow((gt_bin_realImg_r - pred_bin_realImg_r), 2), dim=2).cpu()

    # gt monaural spectrogram
    gt_mono_mag = gt_mono_comps[..., 0].clone()
    gt_mono_phase = gt_mono_comps[..., 1].clone()
    gt_mono_realImg = torch.cat([(gt_mono_mag * torch.cos(gt_mono_phase)).unsqueeze(0),
                                 (gt_mono_mag * torch.sin(gt_mono_phase)).unsqueeze(0)], dim=0)
    gt_mono_realImg = gt_mono_realImg.permute(1, 0, 2, 3).unsqueeze(1).contiguous().view(gt_mono_realImg.size(1), 1,
                                                                                         -1)

    # predicted monaural spectrogram
    pred_mono_mag = pred_mono[..., 0]
    pred_mono_realImg = torch.cat([(pred_mono_mag * torch.cos(gt_mono_phase)).unsqueeze(0),
                                   (pred_mono_mag * torch.sin(gt_mono_phase)).unsqueeze(0)], dim=0)
    pred_mono_realImg = pred_mono_realImg.permute(1, 0, 2, 3).unsqueeze(1).contiguous().view(pred_mono_realImg.size(1), 1,
                                                                                             -1)

    assert gt_mono_realImg.size() == pred_mono_realImg.size(),\
        "shapes of realImg spec. of monaural gt and pred are different, gt_ch: {}, pred_ch: {}"\
        .format(gt_mono_realImg.size(), pred_mono_realImg.size())

    mono_stft_l2_dist = torch.mean(torch.pow((gt_mono_realImg - pred_mono_realImg), 2), dim=2).cpu()

    return bin_stft_l2_dist, mono_stft_l2_dist
