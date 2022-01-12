import numpy as np
import matplotlib.pyplot as plt
import hpbsfilter
import firfilter
import scipy.signal as signal

class TimeStates():
    """
    This class is a very simple state machine that is used to
    determine whether a particular T value is being used while
    calculating the pulse.

    In the detector with threshold graph, it can be seen that the
    peaks where the detector is 1 include 2 samples. This is because
    some peaks in the detector are low in amplitude, and the threshold
    has to be low as well. We need to override the 2nd sample to calculate
    the pulse; as the way to calculate the pulse is to find 2 consecutive
    peaks, calculate the time difference, invert it and multiply by 60.

    The way to override the 2nd sample is by introducing this state machine
    with 2 states: READ and NOREAD. When the state is READ, we take the value
    of the time we currently are in and use it to calculate the pulse. If the
    state is NOREAD, the t value is ignored.

    The state is determined by a threshold. If the time difference between the
    current time and our first peak is smaller than the threshold, it means
    that we are still on the same peak and the state should be NOREAD, which
    means this particulat t value must be ignored. Else, the state is READ and
    we are on another peak.
    """
    _state = 'READ' # the initial state is read

    def switch_states(self, t, t1):
        delta_t = t - t1
        threshold_state = 5E-3
        if delta_t < threshold_state:
            self.state_ = 'NOREAD'
        else:
            self.state_ = 'READ'
        return self.state_

def create_template(ecg, fs):
    """
    Extract the template from the ECG.
    """
    template = ecg[700:880]
    template_reversed = template[::-1]  # template reversed
    N = len(template)
    duration = N / fs
    t = np.linspace(0, duration, N)
    return t, template, template_reversed, len(template)


def create_wavelet(N, fs):
    """
    Time reverse the template and obtain the wavelet
    that will be used to filter the ECG.
    """
    duration = N / fs
    t = np.linspace(-duration/2, duration/2, N)
    wavelet = np.sinc(t*fs)
    return wavelet


def detect(ecg, fs, wavelet):
    """
    Detection algorithm that filters the signal, then
    squares it in order to increase the SNR.
    """
    ecg_cropped = ecg[400:]
    N = len(ecg_cropped)
    duration = N / fs
    t = np.linspace(0, duration, N)
    FIR_wavelet = firfilter.FIRfilter(wavelet)
    detector = np.zeros(N)
    for i in range(N):
        detector[i] = FIR_wavelet.dofilter(ecg_cropped[i])
    detector = detector**2
    return t, detector, N


def apply_threshold(detector, N, template_type):
    """
    Applies a threshold and determines the peaks
    """
    threshold = 0.0
    if template_type == 'wavelet':
        threshold = 9E-7  # amplitude value to determine the peaks
    elif template_type == 'template':
        threshold = 1.8E-11
    detector_new = np.zeros(N)
    for i in range(N):
        if detector[i] > threshold:
            detector_new[i] = 1
    return detector_new


def calculate_heartrate(t, detector, N, fs):
    """
    The momentary heartrate is calculated by going through the peaks
    one-by-one, finding the interval between two consecutive peaks, and
    inversing it. The result gives us the beats per second, and by multiplying
    with 60, we get the beats per minute.
    """
    heart_rate = np.zeros(N)
    t1_temp = 0 # temp value to store previous peak
    peak_range = np.arange(0, N) # we reduce the peak range in order to override previous peaks
    count = 1 # counter to count peaks
    time_state = TimeStates()
    for i in range(N):
        t1 = t1_temp # time of the first peak
        t2 = np.max(t) # time of the second peak initialized as the maximum time
        for i in peak_range:
            if detector[i] == 1: # we found a peak
                state = time_state.switch_states(t[i], t1) # determine state according to delta_t
                if count < 2: # only the first peak is found with this
                    t1 = t[i]
                    count += 1
                elif count == 2: # any subsequent peak falls into this category
                    if state == 'READ':
                        t2 = t[i]
                        count += 1
                else: # two peaks have been located, and we need to leave the loop
                    break

        delta_t = t2 - t1 # time difference between two peaks
        hr_momentary = 1/delta_t * 60 # 1/delta_t is the frequency of peaks, times 60 gives bpm

        sample_1 = int(t1 * fs) # sample where the first peak is
        sample_2 = int(t2 * fs) # sample where the second peak is
        heart_rate[sample_1:sample_2+2] = hr_momentary # momentary heartrate between two samples

        t1_temp = t2 # second peak in iteration i is the first peak in iteration i+1
        peak_range = np.arange(sample_2+2, N) # peak range is updated
        count = 2

    # crop the part before the first peak, as the heart rate is not calculated
    # before it and the array is 0
    hr_init = 0.0
    for sample in heart_rate:
        if sample > 0:
            hr_init = np.where(heart_rate == sample)[0][0]
            break
    heart_rate_cropped = heart_rate[hr_init:]

    # update the t variable for a correct x axis
    t_cropped = t[hr_init:]

    return t_cropped, heart_rate_cropped


def main():
    ecg_filtered, fs = hpbsfilter.main()

    t_template, template, template_reversed, N_template = create_template(ecg_filtered, fs)
    wavelet = create_wavelet(N_template, fs)
    t, detector_wavelet, N = detect(ecg_filtered, fs, wavelet)
    t, detector_template, N = detect(ecg_filtered, fs, template)
    detector_wavelet_new = apply_threshold(detector_wavelet, N, 'wavelet')
    detector_template_new = apply_threshold(detector_template, N, 'template')
    t_heartrate, heart_rate = calculate_heartrate(t, detector_wavelet_new, N, fs)

    plt.figure(1, figsize=(12, 8))
    plt.plot(t_template, template)
    plt.title('One heartbeat')
    plt.xlabel('time (s)')
    plt.ylabel('Amplitude')
    plt.savefig('heartbeat.eps', format='eps')

    plt.figure(2, figsize=(12, 8))
    plt.subplot(121)
    plt.plot(template, label='template, time reversed')
    plt.title('Template, time reversed')
    plt.xlabel('sample')
    plt.ylabel('Amplitude (V)')
    plt.subplot(122)
    plt.plot(wavelet, label='WAVELET')
    plt.title('WAVELET')
    plt.xlabel('sample')
    plt.savefig('template-wavelet.eps', format='eps')

    fig, ax = plt.subplots(4, 1, figsize=(12, 8))
    ax[0].plot(t, detector_wavelet)
    ax[0].set_ylabel('Amplitude')
    ax[0].set_title('Detector with wavelet')
    ax[0].get_xaxis().set_visible(False)

    ax[1].plot(t, detector_wavelet_new)
    ax[1].set_ylabel('Amplitude')
    ax[1].set_title('Detector with wavelet, added threshold')
    ax[1].get_xaxis().set_visible(False)

    ax[2].plot(t, detector_template)
    ax[2].set_ylabel('Amplitude')
    ax[2].set_title('Detector with template')
    ax[2].get_xaxis().set_visible(False)

    ax[3].plot(t, detector_template_new)
    ax[3].set_xlabel('time (s)')
    ax[3].set_ylabel('Amplitude')
    ax[3].set_title('Detector with template, added threshold')
    plt.savefig('detector.eps', format='eps')

    plt.figure(4, figsize=(12, 8))
    plt.plot(t_heartrate, heart_rate)
    plt.xlabel('time (s)')
    plt.ylabel('Heartrate')
    plt.title('Momentary Heartrate against Time')
    plt.savefig('heartrate.eps', format='eps')

    plt.show()


if __name__ == '__main__':
    main()
