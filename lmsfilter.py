import numpy as np
import matplotlib.pyplot as plt
import firfilter
import hpbsfilter

'''
Main Function
Implementation least mean square filter
'''
def main():
    ecg = np.loadtxt('ecg.dat')
    fs = 250            #Sampling rate
    N = len(ecg)        #Number of samples
    fc = 5              #Highpass cutoff frequency to remove baseline wander
    duration = N /fs    #Duration of the ecg
    t = np.linspace(0, duration, N)
    noise = 50
    learning_rate = 0.001

    highpass_spectrum, h_highpass = hpbsfilter.NumericFIRCoefficients.highpassDesign(fs, fc)
    FIR_highpass = firfilter.FIRfilter(h_highpass)

    signal_in = firfilter.FIRfilter(np.zeros(len(h_highpass)))
    lms_output = np.empty(len(ecg))
    filtered_signal = np.empty(len(ecg))

    for i in range(len(ecg)):
        # output removing the 50Hz interference using the LMS filter
        ref_noise = np.sin(2.0 * np.pi * noise / fs * i)
        canceller = signal_in.dofilter(ref_noise)
        output_signal = ecg[i] - canceller
        signal_in.doFilterAdaptive(ecg[i], canceller, learning_rate)
        lms_output[i] = output_signal
        # now remove the baseline wander using the highpass filter
        filtered_signal[i] = FIR_highpass.dofilter(lms_output[i])

    # plotting the ecg signal filtered using LMS
    plt.figure(2, figsize=(12, 8))
    plt.plot(t, filtered_signal)
    plt.xlabel('time(s)')
    plt.ylabel('amplitude(mV)')
    plt.title('ECG filtered using LMS')
    plt.savefig('ecg_filtered_lmsfilter.eps', format='eps')
    plt.show()

if __name__ == '__main__':
    main()