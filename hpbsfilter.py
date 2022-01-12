import numpy as np
import matplotlib.pyplot as plt
import firfilter

'''
Class for generation of Numeric Coefficients for FIR filtering
\Method highPassDesign: Generates Highpass Numeric Coefficients for FIR filtering
\Method bandstopDesign: Generates Bandstop Numeric Coefficients for FIR filtering
'''
class NumericFIRCoefficients():
    '''
    HighPass Filter Function
    \parameter sample_rate: Samping frequency of the data to be filtered
    \parameter cut_f: Cut off frequency of the high pass filter
    \parameter factor: Factor for tap calculation. Helps optimise stopband Rejection. Default value is 6.
    '''
    def highpassDesign(fs, fc, factor = 6):
        return NumericFIRCoefficients.bandstopDesign(fs, 0, fc, factor)
    
    
    #BandStop Function: Filters signal for freqnecies between the two input values
    '''
    Bandstop Filter Function
    \parameter sample_rate: Samping frequency of the data to be filtered
    \parameter cut_f1: Lower limit of frequencies to be filtered
    \parameter cut_f1: Upper limit of frequencies to be filtered
    \parameter factor: Factor for tap calculation. Helps optimise stopband Rejection. Default value is 6.
    '''
    def bandstopDesign(fs, f1, f2, factor=6):
        f_res = np.abs(f2 - f1)   #Resolution, i.e., minimum frequency difference between two consecutive samples
        taps = int(factor*(fs/f_res))   #Calculation of Number of Taps using Sampling Frequency, Frequency Resolution, and Factor
        bandstop_spectrum = np.ones(taps)   #Array for Ideal Bandstop Response in Frequency Domain
        k_cutoff1 = int(f1/fs * taps)   #Calculation of Samples Corresponding to Lower Limit of frequencies to be filtered
        k_cutoff2 = int(f2/fs * taps)   #Calculation of Samples Corresponding to Upper Limit of frequencies to be filtered

        bandstop_spectrum[k_cutoff1: k_cutoff2+1] = 0   #Ideal Bandstop Response in Frequency Domain
        bandstop_spectrum[taps - k_cutoff2:taps - k_cutoff1] = 0  #Mirrored Ideal Bandstop Response in Frequency Domain

        bandstop_coeff = np.real(np.fft.ifft(bandstop_spectrum))   #Real values of IFFT Response, i.e., unswapped coefficients
        h_bandstop = np.append(bandstop_coeff[int(taps/2):taps], bandstop_coeff[0:int(taps/2)])   #Corrected (Swapped) Coefficients
        h_bandstop = h_bandstop * np.blackman(taps)   #Window multiplied for Better Stopband Rejection

        return bandstop_spectrum, h_bandstop   #Values of Ideal Bandstop Response and Coefficients for FIR Filter returned


'''
Main Function
Implementation Highpass and Bandstop filters
'''
def main():
    ecg = np.loadtxt('ecg.dat')
    fs = 250   #Sampling rate
    N = len(ecg)   #Number of samples
    f1 = 45   #Cutoff frequency 1
    f2 = 55   #Cutoff frequency 2
    fc = 5    #Highpass cutoff frequency to remove baseline wander
    duration = N /fs   #Duration of the ecg
    t = np.linspace(0, duration, N)

    # obtaining the coefficients from q1
    highpass_spectrum, h_highpass = NumericFIRCoefficients.highpassDesign(fs, fc)
    bandstop_spectrum, h_bandstop = NumericFIRCoefficients.bandstopDesign(fs, f1, f2)

    # filtering the ECG signal using the FIRfilter class
    FIR_highpass = firfilter.FIRfilter(h_highpass)
    FIR_bandstop = firfilter.FIRfilter(h_bandstop)

    output = np.zeros(N)
    output2 = np.zeros(N)

    for i in range(N):
        # output removing the baseline wander using the highpass filter
        output[i] = FIR_highpass.dofilter(ecg[i])
        # output 2 removing the 50Hz interference using the bandstop filter
        output2[i] = FIR_bandstop.dofilter(output[i])

    #Plots for raw and filtered ecg
    #Plotting the raw ecg
    plt.figure(figsize=(12, 8))
    plt.plot(t, ecg)
    plt.xlabel('time (s)')
    plt.ylabel('Amplitude (V)')
    plt.title('ECG, raw')
    plt.savefig('ecg_raw.eps', format='eps')

    #Plotting raw ecg in frequency domain
    ecgf = np.fft.fft(ecg)
    faxis = np.linspace(0, fs, N)
    plt.figure(figsize=(12, 8))
    plt.plot(faxis, 20*np.log10(np.abs(ecgf)))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('X(\u03C9)')
    plt.title('ECG in the frequency domain, unfiltered')
    plt.savefig('ecgf_unfiltered.eps', format='eps')
    
    #Plotting the filtered ecg
    ecg_filtered = output2
    plt.figure(figsize=(12, 8))
    plt.title('ECG filtered using FIRfilter class')
    plt.xlabel('time(s)')
    plt.ylabel('amplitude(mV)')
    plt.plot(t, ecg_filtered)
    plt.savefig('ecg_filtered_hpbsfilter.eps', format='eps')

    #Plotting raw ecg in frequency domain
    ecgf_filtered = np.fft.fft(ecg_filtered)
    plt.figure(figsize=(12, 8))
    plt.plot(faxis, 20*np.log10(np.abs(ecgf_filtered)))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('X(\u03C9)')
    plt.title('ECG in the frequency domain, filtered')
    plt.savefig('ecgf_filtered.eps', format='eps')

    plt.show()

    return ecg_filtered, fs

if __name__ == '__main__':
    main()