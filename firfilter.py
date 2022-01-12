import numpy as np

'''
FIR Filter Class
\method __init__: For declaration of variables used in following methods
\method dofilter: Function for FIR filter 
\method doFilterAdaptive: Function for Least Mean Square Filter for adaptive filtering
'''
class FIRfilter:

    def __init__(self, _coefficients):
        self.number_of_taps = len(_coefficients)   #Number of Taps
        self.coefficients = _coefficients   
        self.buffer = np.zeros(self.number_of_taps)   #Creating Buffer

    '''
    dofilter
    \parameter v: scalar value of coefficient to process the respective value of ecg 
    '''
    def dofilter(self, v):
        for i in range(self.number_of_taps - 1):   #Shifting Values and Making Room for new values
            self.buffer[self.number_of_taps - i - 1] = self.buffer[self.number_of_taps - i - 2]
        self.buffer[0] = v   #Setting input as new value in buffer
        result = np.inner(self.buffer, self.coefficients)    #Calculating result: Multiplying input with coeff.
        return result
    '''
    doFilterAdaptive
    \parameter signal: Sclar vale from signal to be filtered
    \parameter noise: Noise to be removed from the signal
    \parameter learningRate: Rate of adaptation for filtering
    '''
    def doFilterAdaptive(self, signal, noise, learningRate):
        for j in range(self.number_of_taps):
            self.coefficients[j] = self.coefficients[j] + (signal - noise) * learningRate * self.buffer[j]