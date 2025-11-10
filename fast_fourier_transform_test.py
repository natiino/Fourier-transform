
import numpy as np
from scipy.fft import fft, rfft
from scipy.fft import fftfreq, rfftfreq
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import timeit
from scipy.misc import electrocardiogram


# Building a class Signal for better use.
class Signal:
  """
  Generate sinusoidal signals with specific ampltiudes, frequencies, duration,
  sampling rate, and phase.
  Example:
    signal = Signal(amplitude=10, sampling_rate=2000.0)
    sine = signal.sine()
    cosine = signal.cosine()
  """

  def __init__(self, amplitude=1, frequency=10, duration=1, sampling_rate=100.0, phase=0):
    """
    Initialize the Signal class.
    Args:
        amplitude (float): The amplitude of the signal
        frequency (int): The frequency of the signal Hz
        duration (float): The duration of the signal in second
        sampling_rate (float): The sampling per second of the signal
        phase (float): The phase of the signal in radians
    
    Additional parameters,which are required to generate the signal, are
    calculated and defined to be initialized here too:
        time_step (float): 1.0/sampling_rate
        time_axis (np.array): Generate the time axis from the duration and
                              the time_step of the signal. The time axis is
                              for better representation of the signal.
    """
    self.amplitude = amplitude
    self.frequency = frequency
    self.duration = duration
    self.sampling_rate = sampling_rate
    self.phase = phase
    self.time_step = 1.0/self.sampling_rate
    self.time_axis = np.arange(0, self.duration, self.time_step)
  
  # Generate sine wave
  def sine(self):
    """
    Method of Signal
    Returns:
        np.array of sine wave using the pre-defined variables (amplitude,
        frequency, time_axis, and phase)
    """
    return self.amplitude*np.sin(2*np.pi*self.frequency*self.time_axis+self.phase)
  
  # Generate cosine wave
  def cosine(self):
    """
    Method of Signal
    Returns:
        np.array of cosine wave using the pre-defined variables (amplitude,
        frequency, time_axis, and phase)
    """
    return self.amplitude*np.cos(2*np.pi*self.frequency*self.time_axis+self.phase)
  

# Generate the three signals using Signal class and its method sine()
signal_1hz = Signal(amplitude=3, frequency=1, sampling_rate=400, duration=2)
sine_1hz = signal_1hz.sine()
signal_20hz = Signal(amplitude=1, frequency=20, sampling_rate=400, duration=2)
sine_20hz = signal_20hz.sine()
signal_10hz = Signal(amplitude=0.5, frequency=10, sampling_rate=400, duration=2)
sine_10hz = signal_10hz.sine()

# Sum the three signals to output the signal we want to analyze
signal = sine_1hz + sine_20hz + sine_10hz

# Plot the signal
plt.plot(signal_1hz.time_axis, signal, 'b')
plt.xlabel('Time [sec]')
plt.ylabel('Amplitude')
plt.title('Sum of three signals')
plt.show()

# Apply the FFT on the signal
fourier = fft(signal)

# # Plot the result (the spectrum |Xk|)
# plt.plot(np.abs(fourier))
# plt.show()

# The figure above should represent the frequency spectrum of the signal. 
# Notice that the x-axis is the number of samples (instead of the frequency components) 
# and the y-axis should represent the amplitudes of the sinusoids. 
# To get the actual amplitudes of the spectrum, we have to normalize the output of (fft) 
# by N/2 the number of samples.

# Calculate N/2 to normalize the FFT output
N = len(signal)
normalize = N/2

# Plot the normalized FFT (|Xk|)/(N/2)
plt.plot(np.abs(fourier)/normalize)
plt.ylabel('Amplitude')
plt.xlabel('Samples')
plt.title('Normalized FFT Spectrum')
plt.show()

# # To get the frequency components (x-axis), you can use (fftfreq) from the scipy package. 
# # This method needs the number of samples (N) and the sampling rate as input arguments. 
# # And it returns a frequency axis with N frequency components [5].

# Get the frequency components of the spectrum
sampling_rate = 200.0 # It's used as a sample spacing
frequency_axis = fftfreq(N, d=1.0/sampling_rate)
norm_amplitude = np.abs(fourier)/normalize
# Plot the results
plt.plot(frequency_axis, norm_amplitude)
plt.xlabel('Frequency[Hz]')
plt.ylabel('Amplitude')
plt.title('Spectrum')
plt.show()

# # Plot the frequency axis for more explanation
# plt.plot(frequency_axis)
# plt.ylabel('Frequency[Hz]')
# plt.title('Frequency Axis')
# plt.show()

# # Calculate the time execution of (fft)
# print('Execution time of fft function:')
# timeit.timeit(fft(signal))
# # Calculate the time execution of (rfft)
# print('\nExecution time of rfft function:')
# timeit.timeit(rfft(signal))

# Plot the actual spectrum of the signal
plt.plot(rfftfreq(N, d=1/sampling_rate), 2*np.abs(rfft(signal))/N)
plt.title('Spectrum')
plt.xlabel('Frequency[Hz]')
plt.ylabel('Amplitude')
plt.show()

# Building a class Fourier for better use of Fourier Analysis.
class Fourier:
  """
  Apply the Discrete Fourier Transform (DFT) on the signal using the Fast Fourier 
  Transform (FFT) from the scipy package.

  Example:
    fourier = Fourier(signal, sampling_rate=2000.0)
  """

  def __init__(self, signal, sampling_rate):
    """
    Initialize the Fourier class.

    Args:
        signal (np.ndarray): The samples of the signal
        sampling_rate (float): The sampling per second of the signal
    
    Additional parameters,which are required to generate Fourier calculations, are
    calculated and defined to be initialized here too:
        time_step (float): 1.0/sampling_rate
        time_axis (np.ndarray): Generate the time axis from the duration and
                              the time_step of the signal. The time axis is
                              for better representation of the signal.
        duration (float): The duration of the signal in seconds.
        frequencies (numpy.ndarray): The frequency axis to generate the spectrum.
        fourier (numpy.ndarray): The DFT using rfft from the scipy package.
    """
    self.signal = signal
    self.sampling_rate = sampling_rate
    self.time_step = 1.0/self.sampling_rate
    self.duration = len(self.signal)/self.sampling_rate
    self.time_axis = np.arange(0, self.duration, self.time_step)
    self.frequencies = rfftfreq(len(self.signal), d = self.time_step)
    self.fourier = rfft(self.signal)
    
  # Generate the actual amplitudes of the spectrum
  def amplitude(self):
    """
    Method of Fourier

    Returns:
        numpy.ndarray of the actual amplitudes of the sinusoids.
    """
    return 2*np.abs(self.fourier)/len(self.signal)

  # Generate the phase information from the output of rfft  
  def phase(self, degree = False):
    """
    Method of Fourier

    Args:
        degree: To choose the type of phase representation (Radian, Degree).
                By default, it's in radian. 

    Returns:
        numpy.ndarray of the phase information of the Fourier output.
    """
    return np.angle(self.fourier, deg = degree)

  # Plot the spectrum
  def plot_spectrum(self, interactive=False):
    """
    Plot the Spectrum (Frequency Domain) of the signal either using the matplotlib
    package, or plot it interactive using the plotly package.

    Args:
        interactive: To choose if you want the plot interactive (True), or not
        (False). The default is the spectrum non-interactive.

    Retruns:
        A plot of the spectrum.
    """
    # When the argument interactive is set to True:
    if interactive:
      self.trace = go.Line(x=self.frequencies, y=self.amplitude())
      self.data = [self.trace]
      self.layout = go.Layout(title=dict(text='Spectrum',
                                         x=0.5,
                                         xanchor='center',
                                         yanchor='top',
                                         font=dict(size=25, family='Arial, bold')),
                              xaxis=dict(title='Frequency[Hz]'),
                              yaxis=dict(title='Amplitude'))
      self.fig = go.Figure(data=self.data, layout=self.layout)
      return self.fig.show()
    # When the argument interactive is set to False:
    else:
      plt.figure(figsize = (10,6))
      plt.plot(self.frequencies, self.amplitude())
      plt.title('Spectrum')
      plt.ylabel('Amplitude')
      plt.xlabel('Frequency[Hz]')
  
  # Plot the Signal and the Spectrum interactively
  def plot_time_frequency(self, t_ylabel="Amplitude", f_ylabel="Amplitude",
                          t_title="Signal (Time Domain)",
                          f_title="Spectrum (Frequency Domain)"):
    """
    Plot the Signal in Time Domain and Frequency Domain using plotly.

    Args:
        t_ylabel (String): Label of the y-axis in Time-Domain
        f_ylabel (String): Label of the y-axis in Frequency-Domain
        t_title (String): Title of the Time-Domain plot
        f_title (String): Title of the Frequency-Domain plot 

    Returns:
        Two figures: the first is the time-domain, and the second is the
                     frequency-domain.
    """
    # The Signal (Time-Domain)
    self.time_trace = go.Line(x=self.time_axis, y=self.signal)
    self.time_domain = [self.time_trace]
    self.layout = go.Layout(title=dict(text=t_title,
                                       x=0.5,
                                       xanchor='center',
                                       yanchor='top',
                                       font=dict(size=25, family='Arial, bold')),
                            xaxis=dict(title='Time[sec]'),
                            yaxis=dict(title=t_ylabel),
                            width=1000,
                            height=400)
    fig = go.Figure(data=self.time_domain, layout=self.layout)
    fig.show()
    # The Spectrum (Frequency-Domain)
    self.freq_trace = go.Line(x=self.frequencies, y=self.amplitude())
    self.frequency_domain = [self.freq_trace]
    self.layout = go.Layout(title=dict(text=f_title,
                                       x=0.5,
                                       xanchor='center',
                                       yanchor='top',
                                       font=dict(size=25, family='Arial, bold')),
                            xaxis=dict(title='Frequency[Hz]'),
                            yaxis=dict(title=f_ylabel),
                            width=1000,
                            height=400)
    fig = go.Figure(data=self.frequency_domain, layout=self.layout)
    fig.show()

# Apply the DFT using the class Fourier
fourier = Fourier(signal, sampling_rate=200)
# Plot the spectrum interactively using the class Fourier
fourier.plot_spectrum(interactive=True)

# Import the ECG signal from scipy package
from scipy.misc import electrocardiogram
# Built-in ECG signal
ecg = electrocardiogram()
# DFT using the class Fourier
ecg_spectrum = Fourier(signal = ecg, sampling_rate = 300.0)
# Plot the time-frequency domains of the ECG signal
ecg_spectrum.plot_time_frequency(t_title="ECG Signal", f_title="ECG Spectrum",
                                 t_ylabel="Amplitude[mV]")