import numpy as np

def generate_time_profile(times, peak_time, duration, shape):
    """Generate time profile for a single pulse"""
    t0 = peak_time
    
    if shape == 'gaussian':
        sigma = duration
        profile = np.exp(-0.5 * ((times - t0) / sigma)**2)
        
    elif shape == 'exponential':
        tau = duration
        profile = np.where(times >= t0, np.exp(-(times - t0) / tau), 0.0)
        
    elif shape == 'step':
        delta_t = duration
        profile = np.where((times >= t0) & (times <= t0 + delta_t), 1.0, 0.0)
        
    else:
        raise ValueError(f"Unknown time profile shape: {shape}")
    
    return profile

def generate_frequency_profile(freqs, peak_flux, reference_freq, spectral_index):
    """Generate frequency profile (power law spectrum)"""
    return peak_flux * (freqs / reference_freq)**spectral_index

def generate_transient_spectra(times, freqs, transient_params):
    """
    Generate dynamic spectra for a single transient
    
    Parameters:
    -----------
    times : numpy.ndarray
        Time array (seconds)
    freqs : numpy.ndarray  
        Frequency array (Hz)
    transient_params : dict
        Transient parameters from YAML
        
    Returns:
    --------
    spectra : numpy.ndarray
        Dynamic spectra with shape (ntime, nfreq)
    """
    # Extract parameters
    time_params = transient_params['time']
    freq_params = transient_params['frequency']
    periodicity = transient_params.get('periodicity', {'enabled': False})
    
    # since peak_time is set from start of observation
    times = times - times[0]

    # Generate base time profile
    time_profile = generate_time_profile(
        times, 
        time_params['peak_time'], 
        time_params['duration'], 
        time_params['shape']
    )
    
    # Add periodicity if enabled
    if periodicity.get('enabled', False):
        period = periodicity['period']
        total_duration = periodicity.get('total_duration', times[-1] - times[0])
        
        # Create periodic version by summing shifted copies
        periodic_profile = np.zeros_like(times)
        t_start = time_params['peak_time']
        
        # Calculate how many periods fit in the total duration
        n_periods = int((total_duration - t_start) / period) + 1
        
        for n in range(n_periods):
            t_shift = n * period
            if t_start + t_shift <= total_duration:
                shifted_profile = generate_time_profile(
                    times, 
                    time_params['peak_time'] + t_shift, 
                    time_params['duration'], 
                    time_params['shape']
                )
                periodic_profile += shifted_profile
        
        time_profile = periodic_profile
    
    # Generate frequency profile
    freq_profile = generate_frequency_profile(
        freqs,
        freq_params['peak_flux'],
        freq_params['reference_freq'], 
        freq_params['spectral_index']
    )
    
    return time_profile, freq_profile