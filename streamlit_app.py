import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import mentari_v2 as mtr
import astropy.units as u
from scipy import integrate 
from scipy.integrate import simps

def open_file(filename):
    
    '''
    Open file, read each line, split each line into each float number.
    Create an list consists of all the number (float) in the file.
    
    Input: - filename (str) -- name of file to be opened
    Output: - M (list(float)) -- a list of float consists of all number in the file named filename
    '''
    
    f = open(filename, "r")
    M = []
    for elem in f.read().split():
        try:
            M.append(float(elem))
        except ValueError:
            pass
    f.close()
    
    return M

def luminosity_distance(z, h0=73., omega_m=0.27, omega_l=0.73):
    
    '''
    Computing luminosity distance
    Input:  - z (float) -- redshift
            - h0 (float) (optional) -- hubble constant (in km/pc)
            - omega_m (float) (optional) -- matter density parameter
            - omega_l (float) (optional) -- dark energy density parameter
            
    Output: - luminosity distance (float) -- in parsec
    '''
    
    c = 2.9979e18 #velocity of lights
    omega_k = 1. - omega_m - omega_l
    dh = c/1.e13/h0 * 1.e6 #in pc
    
    if z > 0.:
        dc, edc = integrate.quad(lambda x: (omega_m * (1.+x)** 3 + omega_k * (1+x)**2 + omega_l)**(-.5), 0., z, epsrel=1e-4)
        dc = dh * dc
    else:
    # Bad idea as there is something *wrong* going on
        print('LumDist: z <= 0 -> Assume z = 0!')
        z = 0.
        #dist = 0.
        return 10
    
    if omega_k > 0.:
    	dm = dh * np.sinh(dc/dh * np.sqrt(omega_k)) / np.sqrt(omega_k)
    elif omega_k < 0.:
    	dm = dh * np.sin(dc/dh * np.sqrt(-omega_k)) / np.sqrt(-omega_k)
    else:
    	dm = dc
    return dm * (1+z)

def compute_effective_wavelength(filter):
    filters_wave = eval('F.' + filter + '_wave')
    filters = eval('F.' + filter)
    upper = simps(filters * filters_wave, filters_wave)
    lower = simps(filters / filters_wave, filters_wave)
    lambda_eff = np.sqrt(upper/lower)
    
    return(lambda_eff)

def compute_tau(tau_head, eta, wavelength):

    '''
    Compute optical depth as a function of wavelength using Charlot and Fall (2000) model
    
    Input:  - tau_head (float or array): optical depth at 5500 Angstorm
            - eta (float or array): power law index
            - wavelength (array): in Angstorn           
    Output: - tau (array): the computed optical depth have the same dimension with wavelength. 
    '''


    if type(tau_head) != float:
        tau = np.zeros((len(tau_head), len(wavelength)))
        for i in range(len(tau)):
            tau[i] = tau_head[i] * (wavelength/5500)**eta[i]
    else:
        tau = tau_head * (wavelength/5500)**eta

    return(tau)

def add_IR_Dale (wavelength, spectra, spectra_dusty):

    '''
    Add the NIR-FIR spectra from Dale+ 14 IR template to the UV-NIR spectra from BC03/
    
    Input:  - wavelength (N-dimensional array)
            - spectra (N-dimensional array) - intrinsic stellar spectra corresponding to each wavelength
            - spectra_dusty (N-dimensional array) - attenuated spectra corresponding to each wavelength
    Output: - wavelength (M-dimensional array) - M > N
            - spectra (M-dimensional array) - attenuated spectra with IR addition
    '''
    
    Dale_template = np.loadtxt('files/spectra.0.00AGN.dat', unpack=True)
    lambda_IR = Dale_template[0] * 1e4 #convert from micron to Angstrom

    Ldust = (spectra - spectra_dusty)
    w = np.where(wavelength < 912)[0]
    idx_912 = w[-1]

    all_wave = np.unique(np.concatenate((wavelength, lambda_IR)))
    all_wave.sort(kind='mergesort')
    UVIR = np.zeros((len(Ldust), len(all_wave)))


    LIR_mentari = np.trapz(Ldust[idx_912:-1], wavelength[idx_912:-1])
    #---------------------------------------------------
    
    #Compute alpha based on Rieke+ 2009
    alpha_SF, log_fnu_SF = np.loadtxt('files/alpha.dat', unpack=True)
    
    if (LIR_mentari > 10**11.6):  
        LIR_mentari = 10**11.6

    alpha = 10.096 - 0.741 * np.log10(LIR_mentari)

    delta_alpha = abs(alpha_SF - alpha)
    idx = np.where(delta_alpha==min(delta_alpha))[0]

    spectra_IR = 10 ** Dale_template[idx[0]+1] 

    LIR_dale = np.trapz(spectra_IR, lambda_IR)
    scaling = LIR_mentari / LIR_dale
    spectra_IR_dale = spectra_IR * scaling 

    new_spectra = np.interp(all_wave, wavelength, spectra_dusty)
    new_IR = np.interp(all_wave, lambda_IR, spectra_IR_dale)
    all_spec = new_spectra + new_IR

    return (all_wave, all_spec)

#==========================================================================================================================

FileNames = ["files/bc2003_hr_m22_chab_ssp.ised_ASCII", "/files/bc2003_hr_m32_chab_ssp.ised_ASCII",
            "files/bc2003_hr_m42_chab_ssp.ised_ASCII", "files/bc2003_hr_m52_chab_ssp.ised_ASCII", 
            "files/bc2003_hr_m62_chab_ssp.ised_ASCII", "files/bc2003_hr_m72_chab_ssp.ised_ASCII"]

AllFiles = []

for i in range(len(FileNames)):
    AllFiles.append(open_file(FileNames[i]))
    
File1 = AllFiles[0]
lookback = np.array(File1[1:222])
wavelength = np.array(File1[236: 7136])
metallicity = np.array([0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05]) #metallicity grid in BC03
time_grid = 221
wave_grid = 6900
lum = np.zeros((len(AllFiles), time_grid, wave_grid))

for j in range(len(metallicity)):
    File = AllFiles[j]
    for i in range(time_grid):
        lum[j][i] = File[7137 + (wave_grid+54)*i: 7137 + (wave_grid+54)*i + 6900]
"""
# Mentari
A tool to visualise the spectra of simple stellar populations (SSPs) from the library of Bruzual & Charlot (2003), combine the SSP, and construct spectral energy distribution (SED) for galaxies from ultraviolet to far-infrared wavelength.
"""

function = ['Visualise SSPs', 'Construct a galaxy SED']
selected_option = st.sidebar.radio('What do you want to do?', function)

if selected_option == 'Visualise SSPs':
	"""
	## Visualise SSPs
	An SSP consists of stars born at the same time (have the same age and from a gas with the same metallicity).
	In this page you can explore the spectra of various SSPs, normalize to stellar mass = 1 solar mass.
	Check the 'New SSP' button to add another SSP spectra.
	"""

	left, right = st.beta_columns(2)
	age = left.slider('Age in Gyr', 0.0, 20.0, 0.01)
	metal = right.selectbox(
			'Metallicity', metallicity)

	delta_age = abs((lookback/1.e9) - age)
	age_ind = np.where(delta_age == min(delta_age))[0][0]
	metal_ind = np.where(metallicity == metal)[0][0]

	fig, ax = plt.subplots()
	ax.plot(wavelength, lum[metal_ind][age_ind]*wavelength, label = r'age = {} Gyr, metallicity = {}'.format(age, metal))

	check = st.checkbox('new SSP')
	if check:
		age_a = left.slider('SSP 2: Age in Myr', 0.0, 20.0, 0.01)
		metal_a = right.selectbox(
				'SSP 2: Metallicity', metallicity)

		delta_age = abs((lookback/1.e9) - age_a)
		age_ind = np.where(delta_age == min(delta_age))[0][0]
		metal_ind = np.where(metallicity == metal_a)[0][0]

		ax.plot(wavelength, lum[metal_ind][age_ind]*wavelength, label = r'age = {} Gyr, metallicity = {}'.format(age_a, metal_a))
	ax.set_yscale('log')
	ax.set_xscale('log')
	plt.xlabel(r'$\lambda\ (\AA) $')
	plt.ylabel(r'log $\lambda L_{\lambda} (L_{\odot})$')
	plt.legend()
	st.pyplot(fig)


if selected_option == 'Construct a galaxy SED':
    """
    ## Construct a galaxy SED
    A galaxy consists of multiple stellar populations. Each populations have different stellar mass.
    Total stellar emission from a galaxy is the sum of all SSPs in it. The stellar emission is subjected to attenuation by dust. The heated dust reradiates the attenuated light in IR. 
    Here you can combine up to 5 stellar populations to mimic intrinsic emission from a galaxy. 
    Then, you can apply a Charlot & Fall (2000) attenuation model. 
    To model the mid-infrared emission, you can choose between a model of Dale et al. (2014) or Safarzadeh et al. (2015).

    """

    fig, ax = plt.subplots()
    input_yes = st.sidebar.checkbox('Do you have photometric data to compare to? (check if yes)')
    if input_yes:
        st.write('### Add fluxes datapoints')
        F = mtr.read_filters()
        z = st.number_input('redshift (development note: currently only work with z=0)')
        filterlist = list(np.genfromtxt('files/filterlist', dtype=str))

        filtername = st.multiselect('Choose filters', filterlist , ['Sdss_u', 'Sdss_r'])
        st.write('Observed flux (in Jansky)')
        for i in range(len(filtername)):
            #eff_lambda = left.number_input('Effective wavelength ' + filtername[i], 0.356) * u.micrometer
            eff_lambda = compute_effective_wavelength(filtername[i]) * u.AA
            flux = st.number_input('Flux ' + filtername[i], 0.001) * u.Jy
            distance = luminosity_distance(z) * u.pc
            frequency = eff_lambda.to(u.Hz, equivalencies=u.spectral())
            obs_lum = (flux * 4 * np.pi * distance**2 * frequency).decompose()
            obs_Lsun = obs_lum.to(u.Lsun) 
            #obs_lum = (1. + z) * flux * 3e14 / eff_lambda
            ax.plot(eff_lambda, obs_Lsun, 'o', label = filtername[i]) 

    st.write('### Construct intrinsic spectra')
    lum_list = []
    age_list = []
    total_mass = 0
    ll, left, middle, right = st.beta_columns(4)
    individual = st.checkbox('show individual SSP')
        
    age_units = ['Myr', 'Gyr']
    age_unit = ll.radio('Unit for SSP 1 age', age_units)
    if age_unit == 'Gyr':
        age = left.slider('SSP 1: Age in Gyr', 0.0, 20.0, 0.01) * 1e9
        age_list.append(age * 1e9)
    else:
        age = left.slider('SSP 1: Age in Myr', 0.0, 100.0, 0.1) * 1e6
        age_list.append(age * 1e6)
        
    metal = middle.selectbox('SSP 1: Metallicity', metallicity)
    mass = right.slider('SSP 1: Log mass (mass in solar mass)', 0., 12., 0.5)
    total_mass += 10 ** mass
    
    delta_age = abs((lookback) - age)
    age_ind = np.where(delta_age == min(delta_age))[0][0]
    metal_ind = np.where(metallicity == metal)[0][0]

    lum_ind = lum[metal_ind][age_ind] * 10 ** mass
    total_lum = lum_ind
    lum_list.append(lum_ind)
    if individual:
        ax.plot(wavelength, lum_ind * wavelength,  label = r'SSP 1')

    SSP_2 = st.sidebar.checkbox('add SSP 2')
    if SSP_2:
        age_unit = ll.radio('Unit for SSP 2 age', age_units)
        if age_unit == 'Gyr':
            age_2 = left.slider('SSP 2: Age in Gyr', 0.0, 20.0, 0.01) * 1e9
            age_list.append(age_2 * 1e9)
        else:
            age_2 = left.slider('SSP 2: Age in Myr', 0.0, 100.0, 0.1) * 1e6
            age_list.append(age_2 * 1e6)

        metal_2 = middle.selectbox('SSP 2: Metallicity', metallicity)
        mass_2 = right.slider('SSP 2: Log mass (mass in solar mass)', 0., 12., 0.5)
        total_mass += 10 ** mass_2
        
        delta_age = abs((lookback) - age_2)
        age_ind = np.where(delta_age == min(delta_age))[0][0]
        metal_ind = np.where(metallicity == metal_2)[0][0]

        lum_ind_2 = lum[metal_ind][age_ind] * 10 ** mass_2
        total_lum = lum_ind + lum_ind_2
        lum_list.append(lum_ind_2)

        if individual:
            ax.plot(wavelength, lum_ind_2 * wavelength, label = r'SSP 2')

    SSP_3 = st.sidebar.checkbox('add SSP 3')
    if SSP_3:
        age_unit = ll.radio('Unit for SSP 3 age', age_units)
        if age_unit == 'Gyr':
            age_3 = left.slider('SSP 3: Age in Gyr', 0.0, 20.0, 0.01) * 1e9
            age_list.append(age_3 * 1e9)
        else:
            age_3 = left.slider('SSP 3: Age in Myr', 0.0, 100.0, 0.1) * 1e6
            age_list.append(age_3 * 1e6)
        
        metal_3 = middle.selectbox('SSP 3: Metallicity', metallicity)
        mass_3 = right.slider('SSP 3: Log mass (mass in solar mass)', 0., 12., 0.5)
        total_mass += 10 ** mass_3
        
        delta_age = abs((lookback) - age_3)
        age_ind = np.where(delta_age == min(delta_age))[0][0]
        metal_ind = np.where(metallicity == metal_3)[0][0]

        lum_ind_3 = lum[metal_ind][age_ind] * 10 ** mass_3
        total_lum = lum_ind + lum_ind_3
        lum_list.append(lum_ind_3)

        if individual:
                ax.plot(wavelength, lum_ind_3 * wavelength, label = r'SSP 3')
    
    st.write('Total stellar mass (in Msun) = ', total_mass)
    ax.plot(wavelength, total_lum * wavelength, 'b-', label = r'total SED')

    st.write('### Add dust attenuation')
    attenuation = st.checkbox('Add Charlot & Fall (2000) attenuation model')
    if attenuation:
        left, right = st.beta_columns(2)
        tau_head_BC = left.slider('Birth cloud optical depth', 0.0, 2.0, 1.0)
        eta_BC = right.slider('Birth cloud power law index', 0.0, -2.0, -0.7)
        tau_head_ISM = left.slider('Diffuse ISM optical depth', -0.1, 2.0, 0.3)
        eta_ISM = right.slider('Diffuse ISM power law index', -0.1, -2.0, -0.7)
        age_limit = st.slider('Age limit for stars in birth clouds (Myr)', 0, 100, 10)

        tau_ISM = compute_tau(tau_head_ISM, eta_ISM, wavelength)
        tau_BC = tau_ISM + compute_tau(tau_head_BC, eta_BC, wavelength)

        total_lum_att = 0
        lum_att = np.zeros((len(lum_list), len(lum_ind)))
        #if len(age_list) > 1:
        for i in range(len(age_list)):
            if age_list[i] < age_limit * 1e6:
                lum_att[i] = lum_list[i] * np.e**(-tau_BC)
            else:
                lum_att[i] = lum_list[i] * np.e**(-tau_ISM)
            total_lum_att += lum_att[i]
        ax.plot(wavelength, total_lum_att * wavelength, 'k-', label = r'Attenuated SED')

    st.write('### Add IR spectra')
    IR = st.checkbox('Add IR spectra from Dale et al. (2014) template')
    if IR:
        wave_IR, lum_IR = add_IR_Dale (wavelength, total_lum, total_lum_att)
        ax.plot(wave_IR, lum_IR * wave_IR, 'k-')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlabel(r'$\lambda\ (\AA) $')
    plt.ylabel(r'log $\lambda L_{\lambda} (L_{\odot})$')
    plt.legend()
    st.pyplot(fig)
