'''
Routine to Estimate Mobility from TPC Decays
M. Kober-Czerny, J. Lim & B. Wenger
'''
import os
from pathlib import Path, PurePath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
from gooey import Gooey, GooeyParser
from lmfit import Parameters, minimize
from scipy.integrate import odeint
from scipy.optimize import root, minimize_scalar
from scipy.special import gamma
from scipy.signal import medfilt
import sys
sys.setrecursionlimit(5000)


np.seterr(invalid='ignore')



""" Some Important Constants"""
frequency = 10.0 #Hz
global pulse_fwhm
pulse_fwhm = 3.74e-9        #s

Resistance_System = 1000 #Ohm (terminal resistance on oscilloscope)
Fit_range = np.array([0, 1000])


Mask_Values = pd.DataFrame([[39.51e-3,0.1e-3], [31.6e-3,0.3e-3], [23.23e-3,0.5e-3]],
                           index = ['100 μm', '300 μm', '500 μm'],
                           columns = ['Width','Spacing'])


cfgs_mask = ['100 μm', '300 μm', '500 μm', 'Single Crystal']
folder = os.getcwd()
Ref_Files = os.listdir(str(folder + '\TPC_Files'))
# Use flag --ignore-gooey if you want to use the command line
@Gooey(advanced=True,          # toggle whether to show advanced config or not
       default_size=(800, 500),   # starting size of the GUI
       show_success_modal = False,
       navigation = "Tabbed",
       tabbed_groups=True,
       return_to_config = True,
       sidebar_title = "Navigation",
       image_dir=str(folder + '\Icon\\')
)


def get_args():
    """Get arguments and options"""
    parser = GooeyParser(description='Estimation of Mobility from Photoconductivity')

    req = parser.add_argument_group('Main', gooey_options={'columns': 2})
    req.add_argument('-dp', '--data_path', type=str, widget="FileChooser", help="Path to the '_.txt'", gooey_options={'wildcard':"'01' files (*_01.txt)|*_01.txt|" "All files (*.*)|*.*",'full_width':True})
    req.add_argument('-Filters', '--Filter_Wheel', default="2.0", type=float, help="OD of second filter")
    req.add_argument('-volt', '--Voltage', nargs=1, default="2.6", type=float, help="Voltage applied by battery [V]")
    req.add_argument('-th', '--Thickness', default=500, type=float, help="Film Thickness [nm]")
    req.add_argument('-sname', '--Sample_Name', type=str, help="Special Sample Name?")


    req2 = parser.add_argument_group('Electronic Parameters', gooey_options={'columns': 2})
    req2.add_argument('-mask', '--Mask', default='300 μm', widget="Dropdown", choices=cfgs_mask, type=str,
                     help="Electrode mask spacing. Select 'Single Crystal' for individual length and width.")
    req2.add_argument('-sl', '--Vertical', action='store_true', help="Vertical Device Architecture")


    opt = parser.add_argument_group('Optical Parameters', gooey_options={'columns': 2})

    opt.add_argument('-lw', '--laser_wavelength', default="470", type=float, help="Laser wavelength [nm]")
    opt.add_argument('-ref', '--laser_reference_file', widget="Dropdown",default = Ref_Files[-1], choices=Ref_Files,
                      type=str,
                      help="Pick the Laser Reference File")
    opt.add_argument('-abs', '--Absorption_Coefficient', default="0", type=float,
                      help="Absorption Coefficient [cm⁻¹] (Can be empty, if αd >> 1)")
    opt.add_argument('-Refl', '--Reflectance', default="20", type=float, help="Reflectance at WL [%]")


    opt3 = parser.add_argument_group('Corrections', gooey_options={'columns': 2})
    opt3.add_argument('-eb', '--Exciton_Binding_Energy', default="10", type=float,help="Exciton Binding Energy [meV]")
    opt3.add_argument('-k1', '--k1', type=float, help=r"SRH recombination constant [10⁶ x s⁻¹]")
    opt3.add_argument('-k2', '--k2',  default="0", type=float, help="Radiative recombination constant [10⁻¹⁰ x cm³s⁻¹]\n 3D ~ 1; 2D ~ 10")
    opt3.add_argument('-k3', '--k3',  default="0", type=float, help="Auger recombination constant [10⁻²⁸ x cm⁶s⁻¹]\n 3D ~ 0, 2D ~ 100")

    opt4 = parser.add_argument_group('Single Crystal Measures', gooey_options={'columns': 2})
    opt4.add_argument('-wid', '--Single_Crystal_Width', nargs=1, type=float, help="Width of Single Crystal [mm]")
    opt4.add_argument('-hei', '--Single_Crystal_Distance', nargs=1, type=float,
                      help="Distance of Electrodes in Single Crystal [mm]")
    opt4.add_argument('-thick', '--Single_Crystal_Thickness', nargs=1, type=float,
                      help="Single Crystal Thickness [mm]")


    args = parser.parse_args()

    args.directory = Path(args.data_path).resolve().parent
    args.short_name = Path(args.data_path).name
    args.cwd = Path.cwd()
    return args



"""Function Definitions"""

def unpack_Data(args):
    """ Unpack Data File"""
    time, *Data = np.loadtxt(args.data_path, unpack=True)
    Data = np.array(Data)

    """ Define Filters"""
    OD_Filter = (Data[:, 0]-1)*0.5 + args.Filter_Wheel

    """ Cut Data into Shape"""
    Data = np.delete(Data, 0, axis=1)  # Data in V
    Data_Diff = np.diff(Data[-1])
    Data_Diff[np.argmax(Data_Diff) > 1500] = np.nan
    time = time - time[np.argmax(Data_Diff)]


    ### Only use every second datapoint
    time = np.delete(time, 0)  # Time in ns
    data_filter = np.linspace(0, len(time) , int(len(time) / 2))
    time_cut = time[::2]
    Data_cut = np.zeros(shape=(len(Data),len(time_cut)))
    for i in range(len(Data)):
      Data_cut[i] = Data[i][::2]

    return time_cut, Data_cut, OD_Filter


def photocond_dec(args):
    """ Calculate Conductivity"""
    if args.Vertical == True:
        spacing = args.Thickness * 1e-9  # meters
        width = 1e-3  # meters
        depth = 1e-3  # meters
    elif args.Mask == "Single Crystal":
        spacing = args.Single_Crystal_Distance * 1e-3
        width = args.Single_Crystal_Width * 1e-3
        depth = args.Single_Crystal_Thickness * 1e-3
    else:
        Mask_Choice = args.Mask
        depth = np.asarray(args.Thickness) * 1e-9
        spacing = Mask_Values.loc[Mask_Choice, "Spacing"]
        width = Mask_Values.loc[Mask_Choice, "Width"]

    photo_cond = Data / (Resistance_System * (args.Voltage - Data)) * spacing / (depth * width)  *0.01

    return photo_cond, width, depth, spacing



def biexponential(params, time):
    A1 = params['A1'].value
    tau1 = params['tau1'].value
    y0 = params['y0'].value
    RC = params['RC'].value

    biexp = (1-np.exp(-time/RC))*(A1 * np.exp(-(time / tau1))+y0)

    return biexp



def residual(params, time, Data, args):
   resid = residual2(params, time, Data, args)
   return resid*time


def residual2(params, time, Data, args):
   model = biexponential(params, time)
   return (model - Data)



def Fitting_Cond(photo_cond, time, time_RC_max_old, tau_old, args):
    Data_fit = photo_cond
    #Data_fit[Data_fit < 0] = np.nan
    #if np.isnan(np.nanmean(Data_fit[100:400])):
    #    data_sort = np.sort(Data_fit[~np.isnan(Data_fit)])
    #    darkc = np.nanmedian(data_sort[0:100])
    #else:
    darkc = np.nanmean(Data_fit[100:400])
    dark_cond_value = np.nan_to_num(darkc)
    
    if dark_cond_value < 0:
        dark_cond_value = 0

    limit = np.where((time > Fit_range[0]) & (time < Fit_range[1]))

    Data_fit_cut = Data_fit[limit] - dark_cond_value
    Data_fit_cut[Data_fit_cut <= 0] = np.nan

    time_fit = time[limit]

    time_RC_max_new = time_fit[np.nanargmax(Data_fit_cut)]

    if time_RC_max_new < time_RC_max_old:
        time_RC_max_fit = time_RC_max_new
    else:
        time_RC_max_fit = time_RC_max_old

    params = Parameters()
    params.add('A1', value=0.001, min=0, max=1)
    params.add('tau1', value=tau_old, min=10)
    params.add('y0', value=dark_cond_value, min=0, vary=True)
    params.add('RC', value=time_RC_max_fit , min=3.74, max=time_RC_max_fit, vary=True)

    results_Model = minimize(residual, params, method='leastsq', args=(time_fit, Data_fit_cut, args), nan_policy='omit')
    photo_cond_zero_value = results_Model.params['A1'].value + results_Model.params['y0'].value + dark_cond_value
    delta_cond = results_Model.params['A1'].value + results_Model.params['y0'].value 

    tau_avg_value = results_Model.params['tau1'].value


    pseudo_k1_value = 1/(tau_avg_value*1e-9)

    RC_Fit = results_Model.params['RC'].value
    sigma_fit_value = (results_Model.params['A1'].value * np.exp(-(time / results_Model.params['tau1'].value))) + results_Model.params['y0'].value + dark_cond_value


    return dark_cond_value, photo_cond_zero_value, tau_avg_value, sigma_fit_value, pseudo_k1_value, RC_Fit, delta_cond, results_Model.params['y0'].value



def Exc_Density_Calc(OD_Filter, args):
    """ Unpack Ref Data File - The file contains powerdensities in mW cm-2"""
    wavelength, *Data = np.loadtxt(str(folder + '\TPC_Files\\'+args.laser_reference_file), unpack=True, skiprows=1)



    if np.shape(Data)[0] > 2:
        Ref_File = pd.DataFrame({'Wavelength':[wavelength],'PowerDens':[Data[0]],'Error':[Data[1]],'Correction':[Data[2]]})
    else:
        Ref_File = pd.DataFrame({'Wavelength':[wavelength],'PowerDens':[Data[0]]})


    """ Finding Powerdensity fo any given wavelength """
    wl = args.laser_wavelength

    if wl > Ref_File['Wavelength'][0].max():
        print('Error! Wavelength it out of the range of this reference file')
        exit()

    if wl in Ref_File['Wavelength'][0]:
        PowerDens = Ref_File['PowerDens'][0][np.where(Ref_File['Wavelength'][0] == wl)[0][0]]

        if np.shape(Data)[0] > 2:
            PowerError = Ref_File['Error'][0][np.where(Ref_File['Wavelength'][0] == wl)[0][0]]
            PowerCorrection = Ref_File['Correction'][0][np.where(Ref_File['Wavelength'][0] == wl)[0][0]]

    else:
        marker = (np.where(Ref_File['Wavelength'][0] > wl)[0][0]-1, np.where(Ref_File['Wavelength'][0] > wl)[0][0])
        Energies = 1240/np.array([Ref_File['Wavelength'][0][marker[0]],Ref_File['Wavelength'][0][marker[1]]])
        PowerDens = np.array([Ref_File['PowerDens'][0][marker[0]], Ref_File['PowerDens'][0][marker[1]]])

        wl_energy = 1240 / wl
        energy_ratio = np.sum(Energies) / wl_energy

        PowerDens = np.sum(PowerDens) / energy_ratio

        if np.shape(Data)[0] > 2:
            PowerError = np.array([Ref_File['Error'][0][marker[0]], Ref_File['Error'][0][marker[1]]])
            PowerCorrection = np.array([Ref_File['Correction'][0][marker[0]], Ref_File['Correction'][0][marker[1]]])
            PowerError = np.sum(PowerError) / energy_ratio
            PowerCorrection = np.sum(PowerCorrection) / energy_ratio

    Fluence = PowerDens / (frequency) * 10**(-OD_Filter) * 1000  # in uJ cm-2


    if bool(args.Absorption_Coefficient) or args.Absorption_Coefficient > 0:
        Absorption = (1 - np.exp(-args.Absorption_Coefficient * args.Thickness * 1e-7))*(1 - args.Reflectance / 100)  # absorbed light
    else:
        Absorption = (100 - args.Reflectance) / 100  # absorbed light

    Fluence_per_cm2 = Fluence/1e6 * (args.laser_wavelength * 1e-9 / (299792458 * 6.62607015e-34))# in cm-2


    Exc_Density = Fluence_per_cm2 * Absorption / (args.Thickness * 1e-7) # in cm-3



    return Fluence, Exc_Density, PowerDens, Absorption, Fluence_per_cm2




def gauss(t, mu, sig):
    return 1/(sig*np.sqrt(2*np.pi)) * np.exp(-(1/2)*((t-mu)/sig)**2)



# ODE definition
def df(c0,t,p, sigma, mu):
    ne = c0

    k1, k2, k3, ntot = p
    G = gauss(t, mu, sigma)
    dnedt = ntot*G -k1*ne - k2*ne**2 #- k3*ne**3
    return dnedt








# Define function for calculating free-carrier fraction
def a(n, E_B):
    # Solve the Saha equation to find the exciton density
    n_free = ()
    for m in n:
        T = 292  # temp in K
        mu_ex = 0.1 * 9.10938356e-31  # kg reduced mass
        k = 1.38064852e-23
        h = 6.62607004e-34
        e = 1.60217662e-19
        saha_const = (2*np.pi*mu_ex*k*T / h**2)**(3 / 2) * np.exp(-(e * E_B) / (k * T))

        phi_x = 2*np.sqrt(saha_const)/(np.sqrt(saha_const+4*m)+np.sqrt(saha_const))


        n_free = np.append(n_free, phi_x)

    return n_free




""" Main Part of Code"""
args = get_args()


time, Data, OD_Filter = unpack_Data(args)
time = time*1e9




photo_cond, width, depth, spacing = photocond_dec(args)

num_rows, num_cols = photo_cond.shape

dark_cond = []
photo_cond_zero = []
delta_cond_zero = []
tau_avg = []
pseudo_k1 = []
RC_value = []
y0_list = []
sigma_fit = pd.DataFrame(time, columns=['Time (ns)'])



" Defining Plots"

short_name = args.short_name.replace("0_01.txt", "")

if args.Sample_Name:
    sample_name = args.Sample_Name
else:
    sample_name = short_name


fig = plt.figure(figsize=(9, 9))
plt.subplots_adjust(wspace=0.28, hspace=0.35)
gs = gridspec.GridSpec(2, 2)
fig.suptitle(str('Transient Photoconductivity: '+ sample_name))
color1 = iter(cm.plasma(np.linspace(0, 1, num_rows)))
color2 = iter(cm.binary(np.linspace(1, 0.3, num_rows)))
c1 = pd.DataFrame()

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title('Photoconductivity Transients')
ax1.set_xlim(-40, Fit_range[1]*1.2)
ax1.set_xlabel('Time after Pulse [ns]')
ax1.set_ylabel('Photoconductivity [S cm⁻¹]')



ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title('Estimated  Lifetimes (Mono-exp)')
ax2.set_xlabel('Excitation Fluence [cm$^{-2}$]')
ax2.set_ylabel('Lifetime [ns]')

'''
ax3 = fig.add_subplot(gs[0, 1])
ax3.set_title('Estimated Free Carrier Fraction (Saha+Kinetik)')
ax3.set_xlabel('Time after Pulse [ns]')
ax3.set_ylabel('Free Carrier Fraction [-]')
ax3.set_ylim(1e-2, 1.)
ax3.set_xlim(-5,100)

ax4 = fig.add_subplot(gs[1, 1])
ax4.set_xlabel('Time after Pulse [ns]')
ax4.set_title('Corrections for one Excitation Density')
ax4.set_ylim(1e-2, 1.)
ax4.set_xlim(-5,100)
'''

ax5 = fig.add_subplot(gs[0, 1])
ax5.set_title('Conductivity')
ax5.set_xlabel('Excitation Fluence [cm$^{-2}$]')
ax5.set_ylabel('Conductivity [S cm⁻¹]')


ax6 = fig.add_subplot(gs[1, 1])
ax6.set_title('Estimated Mobility')
ax6.set_xlabel('Excitation Fluence [cm$^{-2}$]')
ax6.set_ylabel('Sum-Mobility [cm$^{2}$ (Vs)⁻¹]')



""" Estimate Initial Carrier Density """

Fluence, Exc_Density, PowerDens, Absorption, Fluence_per_cm2 = Exc_Density_Calc(OD_Filter, args)

ax2.set_xlim(min(Fluence_per_cm2)/100, max(Fluence_per_cm2)*10)
ax5.set_xlim(min(Fluence_per_cm2)/100, max(Fluence_per_cm2)*10)
ax6.set_xlim(min(Fluence_per_cm2)/100, max(Fluence_per_cm2)*10)


""" Fitting Photoconductivity"""
time_RC_max = 1000

tau_previous = 2000
for i in range(num_rows):

    dark_cond_value, photo_cond_zero_value, tau_avg_value, sigma_fit_value, pseudo_k1_value, RC_Fit, delta_cond, y0_val = Fitting_Cond(photo_cond[i], time, time_RC_max, tau_previous,  args)

    dark_cond = np.append(dark_cond, dark_cond_value)
    y0_list = np.append(y0_list, y0_val)
    tau_avg = np.append(tau_avg, tau_avg_value)
    RC_value = np.append(RC_value, RC_Fit)
    photo_cond_zero = np.append(photo_cond_zero,photo_cond_zero_value)
    delta_cond_zero = np.append(delta_cond_zero,delta_cond)
    pseudo_k1 = np.append(pseudo_k1,pseudo_k1_value)
    sigma_fit[str('Filter_'+ str(OD_Filter[i]))] = sigma_fit_value

    time_RC_max = RC_Fit
    tau_previous = tau_avg_value

    c1['Color'+ str(i)]= next(color1)

    ax1.semilogy(time, photo_cond[i], alpha = 0.6,c=c1['Color'+ str(i)], zorder = 10/(i+5), label=str("{:.2f}".format(Fluence[i]*1000) + " nJ cm⁻²"))
    ax1.semilogy(time,sigma_fit[str('Filter_'+ str(OD_Filter[i]))],c=next(color2), zorder =1000)
    ax1.semilogy(0, photo_cond_zero[i],'o',c=c1['Color'+ str(i)])


""" Averaging Dark Conductivity """
dark_cond_mean = np.mean(dark_cond)
dark_cond_std = np.std(dark_cond)

ax1.annotate(
        f'Dark Conductivity:\n {dark_cond_mean:.2e} +/- {dark_cond_std:.2e} S cm⁻¹',
        xy=(0.05, 0.05),backgroundcolor='white' ,xycoords='axes fraction',c='red',zorder=200)


dark_cond_no_zeros = dark_cond[np.where(dark_cond > 0)]
ax1.set_ylim(np.min(dark_cond_no_zeros)*0.1,np.max(photo_cond_zero)*10)
ax5.set_ylim(np.min(dark_cond_no_zeros)*0.1,np.max(photo_cond_zero)*10)
ax1.legend()

ax5.loglog(Fluence_per_cm2, delta_cond_zero, 'o', linestyle = '-',c='darkgrey', label='$\Delta\sigma_{\mathrm{photo}}(t=0)$')
ax5.loglog(Fluence_per_cm2, dark_cond, 'o', linestyle = '-',c='black', label='$\sigma_{\mathrm{dark}}$')



""" Estimation of Phi-Mobility """

Phi_SMu = delta_cond_zero/(Exc_Density*1.602176634e-19)

ax6.loglog(Fluence_per_cm2, Phi_SMu,'o', linestyle = '-',c='darkgrey', label="$\phi\Sigma\mu$")





""" Plotting Fit Values """

ax2.semilogx(Fluence_per_cm2, tau_avg,'o', linestyle = '-',c='darkgrey')
pseudo_k1_value = np.amax(pseudo_k1)
ax2.annotate(
        f'Pseudo-k₁: {pseudo_k1_value:.2e} s⁻¹',
        xy=(0.1, 0.1), xycoords='axes fraction',c='red')





""" Kinetik Equation Fitting """
def Kinetik_Fit(k2x, time,sigma_fit, pseudo_k1_value, dark_cond, y0_list, args):
    if args.k1:
        k1_const = args.k1 * 1e6  # in s-1
        pseudo_k1_marker = "set"
    else:
        k1_const = pseudo_k1_value
        pseudo_k1_marker = "pseudo"


    exciton_binding = args.Exciton_Binding_Energy / 1000  # in eV
    k2_const = 10**(k2x) * 1e-10  # in cm3 s-1
    k3_const = args.k3 * 1e-28  # in cm6 s-1

    I0 =  0

    sigma = pulse_fwhm / (2 * np.sqrt(2 * np.log(2)))
    mu = pulse_fwhm
    res = pd.DataFrame()

    ## Kinetik Model Preparation

    kinetik_range =[-10,1000]
    for i in range(num_rows):
        corr_param = [k1_const, k2_const, k3_const, Exc_Density[i]]

        num_pnts = 1000  # number of points in the simulation


        t = time[np.where((time > kinetik_range[0]) & (time < kinetik_range[1]))]*1e-9  # in ns
        sol = odeint(df, I0, t, args=(corr_param,sigma, mu))
        sol = sol.reshape(-1)
        res['OD-Filter'+str(OD_Filter[i])] = sol

    data_time_index = np.where(sigma_fit['Time (ns)'] >= kinetik_range[0])[0][0]
    
    """ Saha Equation Fitting """

    t_max = []
    free_carr_dens = np.zeros(shape=num_rows)
    SCond = np.zeros(shape=num_rows)

    a1_val = pd.DataFrame()
    sol_saha_df = pd.DataFrame()
    # Calculate free-carrier fraction as a function of charge density
    # and exciton binding energy
    for i in range(num_rows):
        sol = res['OD-Filter' + str(OD_Filter[i])]

        a1 = a(sol * 1e6, exciton_binding)  # Saha's equation is calculated in m3 not cm3, hence 1e6
        a1_val['OD-Filter'+str(OD_Filter[i])] = a1
        sol_saha = np.nan_to_num(sol * a1)
        sol_saha_df['OD-Filter'+str(OD_Filter[i])] = sol_saha

        t_max = np.append(t_max, t[np.nanargmax(sol_saha)] * 1e9)
        free_carr_dens[i] = np.median(sol_saha[np.where(np.isclose(t*1e9,t_max[i]))])#np.nanmax(sol_saha)
        SCond[i] = sigma_fit[str('Filter_'+ str(OD_Filter[i]))][data_time_index+np.nanargmax(sol_saha)]-dark_cond[i]

    SMu = SCond / (free_carr_dens * 1.602176634e-19)
    
    return SMu, t_max, free_carr_dens, SCond, pseudo_k1_marker


def residual_kin(k2x, time, sigma_fit,pseudo_k1_value, dark_cond, y0_list, args):

    SMu, _, _, _, _ = Kinetik_Fit(k2x, time, sigma_fit, pseudo_k1_value, dark_cond, y0_list, args)
    SMu_std = np.std(SMu)


    return SMu_std


results_Model_kin = minimize_scalar(residual_kin, bounds=(-3, 5), method='bounded', args=(time, sigma_fit, pseudo_k1_value, dark_cond, y0_list, args))




""" Calculating Corrected Mobility/Conductivity """
if args.k2 > 0:
    k2_val = np.log10(args.k2)
else:
    k2_val = results_Model_kin.x

SMu, t_max, free_carr_dens, SCond, pseudo_k1_marker = Kinetik_Fit(k2_val, time, sigma_fit, pseudo_k1_value, dark_cond, y0_list, args)#SCond/(free_carr_dens*1.602176634e-19)



SMu_avg = np.median(SMu)
SMu_q1 = abs(np.quantile(SMu,0.25) - np.median(SMu))
SMu_q3 = abs(np.quantile(SMu,0.75) - np.median(SMu))

ax6.loglog(Fluence_per_cm2,SMu,'o', linestyle = '-',c='darkblue', label="$\Sigma\mu$")
ax6.set_ylim(min(Phi_SMu)/10,max(SMu)*10)
ax6.annotate(
        f'Σμ (med): {SMu_avg:.2f} +/- {SMu_q3:.2f} cm²(Vs)⁻¹\n'
        f'OD_set: {args.Filter_Wheel:.1f}\n'
        f'k₁: {pseudo_k1_marker}',
        xy=(0.1, 0.15), xycoords='axes fraction',c='darkblue')


ax5.loglog(Fluence_per_cm2, SCond,'o', linestyle = '-',c='darkblue', label='$\Delta\sigma_{\mathrm{photo}}(t=t_{\mathrm{max}})$')
ax5.legend()
ax6.legend()



""" Saving the Fitted Values """

index_names = []
index_names.append('units')

i = 0
while i < num_rows:
    index_names.append(str(str(i) + '_OD-Filter'+str(OD_Filter[i])))

    i+= 1


Fit_Values = pd.DataFrame(index=[index_names],columns=['Fluence','SMu', 'Cond_tmax','Phi_max','ExcDens','Phi_Mu','Cond_t0','dark_Cond','k_1'])

Fit_Values['ExcDens']['units'] = 'cm-3'
Fit_Values['Fluence']['units'] = 'cm-2'
Fit_Values['SMu']['units'] = Fit_Values['Phi_Mu']['units']=  'cm2/Vs'
Fit_Values['Cond_tmax']['units'] = Fit_Values['Cond_t0']['units'] = 'S/cm'
Fit_Values['k_1']['units'] = 's-1'
Fit_Values['Phi_max']['units'] = 'a.u.'



i = 0
while i < num_rows:
    index_name = str(str(i) + '_OD-Filter'+str(OD_Filter[i]))
    Fit_Values.loc[index_name, 'SMu'] = SMu[i]
    Fit_Values.loc[index_name, 'Cond_tmax'] = SCond[i]
    Fit_Values.loc[index_name, 'Phi_max'] = free_carr_dens[i]/Exc_Density[i]
    Fit_Values.loc[index_name, 'ExcDens'] = Exc_Density[i]
    Fit_Values.loc[index_name, 'Fluence'] = Fluence_per_cm2[i]
    Fit_Values.loc[index_name, 'Phi_Mu'] = Phi_SMu[i]
    Fit_Values.loc[index_name, 'Cond_t0'] = delta_cond_zero[i]
    Fit_Values.loc[index_name, 'k_1'] = pseudo_k1[i]
    Fit_Values.loc[index_name, 'dark_Cond'] = dark_cond[i]
    i += 1


exciton_binding = args.Exciton_Binding_Energy / 1000
if args.k1:
    k1_const = args.k1 * 1e6  # in s-1
    pseudo_k1_marker = "set"
else:
    k1_const = pseudo_k1_value
    pseudo_k1_marker = "pseudo"

k2_const = 10 ** k2_val * 1e-10
k3_const = args.k3 * 1e-28  # in cm6 s-1

if args.Absorption_Coefficient == 0:
    absorption_coeff_est = 1/(args.Thickness*1e-7)
else:
    absorption_coeff_est = -np.log(1-Absorption/(1-args.Reflectance/100))/(args.Thickness*1e-7)

""" Saving the Fitting Parameters """
Fit_Params = pd.DataFrame(index=['SampleName','Wavelength (nm)','Thickness (nm)','Vapp (V)','-','Mask','width (m)', 'depth (m)', 'spacing (m)','Fit_start (ns)','Fit_end (ns)','--','Laser Power Density (mW/cm2)','Absorption Coeff. (cm-1)','Reflectance (%)','---','E_B (meV)','k1 (s-1)','k2 (cm3 s-1)','k3 (cm6 s-1)'],
                          data=[sample_name, int(args.laser_wavelength), int(args.Thickness),args.Voltage,'_________',args.Mask,"{:.2e}".format(width), "{:.2e}".format(depth), "{:.2e}".format(spacing),Fit_range[0],Fit_range[1],'_________',"{:.2f}".format(PowerDens),"{:.2e}".format(absorption_coeff_est),args.Reflectance,'_________',int(exciton_binding*1000),str("{:.2e}".format(k1_const)+" ("+pseudo_k1_marker+")"),"{:.2e}".format(k2_const),"{:.2e}".format(k3_const)])


print(str('k2 est. = ' + "{:.2e}".format(k2_const)) +" cm-3 s-1")


""" Saving Photoconductivity Data"""
index_names[0] = 'Time'
index_names = index_names + ['Fit_' + s for s in index_names]
index_names.remove('Fit_Time')


Data_save = pd.DataFrame(columns=index_names)


Data_save['Time'] = time

i = 0
while i < num_rows:
    index_name = str('OD-Filter'+str(OD_Filter[i]))
    Data_save[index_name] = photo_cond[i]
    Data_save[str('Fit_' + index_name)] = sigma_fit[str('Filter_'+ str(OD_Filter[i]))]
    i += 1


""" Actual Saving of Files"""
Fit_Values.to_csv(str(PurePath(args.directory).joinpath(str(short_name + '_fit-values.dat'))), sep='\t', index= True, mode='w')
Fit_Params.to_csv(str(PurePath(args.directory).joinpath(str(short_name + '_fit-params.txt'))), sep='\t', index= True, mode='w')
Data_save.to_csv(str(PurePath(args.directory).joinpath(str(short_name + '_processed.txt'))), sep='\t', mode='w')
plt.savefig(str(PurePath(args.directory).joinpath(str(short_name + '.png'))), format='png', dpi=100)

plt.show()






