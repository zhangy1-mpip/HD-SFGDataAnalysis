# import libraries

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from datetime import datetime
from scipy.interpolate import PchipInterpolator as Pchi
from scipy.constants import c
from scipy.signal import savgol_filter
from scipy.signal import medfilt

# ========== References ==========  
# 1. Takeshita, N.; Okuno, M.; Ishibashi, T. A. Molecular conformation of DPPC phospholipid Langmuir and Langmuir-Blodgett monolayers studied by heterodyne-detected vibrational sum frequency generation spectroscopy. Phys Chem Chem Phys 2017, 19 (3), 2060-2066.
# 2. arXiv:2312.15267v1 for the discussion of Planck-taper filter as spectra filter.
# 3. Wang, Y.; Tang, F.; Yu, X.; Chiang, K. Y.; Yu, C. C.; Ohto, T.; Chen, Y.; Nagata, Y.; Bonn, M. Interfaces govern the structure of angstrom-scale confined water solutions. Nat Commun 2025, 16 (1), 7288.
# ================================ 

# ========== define of the variables ==========
# reference material: The reference used to analyze the data. The value is either 'gold', 'zpz' (z-cut quartz) or 'D2O'.
RefMaterial='zqz'
# Media1 and 2 definition: Media1 is solid or air, Media2 is water or D2O
# vis and IR light transmit from media 1 to media 2
# Media 1 and 2 should be in the list of optical parameters
# Media 2 is "H2O" or "D2O"
# if Media1 is air, and Media2 is water, 
Media1 = 'air'
Media2 = 'H2O'
# file path of parameter of the reference materials
ParaPath = r'D:\Zhang Yu Data\MPI-P\SFG\Script\Parameters' 
# folder of the experiment files, including backgrounds, reference and experiment data
# REMEMBER TO CHANGE THIS
FolderPath = r'D:\Zhang Yu Data\MPI-P\SFG\20260310\12_k3_0,5M_pH11'

# experiment conditions
# reference exposure time
RefExposure = 1
# sample exposure time
SamExposure = 300

# vis wavelength
VisWavelength = 805.5
# incident angles in air in degree
VisIncidentAngle = 45
IRIncidentAngle = 45

# phase correction for the phase drift
PhaseCorr = 3

# the window for plot
Frequency_min = 2800
Frequency_max = 3800
Amplitude_min = -0.1
Amplitude_max = 0.1
ChiTwoFig_min = -20
ChiTwoFig_max = 20
# figure size in inch
figwidth = 12
figheight = 9

# Fresnel correction parameters
# ModelType is the model chosen for the calculation of n^{\prime} (interfacial refractive index)
# lorentz type: n' = n_j
# slab type: n' = (n_i^2(n_i^2+5)/(4n_i^2+2))^(1/2)
ModelType = 'lorentz' 
PolarizationType = 'psp'

# parameters to locate the SFG/LO delay time T0 and reflection delay time T1
n_skip=20   # skip the data points for the optic rectification
frac=0.1    # defined the threshold of the peak decay
margin=5    # add some safety points for the reflection peak search

# pre-fft Plank-taper filter configuration
# go to function EstimateFringeWidth and PlanckTaperFilter to see the definition
PlanckEps = 0.2 # the sharpness of ramping area
NoofPeaks = 3   # no. of peaks treated in the left
LRRatio= 2   # no. of peaks treated on the right/no. of peaks treated on the left
PlanckFilterRatio_Max = 0.15    # maximum 15% data points are allowed to be treated
PlanckTaperFilterMin = 0.02     # set the minimum value of the filter, avoiding 0 as divider

# pre-ifft Happ-Genzel filter TimeDomainFilter in time domain
# parameters for SFG peaks; T0 is the SFG/LO delay time
T0LBoundary = 0.6     # filter starts at LBoundary*T0, 
T0RBoundary = 1.2     # filter ends at LBoundary*T0, T0 is the SFG/LO delay time
HGWidthT0Ratio = 0.2  # [width of the ramping area]/T0, T0 is the SFG/LO delay time
# for reflection peaks, better not to use it; T1 is the reflection delay time
T1LRel = 0.0        # filter starts at T1-T1LRelT1*T0
T1RRel = 0.0        # filter ends at T1+T1LRelT1*T0
# T1 valley is used to filter out the reflection peaks
eps = 0.5                           # base of the noise on the right side of time axis, eps=1 means no compression of noise
NoiseSamplingStartinPs = 5.5        # starting time of the noise sampling
NoiseHGRampinginPs = 0.2            # ramping width of the Happ-Genzel window for noise
T1ValleyHalfRatio = 0.6             # width of valley = T1ValleyHalfRatio x T0 plateau width x 2
ValleyHGRatio = 0.3                 # width of HG ramping / width of T1 valley

# Savitzky–Golay smoothing parameters
SG_WINDOW = 21       # has to be odd and <= number of frequencies
SG_POLY = 3         # order of smoothing: =1: linear interpolation, =2 parabola interpolation, =3 cubic-parabola-linear combination 

# Spike removal parameters
SPIKE_REMOVE = True
SPIKE_KERNEL = 131
SPIKE_THRESHOLD_FACTOR = 5.0
SPIKE_MIN_FILES = 3
SPIKE_SAFE_EPS = 1e-12
# ========== file manipulation functions ==========

# read the asc raw data file and sort the file according to the different
    # input FolderPath: the absolute path of the data storage folder 
    # return Lists: the dictionary contains 6 lists: 1. GoldBg: Gold background 2. GoldRef: Gold references 3. WaterBg: Water experiment background 4. WaterExp: Water experiment data files 5. D2ORef: D2O reference data 6. D2OExp: D2O experiment data.
    # change the keywords "gold" to "zqz" for the quartz reference.
def SortDataFile (FolderPath):
    SortResult = {
         "GoldBg": [],
         "GoldRef": [],
         "zqzBg": [],
         "zqzRef": [],
         "WaterBg": [],
         "WaterExp": [],
         "D2OBg": [],
         "D2ORef":[]
     }
    
    for filename in os.listdir(FolderPath):
        
        name = filename.lower()
        # find all the .asc files
        if name.endswith(".csv"):
            has_bg = "bg" in name
            has_water = ("water" in name) or ("h2o" in name)
            has_zqz = "zqz" in name
            has_gold = "gold" in name
            has_D2O = "d2o" in name
        
            # sorting the files
            if has_gold and has_bg: SortResult["GoldBg"].append(filename)
            elif has_gold and not has_bg: SortResult["GoldRef"].append(filename)
            elif has_water and has_bg: SortResult["WaterBg"].append(filename)
            elif has_water and not has_bg: SortResult["WaterExp"].append(filename)
            elif has_D2O and has_bg: SortResult["D2OBg"].append(filename)
            elif has_D2O and not has_bg: SortResult["D2ORef"].append(filename)
            elif has_zqz and has_bg: SortResult["zqzBg"].append(filename)
            elif has_zqz and not has_bg: SortResult["zqzRef"].append(filename)
    
    return SortResult

# ========== save the analysis parameters ==========
def save_analysis_parameters(output_dir):
    script_path = os.path.abspath(__file__)

    start_marker = "# ========== define of the variables =========="
    end_marker   = "#========== file manipulation functions =========="

    with open(script_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    in_block = False
    param_lines = []

    for line in lines:
        if start_marker in line:
            in_block = True
            param_lines.append(line)
            continue
        if end_marker in line and in_block:
            param_lines.append(line)
            break
        if in_block:
            param_lines.append(line)

    if not param_lines:
        raise RuntimeError("Parameter block not found. Check marker strings.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"parameter_{timestamp}.txt"

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(param_lines)

    print(f"[INFO] Analysis parameters saved to: {out_path}")

# ========== scientific calculation functions ==========

# read and create interpolate function of the complex refractive index
    # SumFrequency: the wavenumber of the SFG signal (\omega_vis+\omega_ir)
    # Media: material of the media (search keywords)
    # EstimateNFun: interpolation function of the complex refractive index
def CreateRefractionIndexFun (ParaPath, Media):
    # Filepath: the path of the text file for the specific media
    FilePath = os.path.join(ParaPath, f"complex_refractive_index_of_{Media}.txt")
    # file existence check
    if not os.path.isfile(FilePath):
        raise FileNotFoundError(f"[Exp Config ERROR] Refractive index file not found for Media='{Media}'.\n"
        f"Please check the Media1 or Media2 name.")
    print(f'[Exp Config INFO] {Media} data in {FilePath} is used.')

    # read the parameters
    data = np.genfromtxt(FilePath, delimiter =',')

    Wavenumbers_cm = data[:, 0]
    n = data[:, 1]
    kappa = data[:, 2]
    N = n + 1j*kappa
    
    # arrange the refractive index using the increasing wavenumber order for interpolate
    idx = np.argsort(Wavenumbers_cm)
    Wavenumbers_cm = Wavenumbers_cm[idx]
    n = n[idx]
    kappa = kappa[idx]
    
    # create the interpolation function for real part
    fun_real = Pchi(np.asarray(Wavenumbers_cm), np.asarray(n))
    # create the interpolation function for imag part
    fun_imag = Pchi(np.asarray(Wavenumbers_cm), np.array(kappa))
        
    return fun_real, fun_imag

# read the wavelength column in the file and create the wavenumber for SF light (\omega_IR+\omega_vis) and IR
    # FilePath: the absolute file path of the file to create the wavenumber axis
    # visWavelength: the visible light wavelength in nm 
    # SFGFrequency, IRFrequency: frequencies of SFG and IR in cm-1
def CreateFrequencyAxis (FilePath, visWavelength):
    DataRead_Temp = np.genfromtxt(FilePath, delimiter=',')
    SFGWavelength = DataRead_Temp[:, 0]

    # the frequencies are in cm-1
    SFGFrequency = 1E7/SFGWavelength
    IRFrequency = 1E7/SFGWavelength - 1E7/visWavelength

    return  SFGFrequency, IRFrequency 

# use the sort result to subtract the backgrounds
    # Folder: The folder where experiment data files and backgrounds are  
    # BgGroup: the list of background file names
    # ExpDataGroup: the list of raw experiment data file names (both reference and measured data are fine)
    # SpectraBgSub: mean, background subtracted reference spectra
def BgSubtract (Folder, BgGroup, ExpDataGroup):
    # average the background
    BgData = []
    for Bgs in BgGroup:
        BgsPath = os.path.join(Folder, Bgs)
        BgData_Temp = np.genfromtxt(BgsPath, delimiter=',')
        BgData.append(BgData_Temp[:,1])

    BgMean = (np.stack(BgData, axis=0)).mean(axis=0)
    # for every file in experiment data group, subtract the background
    
    ExpDataBgSub_Temp = []
    for Exp in ExpDataGroup:
        ExpsPath = os.path.join(Folder, Exp)
        ExpDataTemp = np.genfromtxt(ExpsPath, delimiter=',')
        Intensity = ExpDataTemp[:,1]

        # subtract the background for each file
        IntensityBgSub = Intensity - BgMean
        ExpDataBgSub_Temp.append(IntensityBgSub)

    SpectraBgSub = np.stack(ExpDataBgSub_Temp, axis=0)
    return SpectraBgSub

# remove the isolated spikes in the signal
def RemoveSpikeOutliers (Spectra, Reference, KernelSize, ThresholdFactor, MinFiles, SpikeSafeEps):
    spectra = np.asarray(Spectra, dtype = float).copy()
    
    # check if the spectra is in correct shape
    if spectra.ndim != 2:
        raise ValueError('[Spike Removal INFO] Spectra is not 2D array!')

    # check if there are enough files for analysis
    n_spec, _ = spectra.shape
    if n_spec < MinFiles:
        print(f'[Spike Removal INFO] Only {n_spec} spectra, skip spike removal.')

    # ensure the kernel size is correct
    if KernelSize % 2 == 0:
        KernelSize = KernelSize + 1

    # check if there are enough reference
    if Reference is None:
        Reference = Spectra[0]
    else:
        Reference = np.asarray(Reference, dtype = float)

    # remove the spike
    SpectraDelBack = []
    SpectraBack = []
    for i in range(n_spec):
        DiffToRef = spectra[i] - Reference
        Baseline_i = medfilt(DiffToRef, KernelSize)
        SpectraDelBack.append(Spectra[i] - Baseline_i)
        SpectraBack.append(Baseline_i)

    SpectraDelBack = np.asarray(SpectraDelBack)
    SpectraBack = np.array(SpectraBack)

    PointwiseMedian = np.median(SpectraDelBack, axis = 0)
    PointwiseStd = np.std(SpectraBack, axis = 0)
    denom = np.maximum(np.abs(PointwiseMedian), SpikeSafeEps)

    RelativeScatter = PointwiseStd/denom
    ScatterFloor = medfilt(RelativeScatter, KernelSize)
    ScatterFloor = np.maximum(ScatterFloor, SpikeSafeEps)

    Cleaned = SpectraDelBack.copy()
    NumDeleted = 0
    for i in range(n_spec):
        deviation = np.abs(Cleaned[i]-PointwiseMedian)/denom
        SpikeMask = deviation > (ThresholdFactor*ScatterFloor)
        NumDeleted = NumDeleted+np.count_nonzero(SpikeMask)
        Cleaned[i,SpikeMask] = PointwiseMedian[SpikeMask]

    Restored = Cleaned + SpectraBack
    print(f'[Spike Removal INFO] Remove {NumDeleted} spike points from {n_spec} spectra.')
    return Restored


# calculate the Fresnel coefficients T and R
    # n_i, n_j: frequency-dependent refractive indices in medium i and j
    # theta_i, theta_j: angles in medium i and j
def _cos_theta_t(n_i, n_t, theta_i):
    # complex Snell: sin(theta_t) = (n_i/n_t) sin(theta_i)
    s_t = (n_i / n_t) * np.sin(theta_i)
    c_t = np.sqrt(1 - s_t*s_t)  # complex sqrt
    # choose physical branch: enforce Im(kz) >= 0  (decaying into medium)
    kz = n_t * c_t
    c_t = np.where(np.imag(kz) < 0, -c_t, c_t)
    return c_t
# s-polarized
def FresnelR_sT_s(n_i, n_t, theta_i):
    c_i = np.cos(theta_i)
    c_t = _cos_theta_t(n_i, n_t, theta_i)
    R_s = (n_i*c_i - n_t*c_t) / (n_i*c_i + n_t*c_t)
    T_s = (2*n_i*c_i) / (n_i*c_i + n_t*c_t)
    return R_s, T_s
# p-polarized
def FresnelR_pT_p(n_i, n_t, theta_i):
    c_i = np.cos(theta_i)
    c_t = _cos_theta_t(n_i, n_t, theta_i)
    R_p = (n_t*c_i - n_i*c_t) / (n_t*c_i + n_i*c_t)
    T_p = (2*n_i*c_i) / (n_t*c_i + n_i*c_t)
    return R_p, T_p

# Fresnel Factor L_{ii}
# compute cos(theta_t) from complex Snell's law in a branch-safe way to use the complex refractive index and avoid unreal divergence (Enforce a physical branch by requiring Im(k_z)>=0, where kz~n_t*cos(theta_t)).
    # n_i, n_t: complex refractive indices of medias
    # theta_i: incident angle
def _cos_theta_t_branch_safe(n_i, n_t, theta_i):

    s_t = (n_i / n_t) * np.sin(theta_i)
    c_t = np.sqrt(1 - s_t * s_t)  # complex sqrt

    kz = n_t * c_t
    c_t = np.where(np.imag(kz) < 0, -c_t, c_t)
    return c_t
# calculate FresnelFactors
# n_i, n_j, theta_i are the same from the previous function _cos_theta_t_branch_safe
# theta_j maybe a complex number from Snell's law.
# ModelType = lorentz or slab
def FresnelFactors (n_i, n_j, theta_i, theta_j, ModelType):
    # dtype safety: keep complex
    n_i = np.asarray(n_i, dtype=complex)
    n_j = np.asarray(n_j, dtype=complex)
    theta_i = np.asarray(theta_i, dtype=complex)

    # Interfacial model
    mt = ModelType.lower()
    if mt == 'lorentz':
        n_prime = n_j
    elif mt == 'slab':
        # keep complex-safe; if you intended this only for real n_i, this still works mathematically
        n_prime = np.sqrt((n_i**2)*((n_i**2 + 5)/(4*n_i**2+2)))
    else:
        raise ValueError('Interfacial model error: ModelType is not defined')

    # cos(theta_i)
    c_i = np.cos(theta_i)

    # Prefer complex Snell to get cos(theta_j) when n_j is complex (more correct),
    # otherwise fall back to passed theta_j
    if np.any(np.abs(np.imag(n_j)) > 0):
        c_j = _cos_theta_t_branch_safe(n_i, n_j, theta_i)
    else:
        theta_j = np.asarray(theta_j, dtype=np.complex128)
        c_j = np.cos(theta_j)

    # L factors (complex)
    denom = (n_i*c_j+n_j*c_i)
    L_xx = 2*n_i*c_j/denom
    L_yy = 2*n_i*c_i/denom
    L_zz = 2*n_j*c_i/(n_j*c_i+ n_i*c_j)*(n_i/n_prime)**2

    return L_xx, L_yy, L_zz

# Generate the absolute amplitude calibration standard for air/water or solid/water interface
    # ParaPath: folder of parameters of the materials
    # Media1: air or solid
    # VisIncidentAngle: incident angle of vis light in air
    # IRIncidentAngle； incident angle of IR in air
    # VisWavenumber, IRWavenumber, SFGWavenumber: uniformly distributed wavenumbers in vis, IR and SFG regime
    # ModelType: Lorentz or Slab, see Yongkang Wang, et al. Nat Commun. 2025, in SI discussion S8
    # PolarizationType: ssp, psp or ppp
def ComputeChiTwoNR(ParaPath, Media1, VisIncidentAngle, IRIncidentAngle, VisWavenumber, IRWavenumber, SFGWavenumber, ModelType, PolarizationType):
    # refractive index of air
    n_air_vis = np.ones_like(VisWavenumber, dtype=float)
    n_air_IR  = np.ones_like(IRWavenumber, dtype=float)
    n_air_SFG = np.ones_like(SFGWavenumber, dtype=float)
    # ***** air/water: z-cut quartz absolute reference *****
    if Media1.lower() == 'air':
        print('[Exp Config INFO] Air/water interface using z-cut quartz as amplitude reference for absolute value.')
        ChiTwo_q = 8e-13   # [m V^-1]
        l_c = 4.3e-8  # [m]
        # refractive index of z-cut quartz (real part only for angles)
        fun_real_zqz, fun_imag_zqz = CreateRefractionIndexFun(ParaPath, 'zqz')
        # vis
        n_zqz_vis = fun_real_zqz(VisWavenumber)
        arg_vis_zqz = n_air_vis / n_zqz_vis * np.sin(VisIncidentAngle)
        arg_vis_zqz = np.clip(arg_vis_zqz, -1.0, 1.0)
        theta_vis_zqz = np.arcsin(arg_vis_zqz)
        # IR
        n_zqz_IR = fun_real_zqz(IRWavenumber)
        arg_IR_zqz = n_air_IR / n_zqz_IR * np.sin(IRIncidentAngle)
        arg_IR_zqz = np.clip(arg_IR_zqz, -1.0, 1.0)
        theta_IR_zqz = np.arcsin(arg_IR_zqz)
        # SFG
        n_zqz_SFG = fun_real_zqz(SFGWavenumber)
        sin_theta_SFG_air = (VisWavenumber*np.sin(VisIncidentAngle)+IRWavenumber*np.sin(VisIncidentAngle))/SFGWavenumber
        sin_theta_SFG_air = np.clip(sin_theta_SFG_air, -1.0, 1.0)
        theta_SFG_air = np.arcsin(sin_theta_SFG_air)
        sin_theta_SFG_zqz = np.clip(sin_theta_SFG_air * n_air_SFG / n_zqz_SFG, -1.0, 1.0)
        theta_SFG_zqz = np.arcsin(sin_theta_SFG_zqz)
        # Fresnel factors
        L_xx_SFG_zqz, L_yy_SFG_zqz, L_zz_SFG_zqz = FresnelFactors(n_air_SFG, n_zqz_SFG, theta_SFG_air, theta_SFG_zqz, ModelType)
        L_xx_vis_zqz, L_yy_vis_zqz, L_zz_vis_zqz = FresnelFactors(n_air_vis, n_zqz_vis, VisIncidentAngle, theta_vis_zqz, ModelType)
        L_xx_IR_zqz,  L_yy_IR_zqz,  L_zz_IR_zqz  = FresnelFactors(n_air_IR, n_zqz_IR, IRIncidentAngle, theta_IR_zqz, ModelType)
        
        # polarization-dependent Fresnel product
        pol = PolarizationType.lower()
        print(f'[Exp Config INFO] Polarization type is {pol}.')

        if pol == 'ssp':
            # from Yongkang Wang, et al. Nat Commun. 2025 SI Eq. S17.
            F_zqz = L_yy_SFG_zqz*L_yy_vis_zqz*L_xx_IR_zqz*np.cos(IRIncidentAngle)
        elif pol == 'psp':
            # from Yan, E. C.; Fu, L.; Wang, Z.; Liu, W. Biological macromolecules at interfaces probed by chiral vibrational sum frequency generation spectroscopy. Chem Rev 2014, 114 (17), 8471-8498.
            F_zqz = (L_zz_SFG_zqz*L_yy_vis_zqz*L_xx_IR_zqz*np.sin(theta_SFG_air)*np.cos(IRIncidentAngle))
        elif pol == 'ppp':
            # from Yu, C. C.; Seki, T.; Chiang, K. Y.; Tang, F.; Sun, S.; Bonn, M.; Nagata, Y. Polarization-Dependent Heterodyne-Detected Sum-Frequency Generation Spectroscopy as a Tool to Explore Surface Molecular Orientation and Angstrom-Scale Depth Profiling. J Phys Chem B 2022, 126 (33), 6113-6124.
            F_zqz = (L_xx_SFG_zqz*L_xx_vis_zqz*L_xx_IR_zqz*np.cos(theta_SFG_air)*np.cos(VisIncidentAngle))
        else:
            raise ValueError('[Fresnel ERROR] Polarization type not supported for quartz reference.')
        # non-resonant effective chi(2) for air/water interface amplitude absolute value
        ChiTwoNR = 2.0*F_zqz*ChiTwo_q*l_c  # [m^2 V^-1]
    else:
    # ***** solid/water: CaF2/gold absolute reference *****
        print('[Exp Config INFO] Solid/water interface CaF2/gold as amplitude reference for absolute value.')
        ProductCaF2Gold = 3.2e-20  # [m^2 V^-1], Yongkang Wang et al. Nat Commun. 2025, Eq. S25 in the SI.
        # refractive indices (complex) for gold and CaF2
        # generate the interpolate function
        fun_real_gold, fun_imag_gold = CreateRefractionIndexFun(ParaPath, 'gold')
        fun_real_CaF2, fun_imag_CaF2 = CreateRefractionIndexFun(ParaPath, 'CaF2')
        # calculate the refractive indics
        CompN_SFG_gold = fun_real_gold(SFGWavenumber) + 1j*fun_imag_gold(SFGWavenumber)
        CompN_vis_gold = fun_real_gold(VisWavenumber) + 1j*fun_imag_gold(VisWavenumber)
        CompN_IR_gold  = fun_real_gold(IRWavenumber)  + 1j*fun_imag_gold(IRWavenumber)

        CompN_SFG_CaF2 = fun_real_CaF2(SFGWavenumber) + 1j*fun_imag_CaF2(SFGWavenumber)
        CompN_vis_CaF2 = fun_real_CaF2(VisWavenumber) + 1j*fun_imag_CaF2(VisWavenumber)
        CompN_IR_CaF2  = fun_real_CaF2(IRWavenumber)  + 1j*fun_imag_CaF2(IRWavenumber)

        # vis/IR angles in CaF2 (Snell, real n)
        arg_vis_CaF2 = n_air_vis / np.real(CompN_vis_CaF2)*np.sin(VisIncidentAngle)
        arg_vis_CaF2 = np.clip(arg_vis_CaF2, -1.0, 1.0)
        theta_vis_CaF2 = np.arcsin(arg_vis_CaF2)

        arg_IR_CaF2 = n_air_IR/np.real(CompN_IR_CaF2)*np.sin(IRIncidentAngle)
        arg_IR_CaF2 = np.clip(arg_IR_CaF2, -1.0, 1.0)
        theta_IR_CaF2 = np.arcsin(arg_IR_CaF2)
        # vis/IR angles in gold
        theta_vis_gold = np.arcsin(np.clip(np.real(CompN_vis_CaF2)/np.real(CompN_vis_gold)*np.sin(theta_vis_CaF2), -1.0, 1.0))
        theta_IR_gold  = np.arcsin(np.clip(np.real(CompN_IR_CaF2) /np.real(CompN_IR_gold) *np.sin(theta_IR_CaF2),  -1.0, 1.0))
        # SFG angle in CaF2 (phase matching, real n)
        arg_SFG_CaF2 = (np.real(CompN_vis_CaF2)*np.sin(theta_vis_CaF2)+np.real(CompN_IR_CaF2)*np.sin(theta_IR_CaF2))/np.real(CompN_SFG_CaF2)
        arg_SFG_CaF2 = np.clip(arg_SFG_CaF2, -1.0, 1.0)
        theta_SFG_CaF2 = np.arcsin(arg_SFG_CaF2)

        # SFG angle in gold (Snell, real n)
        arg_SFG_gold = np.real(CompN_SFG_CaF2)/np.real(CompN_SFG_gold)*np.sin(theta_SFG_CaF2)
        arg_SFG_gold = np.clip(arg_SFG_gold, -1.0, 1.0)
        theta_SFG_gold = np.arcsin(arg_SFG_gold)

        # r* at CaF2/gold for SFG
        rStarCaF2Gold, _ = FresnelR_sT_s(CompN_SFG_CaF2, CompN_SFG_gold, theta_SFG_CaF2)
        ChiTwoeffCaF2Gold = ProductCaF2Gold/rStarCaF2Gold
        # calculate the Fresnel factors for CaF2/gold
        # Fresnel factors for SFG (CaF2 -> gold)
        L_xx_SFG_CaF2Gold, L_yy_SFG_CaF2Gold, L_zz_SFG_CaF2Gold = FresnelFactors(CompN_SFG_CaF2, CompN_SFG_gold, theta_SFG_CaF2, theta_SFG_gold, ModelType)
        L_xx_vis_CaF2Gold, L_yy_vis_CaF2Gold, L_zz_vis_CaF2Gold = FresnelFactors(CompN_vis_CaF2, CompN_vis_gold, theta_vis_CaF2, theta_vis_gold, ModelType)
        L_xx_IR_CaF2Gold, L_yy_IR_CaF2Gold, L_zz_IR_CaF2Gold = FresnelFactors(CompN_IR_CaF2, CompN_IR_gold, theta_IR_CaF2, theta_IR_gold, ModelType)
        F_ssp_CaF2Gold = L_yy_SFG_CaF2Gold*L_yy_vis_CaF2Gold*L_zz_IR_CaF2Gold*np.cos(theta_IR_CaF2)
        # pseudo tensor element (model-defined)
        # using the assumptions of: a) assuming isotropic surface, b) assuming the CaF2/Gold interface also can be described using \chi^{(2)}_{eff} = F_{ijk}*\chi^{(2)}_{ijk} c) all the signal is from the non-resonant component.
        ChiTwo_yyz_pseudo = ChiTwoeffCaF2Gold/F_ssp_CaF2Gold   # [m/V]
        
        pol = PolarizationType.lower()
        if pol == 'ssp':
            ChiTwoNR = ChiTwoeffCaF2Gold  # [m^2/V]
        elif pol == 'psp':
            # minimal single-parameter estimate: assume only yyz contributes and use your chosen prefactor definition
            # (If you want a stricter C∞v psp formula, we can replace this.)
            F_psp = L_zz_SFG_CaF2Gold * L_yy_vis_CaF2Gold * L_xx_IR_CaF2Gold * np.sin(theta_SFG_CaF2) * np.cos(theta_IR_CaF2)
            ChiTwoNR = F_psp * ChiTwo_yyz_pseudo  # [m^2/V]
        elif pol == 'ppp':
            # minimal single-parameter estimate: keep only xxz (=yyz) term, neglect zxx/xzx NR terms
            # This gives a conservative "one-parameter" NR background estimate.
            F_ppp_xxz_only = L_xx_SFG_CaF2Gold * L_xx_vis_CaF2Gold * L_xx_IR_CaF2Gold * np.cos(theta_SFG_CaF2) * np.cos(theta_vis_CaF2)
            ChiTwoNR = F_ppp_xxz_only * ChiTwo_yyz_pseudo  # [m^2/V]
        else:
            raise ValueError('[Calib ERROR] Polarization type not supported.')
    
    return ChiTwoNR


# ========== signal process ==========

# prepare the data for fft 1: Create the uniformly distributed data
    # this function will interpolate the spectra data, making them uniformly distributed on frequency axis
    # Frequency: the measured SFG frequency data in Hz, created using function CreateFrequencyAxis
    # Intensity: the measured SFG intensity data in counts
    # UniDisFrequency: uniformly distributed frequency array in Hz
    # IntpletedSpec: interpolated spectrum in counts
def UniDisTreat (Frequency, Intensity):
    # sort the raw data into the increasing order for the interpolation
    idx = np.argsort(Frequency)
    Frequency = Frequency[idx]
    Intensity = Intensity[idx]
    
    # uniformly distributed frequency axis
    UniDisFrequency = np.linspace(Frequency.min(), Frequency.max(), len(Frequency))
    
    # create the interpolate function SpecInterFun and interpolate
    SpecInterFun = Pchi(Frequency, Intensity)
    IntpledSpec = SpecInterFun (UniDisFrequency)

    return UniDisFrequency, IntpledSpec

# prepare the data for fft 2: Find out the width of the fringe for pre-fft filter
    # find out the width of the fringe
    # only the Intensity was the input, frequency is not included, i.e. sample interval d = 1.0 
    # Intensity: The interpolated zqz/gold reference spectra
def EstimateFringeWidth (Intensity):
    # do the fft
    N = Intensity.size
    Y = np.fft.fft(Intensity)
    Amp = np.abs(Y)

    # create the frequency axis Freq and mask it for the positive half FreqPos, and find the corresponding amplitude AmpInFreqPos
    Freq = np.fft.fftfreq(N, d=1.0)
    FreqPos = Freq[Freq >= 0]
    AmpInFreqPos = Amp[Freq >= 0]
    
    # find the peak, i.e. the frequency of the fringe in terms of inverted number of data points
    # NoForDC: number of data points for DC signal
    NoForDC = max(3, int(0.01*len(AmpInFreqPos)))
    AmpInFreqPos[:NoForDC] = 0
    # find the peak index (where is the peak)
    PeakFreq = FreqPos[np.argmax(AmpInFreqPos)]
    PeakPeriod = 1.0/PeakFreq

    return PeakPeriod

# prepare the data for fft 3: Planck-taper filter
    # this filter is used to avoid ringing of the fft
    # NoOfPoints: total number of points of the data file
    # PeakPeriod: The period of interference fringes in terms of index
    # NoOfPeaks: how many peak will be treated by the filter
    # LRratio: the width of the filter on the left and right sides
    # MaxEdgeRatio: no. of treated points/total no. of points. If too large, the signal will be distorted
    # EpsPlanck: how steep the transition is
def PlanckTaperFilter (NoOfPoints, PeakPeriod, NoOfPeaks, LRRatio, MaxEdgeRatio, EpsPlanck, PlanckTaperFilterMin):
    # initiate the filter
    PreFFTFilter = np.ones(NoOfPoints)

    # robust check: ensure the minimum period of peak
    if PeakPeriod < 7: PeakPeriod = 7

    # the left and right boundary of the filter in terms of no. of treated points
    LeftBoundary = int(NoOfPeaks*PeakPeriod)+1
    RightBoundary = int(LeftBoundary*LRRatio)

    if (LeftBoundary+RightBoundary)/NoOfPoints > MaxEdgeRatio:
        raise ValueError(f'Frequency domain filter error: Too valent treatment, data distorted!')
    
    # left boundary
    xL = np.linspace(0.0, 1.0, LeftBoundary, endpoint=False)
    xL = np.clip(xL, 1.0e-6, 1.0-(1e-6))
    PowerxL = EpsPlanck/xL + EpsPlanck/(xL-1.0)
    PowerxL = np.clip(PowerxL, -50, 50)
    WeightxL = 0.92/(np.exp(PowerxL)+1.0)+PlanckTaperFilterMin
    PreFFTFilter[:LeftBoundary] = WeightxL

    # right boundary
    xR = np.linspace(0.0, 1.0, RightBoundary, endpoint=False)
    xR = np.clip(xR, 1.0e-6, 1.0-(1e-6))
    PowerxR = EpsPlanck/xR + EpsPlanck/(xR-1.0)
    PowerxR = np.clip(PowerxR, -50, 50)
    WeightxR = 0.92/(np.exp(PowerxR)+1.0)+PlanckTaperFilterMin
    WeightxR = np.flip(WeightxR)
    PreFFTFilter[-RightBoundary:] = WeightxR

    return PreFFTFilter

# FFT of the spectra from frequency (Hz) to time domain (s)
    # Frequency: uniformly distributed frequency axis
    # Signal: Spectra
def FFTFromFreqToTime (Frequency, Signal):
    Frequency = np.asarray(Frequency)
    Signal = np.asarray(Signal)

    # protect from irrational analysis
        # check if the Frequency and Signal are in the same size
    AreSameSize = (Frequency.size==Signal.size)
        # check if Frequency is uniformly distributed
    DiffFreq = np.diff(Frequency)
    dF = DiffFreq.mean()
    IsUniform = np.allclose(DiffFreq, dF, rtol=1e-5, atol=0)
    if AreSameSize and IsUniform:
        Amplitude = np.fft.fft(Signal)
        Time = np.fft.fftfreq(Frequency.size, dF)
        # rearrange the time and amplitude to avoid the jump
        idx = np.argsort(Time)
        tSorted = Time[idx]
        AmpSorted = Amplitude[idx]
        return tSorted, AmpSorted   
    else:
        raise ValueError('Sampling Error: Frequency does not match the Spectra or they do not match in size!')

# pre-ifft 1: find the SFG/LO delay time T0 and reflection peak delay time T1
# caution: This can only applied to the reference material such as zqz and gold
# Time: time axis from fft
# Amplitude: amplitude axis from fft
# n_skip: no. of data points representing optic rectification
# frac: the decay ratio of T0 peak, when [Amplitude of signal]/[Amplitude of T0 peak] < frac, the T0 peak is set to be over
# margin: number of data points to jump for safe search of reflection peak
def FindDelayTime (Time, Amplitude, n_skip, frac, margin):
    # find the SFG peak    
    AbsAmp = np.abs(Amplitude)
    TimeP = Time > 0
    IdxTimeP1 = np.where(TimeP)[0]           # this returns the index of the positive half time axis
    IdxSearch1 = IdxTimeP1[n_skip:]           # skip the optic rectification DC components
    AmpSearch1 = AbsAmp[IdxSearch1]
    IdxPeak1 = np.argmax(AmpSearch1)  # this returns the index of the peak
    Idx0 = IdxSearch1[IdxPeak1]               # this returns the index of peak in the fft-ed tat
    T0 = Time[Idx0]                         # find the period time
    
    # find the secondary reflection peak
        # decide if the T0 peak is over; if not, find the index by increasing the index while check the amplitude
    Peak0 = AbsAmp[Idx0]
    Threshold = Peak0*frac
    N = len(Time)
    IdxEndPeak0 = Idx0
    while IdxEndPeak0 + 1 < N and AbsAmp[IdxEndPeak0 + 1] >= Threshold:
        IdxEndPeak0 = IdxEndPeak0 + 1
    
        # start searching for the reflection peak
    IdxStart = IdxEndPeak0 + margin
    IdxTimeP2 = IdxTimeP1[IdxTimeP1 >= IdxStart]
        # decide if we have enough data points to search
    if IdxTimeP2.size < 5:
        T1=None
    else:
        AbsSearch2 = AbsAmp[IdxTimeP2]
        IdxPeak2 = np.argmax(AbsSearch2)
        Idx1 = IdxTimeP2[IdxPeak2]
        T1 = Time[Idx1]
    return T0, T1

# pre-ifft 2: Build the half Happ-Genzel window
    # L: width of window, in number of points
def HalfHappGenzal (L):
    j = np.arange(L+1, dtype=float)
    w = 0.54 - 0.46*(np.cos(np.pi*j/L))
    return w

# pre-ifft 3: Build the boxcar window
    # Time: time axis from fft, in second
    # tL: [left boundary of time]/[T0]
    # tR: [right boundary of time]/[T0]
def Boxcar (Time, T0, tL, tR):
    Time = np.asarray(Time, dtype=float)
    BoxCarWindow = np.zeros(len(Time), dtype=float)
    # find the index
    IdxtL = int(np.argmin(np.abs(Time-tL*T0)))
    IdxtR = int(np.argmin(np.abs(Time-tR*T0)))
    # safety check
    if IdxtL < IdxtR:
        BoxCarWindow[IdxtL:IdxtR+1] = 1
    else:
        raise ValueError('[BoxCar] Wrong range!')
    return BoxCarWindow 

# pre-ifft 4: Build the time domain filter
    # Time: time axis from fft
    # T0: Delay time of LO/SFG
    # T1: Delay time of reflection peaks
    # T0LBoundary: [SFG part plateau starting time]/T0
    # T0RBoundary: [SFG part plateau ending time]/T0
    # HGWidthT0Ratio: [half Happ-Genzel window width in sec.]/[T0]
    # T1LRel: Reflection peak left boundary is [T1-T1LRel*T0]
    # T1RRe: Reflection peak right boundary is [T1+T1LRel*T0]
    # eps: base of the noise on the right side of time axis, eps=1 means no compression of noise
    # NoiseSamplingStartinPs: starting time of the noise sampling
    # NoiseHGRampinginPs: ramping width of the Happ-Genzel window for noise
    # T1ValleyHalfRatio: width of valley = T1ValleyHalfRatio x T0 plateau width x 2
    # ValleyHGRatio: width of HG ramping / width of T1 valley
def TimeDomainFilter(Time, T0, T1, T0LBoundary, T0RBoundary, HGWidthT0Ratio, T1LRel, T1RRel, eps, NoiseSamplingStartinPs,NoiseHGRampinginPs, T1ValleyHalfRatio, ValleyHGRatio):    
    # initiation
    Time = np.asarray(Time, dtype=float)
    N = len(Time)
    Window = np.zeros(N, dtype=float)

    # T0 window for the SFG peaks
    tL0 = T0LBoundary * T0
    tR0 = T0RBoundary * T0

    idxL0 = int(np.argmin(np.abs(Time - tL0)))
    idxR0 = int(np.argmin(np.abs(Time - tR0)))
    idxL0 = max(0, min(N - 1, idxL0))
    idxR0 = max(0, min(N - 1, idxR0))
    if idxR0 < idxL0:
        idxL0, idxR0 = idxR0, idxL0
    # create plateau = 1
    Window[idxL0:idxR0 + 1] = 1.0
    plateau_width = max(idxR0 - idxL0, 1)
    HGWidthPts = int(HGWidthT0Ratio * plateau_width)
    # create the HG ramping area for SFG peaks
    if HGWidthPts > 0:
        # left HG ramping 0->1 
        startL = max(0, idxL0 - HGWidthPts)
        Leff = idxL0 - startL
        if Leff > 0:
            HGL = HalfHappGenzal(Leff)[:-1]
            Window[startL:idxL0] = np.maximum(Window[startL:idxL0], HGL)
        # right HG ramping 1->0
        endR = min(N - 1, idxR0 + HGWidthPts)
        Reff = endR - idxR0
        if Reff > 0:
            HGR = np.flip(HalfHappGenzal(Reff))[:-1] 
            Window[idxR0 + 1:endR + 1] = np.maximum(Window[idxR0 + 1:endR + 1], HGR)
    else:
        # safety check: if no area for HG, then right boundary = right boundary ot plateau
        endR = idxR0 
    # set negative time axis to 0
    Window[Time < 0] = 0.0
    # Noise sampling on the positive time axis, using HG-ramping transfer
    if eps > 0.0:
        t_eps_start = NoiseSamplingStartinPs * 1e-12   # in ps
        dt = np.mean(np.diff(Time))
        if dt <= 0:
            dt = (Time[-1] - Time[0]) / max(N - 1, 1)
        # calculate the number of data points of HG
        eps_HG_pts = int(abs(NoiseHGRampinginPs * 1e-12 / dt))
        # calculate the index
        if eps_HG_pts > 1 and (t_eps_start < Time[-1]):
            # start index
            idx_eps_start = int(np.argmin(np.abs(Time - t_eps_start)))
            idx_eps_start = max(0, min(N - 1, idx_eps_start))
            # end index
            idx_eps_end = min(N - 1, idx_eps_start + eps_HG_pts)
            ramp_len = max(idx_eps_end - idx_eps_start + 1, 1)
            # ramping area 
            HGeps = HalfHappGenzal(ramp_len)[:-1]
            ramp = eps * HGeps
            # HG ramping area:0->eps, eps=1 means no compression of the noise
            Window[idx_eps_start:idx_eps_start + len(ramp)] = np.maximum(Window[idx_eps_start:idx_eps_start + len(ramp)], ramp)
            # safety check: if filter < eps after ramping, use eps value directly
            if idx_eps_start + len(ramp) < N:
                tail_slice = slice(idx_eps_start + len(ramp), N)
                mask_tail = Window[tail_slice] < eps
                Window[tail_slice][mask_tail] = eps
    # removal of T1 reflection peaks
    if (T1 is not None) and np.isfinite(T1) and (T1 > 0):
        # Define the width using T0 as units
        T0_width_time = (T0RBoundary - T0LBoundary) * T0
        if T0_width_time <= 0:
            T0_width_time = abs(T0) * 0.5
        # half of the valley width in picoseconds
        valley_half = T1ValleyHalfRatio * T0_width_time
        tL1 = T1 - valley_half
        tR1 = T1 + valley_half
        # calculate the indices
        idxL1 = int(np.argmin(np.abs(Time - tL1)))
        idxR1 = int(np.argmin(np.abs(Time - tR1)))
        idxL1 = max(0, min(N - 1, idxL1))
        idxR1 = max(0, min(N - 1, idxR1))
        if idxR1 < idxL1:
            idxL1, idxR1 = idxR1, idxL1
        # calculate the T0 window
        # safety check: make sure the valley is totally on the right side of T0 end point
        if idxL1 > endR:
            # suppressing of multiple reflections
            Window[idxL1:idxR1 + 1] = 0.0
            # calculate the number of data points of valley and its HG
            valley_width_pts = max(idxR1 - idxL1, 1)
            valley_HG_pts = int(ValleyHGRatio * valley_width_pts)
            # safety check: ensure it's needed to create the valley
            if valley_HG_pts > 0 and eps > 0.0:
                # HG-ramping on the left side
                startVL = max(endR + 1, idxL1 - valley_HG_pts)  # make sure ends on the right side of T0
                LeffV = idxL1 - startVL
                if LeffV > 0:
                    HGLV = HalfHappGenzal(LeffV)[:-1]  # 0->1
                    transVL = eps * (1.0 - HGLV)      # eps->0
                    Window[startVL:idxL1] = np.minimum(Window[startVL:idxL1], transVL)
                # HG-ramping on the right side
                endVR = min(N - 1, idxR1 + valley_HG_pts)
                ReffV = endVR - idxR1
                if ReffV > 0:
                    HGRV = np.flip(HalfHappGenzal(ReffV))[:-1]  # 0->1
                    transVR = eps * (1.0 - HGRV)               # 0->eps
                    Window[idxR1 + 1:endVR + 1] = np.minimum(Window[idxR1 + 1:endVR + 1], transVR)

    # ensure the negative time axis is 0
    Window[Time < 0] = 0.0

    Window = np.clip(Window, 0.0, 1.0)
    NumOnes = int(np.sum(np.isclose(Window, 1.0, atol=1e-6, rtol=0.0)))
    return Window, NumOnes

# iFFT function: From time domain to frequency domain
    # TFiltered: time domain filtered data
    # Time: time axis
def iFFTFromTimeToFrequency (TFiltered, PhaseCorr):
    # Unsort the time to ensure the correct order of data
    AmpForiFFT = np.fft.ifftshift(TFiltered)
    # phase correction
    PhaseCorrRad = np.deg2rad(PhaseCorr)
    # ifft
    iFFTAmp = np.fft.ifft(AmpForiFFT)*np.exp(1j * PhaseCorrRad)
    return iFFTAmp

# ========== the main work flow ==========
# ***** save the analysis parameters *****
save_analysis_parameters(FolderPath)

# ***** file manipulation *****
# data file sort
FileSortResult = SortDataFile(FolderPath)

# find out what is the reference material
if RefMaterial == 'gold':
    RefBgList = FileSortResult["GoldBg"]
    RefList = FileSortResult["GoldRef"]
elif RefMaterial == 'zqz':
    RefBgList = FileSortResult["zqzBg"]
    RefList = FileSortResult["zqzRef"]
elif RefMaterial == 'd2o':
    RefBgList = FileSortResult["D2OBg"]
    RefList = FileSortResult["D2ORef"]
elif RefMaterial in ['water','Water','H2O', 'h2o']:
    RefBgList = FileSortResult["WaterBg"]
    RefList = FileSortResult["WaterExp"]
    print(f'[WARNING] Caution: H2O as reference! Check if Media2 == D2O')
    
    # check if the media2 is d2o in this condition
    if Media2.lower()!='d2o':
        print(f"[WARNING] Caution: Using H2O as reference, but Media2 is not D2O\n"
            f"[WARNING] Check if this is intended (e.g. H2O is sample)!")
else:
    raise ValueError(f"Unknown reference material: {RefMaterial}!")

# check if the RefMaterial is correctly chosen
if len(RefBgList)==0 or len(RefList)==0:
    # background does not exist or wrong reference material
    raise ValueError("Check the background file, RefMaterial and reference files!")

else: 
    # start analyzing the reference
    print(f"Start analyzing the reference:")
    #RefBgSubRaw is a array of background-subtracted reference
    RefBgSubRaw = BgSubtract(FolderPath, RefBgList, RefList) 
    print(f"[Exp Config INFO] Reference is {RefMaterial} spectra.")
    print(f"[Exp Config INFO] In total {RefBgSubRaw.shape[0]} {RefMaterial} reference files.")
    
    # remove the spikes in the reference spectra
    # RemoveSpikeOutliers (Spectra, Reference, KernelSize, ThresholdFactor, MinFiles, SpikeSafeEps)
    if SPIKE_REMOVE:
        RefBgSub = RemoveSpikeOutliers(RefBgSubRaw, RefBgSubRaw[0],SPIKE_KERNEL, SPIKE_THRESHOLD_FACTOR, SPIKE_MIN_FILES, SPIKE_SAFE_EPS)
    else:
        BgRefSub = BgRefSub.copy()

    # calculate the mean of the background-subtracted reference
    RefSpectra = RefBgSub.mean(axis=0)
    
    # extract SFG and IR frequency (in cm-1)
    Filepath = os.path.join(FolderPath, RefList[0])
    Omega_SFG, Omega_IR = CreateFrequencyAxis(Filepath, VisWavelength)

    # create the frequency axis in Hz
    TemporalFrequency_SFG, TemporalFrequency_IR = Omega_SFG*c*100, Omega_IR*c*100
   
    # prepare the frequency uniformly distributed data
    UniFreRef, IntpRef = UniDisTreat(TemporalFrequency_SFG, RefSpectra)
        
    # create the IR wavenumber axis for plot
    SFGWavenumber = np.asarray(UniFreRef/(c*100.0), dtype=float)
    VisWavenumber = 1.0e7/VisWavelength
    IRWavenumber = np.asarray(SFGWavenumber - VisWavenumber, dtype=float)

# find out the experiment data
SampleBgList = FileSortResult["WaterBg"]
SampleList = FileSortResult["WaterExp"]

if len(SampleBgList)==0 or len(SampleList)==0:
    # background does not exist or wrong reference material
    raise ValueError("[SAMPLE ERROR] Check the experiment files, RefMaterial and reference files!")

else: 
    # start analyzing the experiment data files
    print(f"Start analyzing the experiment data files:")
    print(f"[Sample INFO] In total {len(SampleList)} spectra.")
    print(f"[Sample INFO] In total {len(SampleBgList)} water background files.")
    
    # RawSampleBgSub is a array of background-subtracted reference
    RawSampleBgSubRaw = BgSubtract(FolderPath, SampleBgList, SampleList) 
    
    # remove the spike for the sample
    if SPIKE_REMOVE:
        RawSampleBgSub = RemoveSpikeOutliers(RawSampleBgSubRaw, RawSampleBgSubRaw[0],SPIKE_KERNEL, SPIKE_THRESHOLD_FACTOR, SPIKE_MIN_FILES, SPIKE_SAFE_EPS)
    else:
        RawSampleBgSub = RawSampleBgSubRaw.copy()

    # calculate the mean of the background-subtracted experiment data files
    SampleBgSub = RawSampleBgSub.mean(axis=0)
    
    # calculate the SFG frequency axix in Hz
    SampleFilePath = os.path.join(FolderPath, SampleList[0])
    Omega_SFG_Samp, Omega_IR_Samp = CreateFrequencyAxis(SampleFilePath, VisWavelength)
    TemporalFrequency_SFG_Samp = Omega_SFG_Samp*c*100
    
    # interpolate of the experiment data files
    UniFreSamp, IntpSamp = UniDisTreat(TemporalFrequency_SFG_Samp, SampleBgSub)
    
    # check if the sample and reference frequency axis close enough
    if not np.allclose(UniFreRef, UniFreSamp, rtol=1.0e-6, atol=0.0):
        print(r'[Sample WARNING] The experiment data is not collected using the same CCD configuration!')
    else: 
        print(r'[Exp Config INFO] The experiment data is collected using the same CCD configuration.')

# ***** signal processing *****
# pre-fft treatment for the reference spectra    
    # find the peak width for the pre-fft filter using the interpolated frequency-uniformly-distributed reference spectra
PeakWidth = EstimateFringeWidth(IntpRef)

    # pre-fft filter preparation
PreFFTFilter = PlanckTaperFilter(len(UniFreRef),7,NoofPeaks,LRRatio,PlanckFilterRatio_Max,PlanckEps, PlanckTaperFilterMin)
    # filtered reference
RefForFFT = IntpRef * PreFFTFilter

# FFT of the treated Reference
DelayTime, CompAmplitude = FFTFromFreqToTime(UniFreRef, RefForFFT)

# Find the location of SFG signal in time domain (i.e. time difference between SFG and LO) based on the reference spectra
T0, T1 = FindDelayTime (DelayTime, CompAmplitude, n_skip, frac, margin)

# time domain filter of the reference signal
TimeWindow, NumOnes = TimeDomainFilter(DelayTime, T0, T1, T0LBoundary, T0RBoundary, HGWidthT0Ratio, T1LRel, T1RRel,eps, NoiseSamplingStartinPs,NoiseHGRampinginPs, T1ValleyHalfRatio, ValleyHGRatio)
print (f'[Time Domain Filter INFO] Current time domain filter will keep {NumOnes} points intact.')

# time domain filtration of the reference signal
TFilteredRef = TimeWindow*CompAmplitude

# iFFT of the time domain filtered signal
    # complex amplitude
iFFTCompAmp = iFFTFromTimeToFrequency(TFilteredRef, 0)

# calculate the phase for the references
    # initiate the data storage
NumOfRef, NumOfDataPtsRef = RefBgSub.shape
RefiFFT_all = np.zeros((NumOfRef, NumOfDataPtsRef), dtype=complex)
PhaseRef = np.zeros((NumOfRef, NumOfDataPtsRef), dtype=float)
    # ifft to calculate the phase for each collection
for i in range(NumOfRef):
    Ref_i = RefBgSub[i, :]
    # interpolate each reference shot onto UniFreRef grid
    UniFreRef_i, IntpRef_i = UniDisTreat(TemporalFrequency_SFG, Ref_i)
    # apply the SAME pre-FFT filter used for the average reference
    FFT_amp_i, _ = FFTFromFreqToTime(UniFreRef_i, IntpRef_i * PreFFTFilter)
    # apply the SAME time-domain window
    TFilteredRef_i = TimeWindow * FFT_amp_i
    # iFFT back to frequency domain (apply same phase correction)
    RefiFFT_all[i, :] = iFFTFromTimeToFrequency(TFilteredRef_i, 0)
    # calculate the phase
    PhaseRef[i,:] = np.angle(RefiFFT_all[i, :])
    # mean phase 
PhaseRefMean = np.angle(np.mean(np.exp(1j*PhaseRef),axis=0))

# analysis for the experiment data
    # most of the conditions are inherited from the refererence
    # pre-fft treatment for experiment data
SampleForFFT = PreFFTFilter*IntpSamp

    # fft of the sample: frequency to delay time domain
DelayTimeSamp, CompAmplitudeSamp = FFTFromFreqToTime(UniFreSamp, SampleForFFT)
    # safety check: if the experiment data Delay time is close enough to the reference
if not np.allclose(DelayTime, DelayTimeSamp, rtol=1.0e-6, atol=0.0):
    print(r'[Sample WARNING] The sample delay time is different from the reference!')
else:
    print(r'[Exp Config INFO] The reference and sample are close enough in delay time.')
    # time domain filtration of the experiment data
    TFilteredSamp = TimeWindow*CompAmplitudeSamp
    # iFFT from time domain to frequency domain
    # follow the common definition of the phase correction
    if RefMaterial == 'zqz':
        PhaseCorr = PhaseCorr - 90 
    iFFTCompAmpSamp = iFFTFromTimeToFrequency(TFilteredSamp, PhaseCorr)

# calculate the average \chi^{(2)}_{eff}
    # calculate exposure time ratio for the correction of exposure time
ExpRatio = RefExposure/SamExposure    
    # analysis for the \chi^{(2)}_{eff} for each sample spectrum
    # initiate the \chi^{(2)}_{eff} data storage
    # initiate the phase storage
NumOfSamp, NumOfDataPts = RawSampleBgSub.shape
ChiTwoMeased_all = np.zeros((NumOfSamp, NumOfDataPts), dtype=complex)
PhaseSamp_all = np.zeros((NumOfSamp, NumOfDataPts), dtype=float)
PhaseChiTwo_all = np.zeros((NumOfSamp, NumOfDataPts), dtype=float)

# analyze all the sample spectra
for i in range(NumOfSamp):
    # read the background substrated raw spectra data
    Samp_i = RawSampleBgSub[i,:]
    # interpolation of the sample raw data; UniDisTreat function will train the interpolater and then use it. 
    UniFreSamp_i, IntpSamp_i = UniDisTreat(TemporalFrequency_SFG_Samp, Samp_i)
    # safety check for the intropolation
    if not np.allclose(UniFreSamp_i, UniFreRef, rtol=1.0e-6, atol=0.0):
        print(f'[Sample WARNING] Frequency axis mismatch in file: {SampleList[i]}')
    # pre-FFT filter in the time domain
    SampleForFFT_i = IntpSamp_i*PreFFTFilter
    # fft for the sample data
    DelayTSamp_i, CompAmpSamp_i = FFTFromFreqToTime(UniFreSamp_i, SampleForFFT_i)
    # safety check for the fft
    if not np.allclose(DelayTSamp_i, DelayTime, rtol=1.0e-6, atol=0.0):
        print(f'[Sample WARNING] Delay time mismatch in file: {SampleList[i]}')
    # time-domain filtration of the data for ifft
    TFilteredSamp_i = CompAmpSamp_i*TimeWindow
    # ifft
    iFFTCompAmp_i = iFFTFromTimeToFrequency(TFilteredSamp_i, PhaseCorr)
    # phase calculation
    PhaseSamp_all[i,:]=np.angle(iFFTCompAmp_i)
    # calculate \chi^{(2)}_{eff}
    ChiTwoMeased_i = (iFFTCompAmp_i/iFFTCompAmp)*ExpRatio
    ChiTwoMeased_i = np.conj(ChiTwoMeased_i)
    PhaseChiTwo_all[i,:] = np.angle(ChiTwoMeased_i)
    # data storage
    ChiTwoMeased_all[i,:] = ChiTwoMeased_i
    # Savitzky–Golay smoothing
    if ChiTwoMeased_i.size >= SG_WINDOW and SG_WINDOW % 2 == 1:
        # smoothing for real/imag parts
        ChiTwoMeased_i_real_sm = savgol_filter(ChiTwoMeased_i.real, SG_WINDOW, SG_POLY, mode='interp')
        ChiTwoMeased_i_imag_sm = savgol_filter(ChiTwoMeased_i.imag, SG_WINDOW, SG_POLY, mode='interp')
        ChiTwoMeased_i_sm = ChiTwoMeased_i_real_sm + 1j * ChiTwoMeased_i_imag_sm
    else:
        # not enough points, use original data
        ChiTwoMeased_i_sm = ChiTwoMeased_i

    # data storage: smoothed \chi^{(2)}_{eff}
    ChiTwoMeased_all[i, :] = ChiTwoMeased_i_sm
# phase
PhaseSampMean = np.angle(np.mean(np.exp(1j * PhaseSamp_all), axis=0)) # wrapped mean, rad
PhaseChiTwoMean = np.angle(np.mean(np.exp(1j * PhaseChiTwo_all), axis=0))  # wrapped rad

# ***** refractive indices *****
# get the media1 complex refractive in the experiment regime
# Media1 is either solid or air
# Media2 is either H2O or D2O
fun_real1, fun_imag1 = CreateRefractionIndexFun(ParaPath, Media1)
fun_real2, fun_imag2 = CreateRefractionIndexFun(ParaPath, Media2)
# calculate the complex refractive index
CompNMedia1 = fun_real1(SFGWavenumber)+1j*fun_imag1(SFGWavenumber)
CompNMedia2 = fun_real2(SFGWavenumber)+1j*fun_imag2(SFGWavenumber)
# take only the real part
nMedia1 = np.real(CompNMedia1)
nMedia2 = np.real(CompNMedia2)
# print out for safety check
print(f"[Exp Config INFO] n({Media1}) @ SF: {nMedia1.min():.4f} - {nMedia1.max():.4f}")
print(f"[Exp Config INFO] n({Media2}) @ SF: {nMedia2.min():.4f} - {nMedia2.max():.4f}")

# ***** calculate the incident angles to the interface *****
# refractive index of the air
n_air_vis = np.ones_like(VisWavenumber, dtype=float)
n_air_IR  = np.ones_like(IRWavenumber, dtype=float)
n_air_SFG = np.ones_like(SFGWavenumber, dtype=float)

# change the incident angle from degree to rad
VisIncidentAngle_deg = VisIncidentAngle
IRIncidentAngle_deg  = IRIncidentAngle

VisIncidentAngle = np.deg2rad(VisIncidentAngle_deg)
IRIncidentAngle = np.deg2rad(IRIncidentAngle_deg)
# using Snell's Law, and the adsorbance is neglected (i.e. the imag part = 0) to calculate the air/Media1 interface
# for vis incident light
    # calculate its refractive index
CompNMedia1Vis = fun_real1(VisWavenumber)+1j*fun_imag1(VisWavenumber)
nMedia1Vis = np.real(CompNMedia1Vis)
# the incident light is from air to solid, then from solid to solid/water interface
arg_vis = n_air_vis/nMedia1Vis*np.sin(VisIncidentAngle)
arg_vis = np.clip(arg_vis, -1.0, 1.0)
IncVisMedia1To2 = np.arcsin(arg_vis)
print(f'[Exp Config INFO] Incident angle of vis to interface is {np.rad2deg(IncVisMedia1To2):.2f} degree.')
# for IR beams
CompNMedia1IR = fun_real1(IRWavenumber)+1j*fun_imag1(IRWavenumber)
nMedia1IR = np.real(CompNMedia1IR)
# the incident light is from air Media1
arg_IR = n_air_IR/nMedia1IR*np.sin(np.asarray(IRIncidentAngle))
arg_IR = np.clip(arg_IR, -1.0, 1.0)
IncIRMedia1To2 = np.arcsin(arg_IR)
print(f'[Exp Config INFO] Incident angle of IR to interface is {np.rad2deg(IncIRMedia1To2).min():.2f} - {np.rad2deg(IncIRMedia1To2).max():.2f} degree.')

# calculate the incident angle of SFG at Media1/Media2 interface
arg_IncSFGMedia1To2 = (IRWavenumber*np.sin(IncIRMedia1To2)*nMedia1IR+VisWavenumber*np.sin(IncVisMedia1To2)*nMedia1Vis)/(nMedia1*SFGWavenumber)
arg_IncSFGMedia1To2 = np.clip(arg_IncSFGMedia1To2, -1.0, 1.0)
IncSFGMedia1To2 = np.arcsin(arg_IncSFGMedia1To2)
print(f'[Exp Config INFO] Incident angle of SF light to interface is {np.rad2deg(IncSFGMedia1To2).min():.2f} - {np.rad2deg(IncSFGMedia1To2).max():.2f} degree.')

# SFG angle in Media2, using Snell's Law and real refractive index
arg_AngleSFGMedia2 = (nMedia1/nMedia2)*np.sin(IncSFGMedia1To2)
arg_AngleSFGMedia2 = np.clip(arg_AngleSFGMedia2, -1.0, 1.0)
AngleSFGMedia2 = np.arcsin(arg_AngleSFGMedia2)

# calculate the reflectivity, depending on the polarization type
PolarizationType = PolarizationType.lower()

if PolarizationType == 'ssp':    
    rStarSamp, _ = FresnelR_sT_s(CompNMedia1, CompNMedia2, IncSFGMedia1To2)
else: # polarization type is psp or ppp
    rStarSamp, _ = FresnelR_pT_p(CompNMedia1, CompNMedia2, IncSFGMedia1To2) 

# ***** chitwo amplitude absolute value *****
# calculate the nonresonant signel
ChiTwoNR = ComputeChiTwoNR(ParaPath, Media1, VisIncidentAngle, IRIncidentAngle, VisWavenumber, IRWavenumber, SFGWavenumber, ModelType, PolarizationType)

# calculate the \chi^{(2)}_{effective}
ChiTwoEff_all = ChiTwoMeased_all*ChiTwoNR/rStarSamp
# calculate the mean and standard derivation of \chi^{(2)}_{effective}
ChiTwoEff_mean = np.mean(ChiTwoEff_all, axis=0)
ChiTwoEff_realSTD = np.std(ChiTwoEff_all.real, axis=0)
ChiTwoEff_imagSTD = np.std(ChiTwoEff_all.imag, axis=0)

# data preparation for ploting
    # calculate the mean and std of \chi^{(2)}_{measured}
ChiTwoMeasedMean = np.mean(ChiTwoMeased_all, axis=0)
ChiTwoMeasedRealSTD = np.std(ChiTwoMeased_all.real, axis=0)
ChiTwoMeasedImagSTD = np.std(-ChiTwoMeased_all.imag, axis=0)
    # simulation for homodyne \chi^{(2)}_{measured}
ChiTwoMeased2 = np.abs(ChiTwoMeased_all)**2
ChiTwoMeased2Mean = np.mean(ChiTwoMeased2, axis=0)
ChiTwoMeased2STD = np.std(ChiTwoMeased2, axis=0)

# calculate the Fresnel factors to translate effective \chi^{(2)} to tensor components
# refractive indices in Media2 for vis/IR (complex)
CompNMedia2Vis = fun_real2(VisWavenumber)+1j*fun_imag2(VisWavenumber)
CompNMedia2IR = fun_real2(IRWavenumber)+1j*fun_imag2(IRWavenumber)

# real parts for geometry (consistent with your "real-n angles" strategy)
nMedia2Vis = np.real(CompNMedia2Vis)
nMedia2IR = np.real(CompNMedia2IR)

# angles in Media2 for vis/IR using Snell's law, only real part of refractive index is considered
arg_VisMedia2 = (nMedia1Vis/nMedia2Vis)*np.sin(IncVisMedia1To2)
arg_VisMedia2 = np.clip(arg_VisMedia2, -1.0, 1.0)
AngleVisMedia2 = np.arcsin(arg_VisMedia2)

arg_IRMedia2 = (nMedia1IR/nMedia2IR)*np.sin(IncIRMedia1To2)
arg_IRMedia2 = np.clip(arg_IRMedia2, -1.0, 1.0)
AngleIRMedia2 = np.arcsin(arg_IRMedia2)

# Fresnel L-factors for each frequency at Media1/Media2 (air/water or solid/water interface)
L_xx_SFG_Samp, L_yy_SFG_Samp, L_zz_SFG_Samp = FresnelFactors(CompNMedia1, CompNMedia2, IncSFGMedia1To2, AngleSFGMedia2, ModelType)
L_xx_vis_Samp, L_yy_vis_Samp, L_zz_vis_Samp = FresnelFactors(CompNMedia1Vis, CompNMedia2Vis, IncVisMedia1To2, AngleVisMedia2, ModelType)
L_xx_IR_Samp, L_yy_IR_Samp, L_zz_IR_Samp = FresnelFactors(CompNMedia1IR, CompNMedia2IR, IncIRMedia1To2, AngleIRMedia2, ModelType)

if PolarizationType == 'ssp':
    # same as ComputeChiTwoNR
    F_Samp = L_yy_SFG_Samp*L_yy_vis_Samp*L_xx_IR_Samp*np.cos(IncIRMedia1To2)
elif PolarizationType == 'psp':
    # same structure as ComputeChiTwoNR
    F_Samp = L_zz_SFG_Samp*L_yy_vis_Samp*L_xx_IR_Samp*np.sin(IncSFGMedia1To2)*np.cos(IncIRMedia1To2)
elif PolarizationType == 'ppp':
    # same simplified single-term as ComputeChiTwoNR
    F_Samp = L_xx_SFG_Samp*L_xx_vis_Samp*L_xx_IR_Samp*np.cos(IncSFGMedia1To2)*np.cos(IncVisMedia1To2)
else:
    raise ValueError('[Fresnel ERROR] Polarization type not supported.')
print(f"[Exp Config INFO] Fresnel factor F$_{PolarizationType}$ (sample) computed.")

# calculate the tensor component
ChiTwoComp_all = ChiTwoEff_all/F_Samp
ChiTwoCompMean = np.mean(ChiTwoComp_all, axis=0)
ChiTwoCompRealSTD = np.std(ChiTwoComp_all.real, axis=0)
ChiTwoCompImagSTD = np.std(ChiTwoComp_all.imag, axis=0)

# ***** output figures/messages *****
# print the SFG/LO delay time
print (f'[Spectra INFO] SFG/LO delay time is {T0*1e12:.3f} ps.')
print (f'[Spectra INFO] The secondary reflection peak is at {T1*1e12:.3f} ps')
# print the range of results
print(f"[Calib INFO] |ChiTwoNR|: {np.abs(ChiTwoNR).min():.3e} - {np.abs(ChiTwoNR).max():.3e} (m^2/V)")
print(f"[Calib INFO] |r*_sample|: {np.abs(rStarSamp).min():.3f} - {np.abs(rStarSamp).max():.3f}")
print(f"[Calib INFO] |F_{PolarizationType.lower()}|: {np.abs(F_Samp).min():.3e} - {np.abs(F_Samp).max():.3e}")
print(f"[Calib INFO] |chi_component_mean|: {np.abs(ChiTwoCompMean).min():.3e} - {np.abs(ChiTwoCompMean).max():.3e} (m/V)")

# fig1: plot the background subtracted references (raw, average, pre-fft filter-treated and filter) and the fft results  
    # initiate the figure
fig1, axs1 = plt.subplots(2, 2, figsize=(figwidth, figheight), constrained_layout=True)
RefFig = axs1[0, 0]
RefFFT = axs1[0, 1]
RefPhase = axs1[1, 0]
RefiFFT = axs1[1,1]
    # raw references
    # RefBgSub.shape = (No of the measurements, No of the data points per measurement)

# top left figure: plot the background subtracted reference, mean reference (background subtracted), interpolated reference, filter and filtered reference.
    # all raw data
for i in range(RefBgSub.shape[0]):
    RefFig.plot(TemporalFrequency_SFG*1e-14, RefBgSub[i], alpha=0.5, linewidth=1)
    # average reference
RefFig.plot(TemporalFrequency_SFG*1e-14, RefSpectra, linewidth=1, color='black', label='Average reference')
    # interpolated reference
RefFig.plot(UniFreRef*1e-14, IntpRef, linewidth=1, color='red', label='Interpolated reference')
RefFig.plot(UniFreRef*1e-14, RefForFFT, linewidth=1, color='blue', label='Filtered reference')
    # figure configuration
RefFig.set_xlabel(r"$f_{\mathrm{vis}}+f_{\mathrm{IR}}\,\mathrm{[10^{14} Hz]}$")
RefFig.set_ylabel("Counts")
RefFig.grid(True)
    # plot the pre-filter
ax2 = RefFig.twinx()
ax2.plot(UniFreRef*1e-14, PreFFTFilter, linewidth=1, color='green', label='Planck-taper filter')
ax2.set_ylabel("Filter weight")
ax2.set_ylim(-0.1, 1.6)
    # combine the legends
lines1, labels1 = RefFig.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
RefFig.legend(lines1 + lines2, labels1 + labels2, loc='best')
RefFig.set_title(f"Reference ({RefMaterial}) data and Pre-FFT filter")

# top right figure: plot the time domain FFt results and filter
    # Amplitude vs. Delay time
RefFFT.plot(DelayTime*1e12, np.real(CompAmplitude)*1e-6, color='blue', label='Real part')
RefFFT.plot(DelayTime*1e12, np.imag(CompAmplitude)*1e-6, color='red', label='Imag part')
RefFFT.plot(DelayTime*1e12, np.abs(CompAmplitude)*1e-6, color='green', label='Abs')
RefFFT.set_xlabel('Delay time [ps]')
RefFFT.set_ylabel('Amplitude [$10^{6}$ a.u.]')
RefFFT.grid(True)
RefFFT.set_xlim(-5, 5)
RefFFT.set_ylim(-0.8, 1.2)
RefFFT.legend()
RefFFT.set_title(f"FFT of the {RefMaterial} spectrum and time domain filter")
axw = RefFFT.twinx()
    # plot the time filter
axw.plot(DelayTime*1e12, TimeWindow, 'k--', linewidth=1, label='Time-domain filter')
axw.set_ylabel('Filter weight')
axw.set_ylim(-0.05, 1.55)
    # combine the legends
lines1, labels1 = RefFFT.get_legend_handles_labels()
lines2, labels2 = axw.get_legend_handles_labels()
RefFFT.legend(lines1 + lines2, labels1 + labels2, loc='best')

# bottom left figure: plot the phase data
for i in range(NumOfRef):
    PhaseRef_i = np.unwrap(PhaseRef[i,:])    
    RefPhase.plot(IRWavenumber, np.rad2deg(PhaseRef_i), alpha=0.5, linewidth=0.5)
PhaseRefMean_unwrap = np.unwrap(PhaseRefMean)
RefPhase.plot(IRWavenumber, np.rad2deg(PhaseRefMean_unwrap), color='black', linewidth=1, label='Mean phase')
RefPhase.set_xlabel(r'$\nu_{\mathrm{IR}} \mathrm{[cm^{-1}]}$')
RefPhase.set_ylabel('Phase [degree]')
RefPhase.set_xlim(Frequency_min, Frequency_max)
RefPhase.grid(True)
RefPhase.legend()
RefPhase.set_title(f"Phases of {RefMaterial}")

# bottom right figure: iFFT result of the time domain filtered data
RefiFFT.plot(IRWavenumber, np.real(iFFTCompAmp), color='blue', label='Real')
RefiFFT.plot(IRWavenumber, np.imag(iFFTCompAmp), color='red', label='Imag')
RefiFFT.plot(IRWavenumber, np.abs(iFFTCompAmp), color='green', label='Abs')
RefiFFT.set_xlim(np.min(IRWavenumber), np.max(IRWavenumber))
RefiFFT.set_xlabel(r'$\nu_{\mathrm{IR}}\ \mathrm{[cm^{-1}]}$')
RefiFFT.set_ylabel('Intensity [a.u.]')
RefiFFT.set_title(f'Filtered SFG signal of {RefMaterial} reference')
RefiFFT.legend()
RefiFFT.grid(True)

# fig2: plot the sample data
    # initialize the plot
fig2, axs2 = plt.subplots(2, 2, figsize=(figwidth, figheight), constrained_layout=True)
SampFig = axs2[0, 0]
SampleFFT = axs2[0, 1]
SamplePhase = axs2[1, 0]
SampleiFFT = axs2[1,1]

# top left figure: plot the background subtracted Sample, mean Sample (background subtracted), interpolated Sample, filter and filtered Sample.
    # all raw data
for i in range(RawSampleBgSub.shape[0]):
    SampFig.plot(TemporalFrequency_SFG*1e-14, RawSampleBgSub[i], alpha=0.5, linewidth=1)
    # average reference
SampFig.plot(TemporalFrequency_SFG*1e-14, SampleBgSub, linewidth=1, color='black', label='Average sample')
    # interpolated reference
SampFig.plot(UniFreRef*1e-14, IntpSamp, linewidth=1, color='red', label='Interpolated sample')
SampFig.plot(UniFreRef*1e-14, SampleForFFT, linewidth=1, color='blue', label='Filtered reference')
    # figure configuration
SampFig.set_xlabel(r"$f_{\mathrm{vis}}+f_{\mathrm{IR}}\,\mathrm{[10^{14} Hz]}$")
SampFig.set_ylabel("Counts")
SampFig.grid(True)
    # plot the pre-filter
ax2 = SampFig.twinx()
ax2.plot(UniFreRef*1e-14, PreFFTFilter, linewidth=1, color='green', label='Planck-taper filter')
ax2.set_ylabel("Filter weight")
ax2.set_ylim(-0.1, 1.6)
    # combine the legends
lines1, labels1 = SampFig.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
SampFig.legend(lines1 + lines2, labels1 + labels2, loc='best')
SampFig.set_title(r"Sample data and Pre-FFT filter")
# top right figure: plot the time domain FFt results and filter
    # Amplitude vs. Delay time
SampleFFT.plot(DelayTime*1e12, np.real(CompAmplitudeSamp)*1e-6, color='blue', label='Real part')
SampleFFT.plot(DelayTime*1e12, np.imag(CompAmplitudeSamp)*1e-6, color='red', label='Imag part')
SampleFFT.plot(DelayTime*1e12, np.abs(CompAmplitudeSamp)*1e-6, color='green', label='Abs')
SampleFFT.set_xlabel('Delay time [ps]')
SampleFFT.set_ylabel('Amplitude [$10^{6}$ a.u.]')
SampleFFT.grid(True)
SampleFFT.set_xlim(-5, 5)
SampleFFT.legend()
SampleFFT.set_title(f"FFT of the sample spectrum and time domain filter")
axw = SampleFFT.twinx()
    # plot the time filter
axw.plot(DelayTime*1e12, TimeWindow, 'k--', linewidth=1, label='Time-domain filter')
axw.set_ylabel('Filter weight')
axw.set_ylim(-0.05, 1.55)
    # combine the legends
lines1, labels1 = SampFig.get_legend_handles_labels()
lines2, labels2 = axw.get_legend_handles_labels()
SampleFFT.legend(lines1 + lines2, labels1 + labels2, loc='best')

# bottom left figure: plot the phase data
for i in range(NumOfSamp):
    PhaseUnwrap_i = np.unwrap(PhaseSamp_all[i,:])
    SamplePhase.plot(IRWavenumber, np.rad2deg(PhaseUnwrap_i), alpha=0.5, linewidth=1)
PhaseSampMean_unwrap = np.unwrap(PhaseSampMean)
SamplePhase.plot(IRWavenumber, np.rad2deg(PhaseSampMean_unwrap), color='black', linewidth=1, label='Mean phase')
SamplePhase.set_xlabel(r'$\nu_{\mathrm{IR}}$ [cm$^{-1}$]')
SamplePhase.set_ylabel(r'Phase [degree]')
SamplePhase.set_xlim(Frequency_min, Frequency_max)
SamplePhase.grid(True)
SamplePhase.legend()
SamplePhase.set_title('Phase of the sample')

# bottom right figure: iFFT result of the time domain filtered data
SampleiFFT.plot(IRWavenumber, np.real(iFFTCompAmpSamp), color='blue', label='Real')
SampleiFFT.plot(IRWavenumber, np.imag(iFFTCompAmpSamp), color='red', label='Imag')
SampleiFFT.plot(IRWavenumber, np.abs(iFFTCompAmpSamp), color='green', label='Abs')
SampleiFFT.set_xlim(np.min(IRWavenumber), np.max(IRWavenumber))
SampleiFFT.set_xlabel(r'$\nu_{\mathrm{IR}}\ \mathrm{[cm^{-1}]}$')
SampleiFFT.set_ylabel('Intensity [a.u.]')
SampleiFFT.set_title(r'Filtered SFG signal of samples')
SampleiFFT.legend()
SampleiFFT.grid(True)
    
# fig 3: \chi^{(2)}, phase and the related r*
    # initialize the figure
fig3, axs3 = plt.subplots(2, 2, figsize=(figwidth, figheight), constrained_layout=True)
ChiTwoMeased = axs3[0, 0]
Fresnel = axs3[0, 1]
ChiTwoMeasedPhase = axs3[1, 0]
ChiTwo = axs3[1,1]
    # top left panel: measured \chi^{(2)}
ChiTwoMeased.plot(IRWavenumber, np.real(ChiTwoMeasedMean), linewidth=1.5, color='blue', label=r'$\mathrm{Re}\left[\langle\chi^{(2)}_{\mathrm{measured}}\rangle\right]$')
ChiTwoMeased.fill_between(IRWavenumber, (ChiTwoMeasedMean.real-ChiTwoMeasedRealSTD), (ChiTwoMeasedMean.real+ChiTwoMeasedRealSTD), color='blue', alpha=0.25)
ChiTwoMeased.plot(IRWavenumber, np.imag(ChiTwoMeasedMean),linewidth=1.5, color='red', label=r'$\mathrm{Im}\left[\langle\chi^{(2)}_{\mathrm{measured}}\rangle\right]$')
ChiTwoMeased.fill_between(IRWavenumber, (ChiTwoMeasedMean.imag-ChiTwoMeasedImagSTD), (ChiTwoMeasedMean.imag+ChiTwoMeasedImagSTD), color='red', alpha=0.25)
# for i in range(NumOfSamp):
    # ChiTwoMeased.plot(IRWavenumber, np.real(ChiTwoMeased_all[i, :]), linewidth=1.0, alpha=0.3)
    # ChiTwoMeased.plot(IRWavenumber, np.imag(ChiTwoMeased_all[i, :]), linewidth=1.0, alpha=0.3)
ChiTwoMeased.set_xlabel(r'$\nu_{\mathrm{IR}} \mathrm{[cm^{-1}]}$')
ChiTwoMeased.set_ylabel(r'$\chi^{(2)}_{\mathrm{measured}} \mathrm{[a.u.]}$')
ChiTwoMeased.set_xlim(Frequency_min, Frequency_max)
ChiTwoMeased.set_ylim(Amplitude_min, Amplitude_max)
ChiTwoMeased.set_title(r'Normalized $\chi^{(2)}_{\mathrm{measured}}$')
ChiTwoMeased.grid(True)
ChiTwoMeased.legend()

    #  top right panel: plot ALL individual \chi^(2) curves (for quick sanity check)
FresnelR = Fresnel.twinx()
Fresnel.plot(IRWavenumber, np.real(F_Samp), linewidth=1.0, color='blue', label=r'$\mathrm{Re}\left[F_{sample}\right]$')
Fresnel.plot(IRWavenumber, np.imag(F_Samp), linewidth=1.0, color='red', label=r'$\mathrm{Im}\left[F_{sample}\right]$')
Fresnel.set_xlabel(r'$\nu_{\mathrm{IR}} \mathrm{[cm^{-1}]}$')
Fresnel.set_ylabel(r'Fresnel Factor [a.u.]')
FresnelR.plot(IRWavenumber, np.real(rStarSamp), linewidth=1.0, color='green', label=r'$\mathrm{Re}\left[r^{*}_{sample}\right]$')
FresnelR.plot(IRWavenumber, np.imag(rStarSamp), linewidth=1.0, color='orange', label=r'$\mathrm{Im}\left[r^{*}_{sample}\right]$')
FresnelR.set_ylabel(r'$r^*_{sample}$')
Fresnel.set_xlim(Frequency_min, Frequency_max)
Fresnel.set_title(r'Fresnel factors and $r^*_{sample}$')
Fresnel.grid(True)
hL, lL = Fresnel.get_legend_handles_labels()
hR, lR = FresnelR.get_legend_handles_labels()
Fresnel.legend(hL+hR, lL+lR, frameon=True)

    # bottom left panel: phase of \chi^(2)
for i in range(NumOfSamp):
    ChiTwoMeasedPhase.plot(IRWavenumber, np.degrees(PhaseChiTwo_all[i,:]), linewidth=1, alpha=0.5)
ChiTwoMeasedPhase.plot(IRWavenumber, np.degrees(PhaseChiTwoMean), linewidth=1, color='black', label='Mean phase')
ChiTwoMeasedPhase.set_xlabel(r'$\nu_{\mathrm{IR}} \mathrm{[cm^{-1}]}$')
ChiTwoMeasedPhase.set_ylabel('Phase [degree]')
ChiTwoMeasedPhase.set_xlim(Frequency_min, Frequency_max)
ChiTwoMeasedPhase.grid(True)
ChiTwoMeasedPhase.legend(loc='best')
ChiTwoMeasedPhase.set_title(r'$\chi^{(2)}_{\mathrm{measured}}$ phase')

    # bottom right panel: the tensor component \chi^{(2)}
ChiTwo.plot(IRWavenumber, -np.real(ChiTwoCompMean)*1.0e20, linewidth=1.5, color='blue', label=r'$\mathrm{Re}\left[\langle\chi^{(2)}\rangle\right]$')
ChiTwo.fill_between(IRWavenumber, -(ChiTwoCompMean.real-ChiTwoCompRealSTD)*1.0e20, -(ChiTwoCompMean.real+ChiTwoCompRealSTD)*1.0e20, color='blue', alpha=0.25)
ChiTwo.plot(IRWavenumber, -np.imag(ChiTwoCompMean)*1.0e20, linewidth=1.5, color='red', label=r'$\mathrm{Im}\left[\langle\chi^{(2)}\rangle\right]$')
ChiTwo.fill_between(IRWavenumber, -(ChiTwoCompMean.imag-ChiTwoCompImagSTD)*1.0e20, -(ChiTwoCompMean.imag+ChiTwoCompImagSTD)*1.0e20, color='red', alpha=0.25)
ChiTwo.set_xlabel(r'$\nu_{\mathrm{IR}} \mathrm{[cm^{-1}]}$')
ChiTwo.set_ylabel(r'$\chi^{(2)}$ component [$10^{-20}$ m/V]')
ChiTwo.set_title(r'Tensor component $\chi^{(2)}$')
ChiTwo.set_xlim(Frequency_min, Frequency_max)
ChiTwo.set_ylim(ChiTwoFig_min, ChiTwoFig_max)
ChiTwo.grid(True)
ChiTwo.legend()

# fig4: raw vs despiked spectra of reference and sample
fig4, axs4 = plt.subplots(2, 2, figsize = (figwidth, figheight), constrained_layout=True)
Fig4SampleRaw = axs4[0, 0]
Fig4SampleClean = axs4[1, 0]
Fig4RefRaw = axs4[0, 1]
Fig4RefClean = axs4[1, 1]

for i in range(RawSampleBgSubRaw.shape[0]):
    Fig4SampleRaw.plot(TemporalFrequency_SFG_Samp*1e-14, RawSampleBgSubRaw[i], alpha=0.5, linewidth=1)
Fig4SampleRaw.plot(TemporalFrequency_SFG_Samp*1e-14, RawSampleBgSubRaw.mean(axis=0), color='black', linewidth=1.2, label='Mean raw sample')
Fig4SampleRaw.set_xlabel(r"$f_{\mathrm{vis}}+f_{\mathrm{IR}}\,\mathrm{[10^{14} Hz]}$")
Fig4SampleRaw.set_ylabel('Counts')
Fig4SampleRaw.set_title('All raw sample spectra')
Fig4SampleRaw.grid(True)
Fig4SampleRaw.legend(loc='best')

for i in range(RawSampleBgSub.shape[0]):
    Fig4SampleClean.plot(TemporalFrequency_SFG_Samp*1e-14, RawSampleBgSub[i], alpha=0.5, linewidth=1)
Fig4SampleClean.plot(TemporalFrequency_SFG_Samp*1e-14, SampleBgSub, color='black', linewidth=1.2, label='Mean despiked sample')
Fig4SampleClean.set_xlabel(r"$f_{\mathrm{vis}}+f_{\mathrm{IR}}\,\mathrm{[10^{14} Hz]}$")
Fig4SampleClean.set_ylabel('Counts')
Fig4SampleClean.set_title('Sample spectra after spike removal')
Fig4SampleClean.grid(True)
Fig4SampleClean.legend(loc='best')

for i in range(RefBgSubRaw.shape[0]):
    Fig4RefRaw.plot(TemporalFrequency_SFG*1e-14, RefBgSubRaw[i], alpha=0.5, linewidth=1)
Fig4RefRaw.plot(TemporalFrequency_SFG*1e-14, RefBgSubRaw.mean(axis=0), color='black', linewidth=1.2, label='Mean raw reference')
Fig4RefRaw.set_xlabel(r"$f_{\mathrm{vis}}+f_{\mathrm{IR}}\,\mathrm{[10^{14} Hz]}$")
Fig4RefRaw.set_ylabel('Counts')
Fig4RefRaw.set_title(f'All raw {RefMaterial} reference spectra')
Fig4RefRaw.grid(True)
Fig4RefRaw.legend(loc='best')

for i in range(RefBgSub.shape[0]):
    Fig4RefClean.plot(TemporalFrequency_SFG*1e-14, RefBgSub[i], alpha=0.5, linewidth=1)
Fig4RefClean.plot(TemporalFrequency_SFG*1e-14, RefSpectra, color='black', linewidth=1.2, label='Mean despiked reference')
Fig4RefClean.set_xlabel(r"$f_{\mathrm{vis}}+f_{\mathrm{IR}}\,\mathrm{[10^{14} Hz]}$")
Fig4RefClean.set_ylabel('Counts')
Fig4RefClean.set_title(f'{RefMaterial} reference spectra after spike removal')
Fig4RefClean.grid(True)
Fig4RefClean.legend(loc='best') 

# analysis result saving
    # result of the normalized \chi^{(2)}
out_lefttop = np.column_stack([IRWavenumber, ChiTwoMeasedMean.real, ChiTwoMeasedRealSTD, ChiTwoMeasedMean.imag, ChiTwoMeasedImagSTD])
lefttop_path = os.path.join(FolderPath, "ChiTwoMeasured.txt")
np.savetxt(lefttop_path, out_lefttop, delimiter=',', header="IRWavenumber(cm^-1), Re_mean, Re_std, Im_mean, Im_std", comments='', fmt="%.6f, %.10e, %.10e, %.10e, %.10e")
print(f"[Output INFO] Fig3 left-top saved to {lefttop_path}")

    # result of the tensor component
out_rightbot = np.column_stack([IRWavenumber, -ChiTwoCompMean.real, ChiTwoCompRealSTD, -ChiTwoCompMean.imag, ChiTwoCompImagSTD])
rightbot_path = os.path.join(FolderPath, "ChiTwoTensorComponent.txt")
np.savetxt(rightbot_path, out_rightbot, delimiter=',', header="IRWavenumber(cm^-1),Re_mean,Re_std,Im_mean,Im_std", comments='', fmt="%.6f,%.10e,%.10e,%.10e,%.10e")
print(f"[Output INFO] Fig3 bottom-right saved to {lefttop_path}")

    # save the figure
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

fig1.savefig(os.path.join(FolderPath, "Reference.pdf"), format="pdf", bbox_inches="tight")
fig2.savefig(os.path.join(FolderPath, "Sample.pdf"), format="pdf", bbox_inches="tight")
fig3.savefig(os.path.join(FolderPath, "ChiTwo.pdf"), format="pdf", bbox_inches="tight")

plt.show()