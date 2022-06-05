# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:50:28 2021

@author: mozhenling
"""
import copy
import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis
from scipy.stats.mstats import gmean
from scipy.linalg import norm
import scipy.io as scio
import matplotlib.pyplot as plt

#-- import different filters
from dbtpy.filters.gvmdpy import gvmd
from dbtpy.filters.svmdpy import svmd
from dbtpy.filters.vmdpy  import vmd
from dbtpy.filters.meyerpy  import meyer

#-- import tools
from dbtpy.tools.file_tools import get_time, file_type

"""
#-- References
[1] Wang, Dong. "Some further thoughts about spectral kurtosis, spectral L2/L1 norm, 
    spectral smoothness index and spectral Gini index for characterizing repetitive 
    transients." Mechanical Systems and Signal Processing 108 (2018): 360-368.   
[2] Wang, Dong. "Spectral L2/L1 norm: A new perspective for spectral kurtosis 
    for characterizing non-stationary signals." Mechanical Systems and Signal 
    Processing 104 (2018): 290-293.
[3] Borghesani, Pietro, Paolo Pennacchi, and Steven Chatterton. "The relationship 
    between kurtosis-and envelope-based indexes for the diagnostic of rolling
    element bearings." Mechanical Systems and Signal Processing 43.1-2 (2014): 25-43.
[4] Miao, Yonghao, Ming Zhao, and Jiadong Hua. "Research on sparsity indexes 
    for fault diagnosis of rotating machinery." Measurement 158 (2020): 107733.
[5] Hoyer, Patrik O. "Non-negative matrix factorization with sparseness constraints.
    " Journal of machine learning research 5.9 (2004).
[6] Li, Yongbo, et al. "The entropy algorithm and its variants in the fault 
    diagnosis of rotating machinery: A review." Ieee Access 6 (2018): 66723-66741.
[7] Antoni, Jerome. "The infogram: Entropic evidence of the signature of
    repetitive transients." Mechanical Systems and Signal Processing 74 
    (2016): 73-94.
[8] Mo, Zhenling, et al. "Weighted cyclic harmonic-to-noise ratio for rolling
            element bearing fault diagnosis." IEEE Transactions on Instrumentation and
            Measurement 69.2 (2019): 432-442.
---    

"""
###############################################################################

#-- store the filters in the dicttionary
#-- 
# obj_diag_dict={'filters':{}, 'indexes':{'seq':{},'env':{}, 'se':{}, 'ses':{}, 'sses':{}}} # future use
obj_diag_dict={'filters':{}, 'indexes':{}}
obj_diag_dict['filters']['gvmd'] = gvmd
obj_diag_dict['filters']['svmd'] = svmd
obj_diag_dict['filters']['vmd'] = vmd
obj_diag_dict['filters']['meyer'] = meyer
########################################################################################################################
#------------------------------------------------------------------------------
#-------------- SE, SSES, and others
#------------------------------------------------------------------------------
def non_zeros(sig):
    return sig[sig!=0]      

def sig_real_to_env(sig_real):
    """
    obtain the envelope module of the real signal [1],[2]
    """
    sig_real -= np.mean(sig_real)
    
    if np.iscomplexobj(sig_real): # just in case if it is complex valued
        return np.abs(sig_real)
    else:
        return np.abs(hilbert(sig_real)) # sig_real + j * Hilbert(sig_real)

def sig_real_to_se(sig_real):
    """
    obtain the squared envelope (SE) of the real signal [1],[2]
    """
    return sig_real_to_env(sig_real) ** 2

def sig_real_to_ses(sig_real, DC = 0):
    """
    obtain  the squared envelope spectrum [1],[2]
    Note that the SSES (the square of SES) sometimes is also called squared envelope spectrum (SES) 
    for simplicity see ref.[3]
    """ 
    se = sig_real_to_se(sig_real)
    #-- we remove the DC here before Fourier transform
    se = se - np.mean(se) if DC == 0 else se
    return np.abs(np.fft.fft(se)) / len(se)

def sig_real_to_sses(sig_real):
    """
    obtain the square of the squared envelope 
    spectrum (SSES) of the real signal [1],[2]
    Note that the SSES sometimes is also called squared envelope spectrum (SES) 
    for simplicity see ref.[3]
    """
    return sig_real_to_ses(sig_real)**2

########################################################################################################################
#--------------fault indexes independent of  fault frequencies
###############################################################################
# Euler-Mascheroni constant for future use
r = 0.5772156649
no_zero = np.spacing(1) # prevent troubles caused by zeros
###############################################################################
#------------------------------------------------------------------------------
#--------------the kurtosis,see ref. [1],[2]
#------------------------------------------------------------------------------
obj_diag_dict['indexes']['kurt'] = kurtosis
# ------------------------------------------------------------------------------
# --------------the smoothness index, see ref. [1],[2]
# ------------------------------------------------------------------------------
def rsmooth(seq):
    """
    the reciprocal of smoothness index of a sequence

    inputs:
        -seq # 1-d sequence

    ref.:
        Wang, Dong. "Some further thoughts about spectral kurtosis, spectral L2/L1 norm,
        spectral smoothness index and spectral Gini index for characterizing repetitive
        transients." Mechanical Systems and Signal Processing 108 (2018): 360-368.
    """
    seq_non_zeros = non_zeros(seq)
    return np.mean(seq_non_zeros) / gmean(seq_non_zeros)
#--------------store in the dictionary
obj_diag_dict['indexes']['rsmooth'] = rsmooth
################################################################################
# ------------------------------------------------------------------------------
# --------------the Gini index, see ref. [1],[2]
# see also https://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
# ------------------------------------------------------------------------------
def gini(seq):
    """
    the Gini index of a sequence

    inputs:
        -seq # 1-d sequence

    ref.:
        Wang, Dong. "Some further thoughts about spectral kurtosis, spectral L2/L1 norm,
        spectral smoothness index and spectral Gini index for characterizing repetitive
        transients." Mechanical Systems and Signal Processing 108 (2018): 360-368.
    """
    minValue = min(seq)
    if minValue <= 0:
        seq -= minValue  # non-negtives
        seq += np.spacing(1)  # non-zeros
    seq_r = np.sort(seq)  # sort in ascending order
    N = len(seq)
    # python start with 0 for i
    numerator = sum([(2 * (i + 1) - N - 1) * seq_r_i for i, seq_r_i in enumerate(seq_r)])
    denominator = N * sum(seq)  # norm(se, 1)
    return numerator / denominator


# --------------store in the dictionary
obj_diag_dict['indexes']['gini'] = gini
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def gini2(seq):
    """
    ref.:
        [18]	B. Hou, D. Wang, T. Yan, Y. Wang, Z. Peng, and K.-L. Tsui, “Gini Indices II
        and III: Two New Sparsity Measures and Their Applications to Machine Condition
        Monitoring,” IEEE/ASME Trans. Mechatronics, pp. 1–1, 2021, doi: 10.1109/TMECH.2021.3100532.
    """
    minValue = min(seq)
    if minValue <= 0:
        seq -= minValue  # non-negtives
        seq += np.spacing(1)  # non-zeros
    seq_r = np.sort(seq)  # sort in ascending order
    N = len(seq)
    # python start with 0 for i
    M_G1 = sum([(2 * N - 2 * n + 1) * Cn / N ** 2 for n, Cn in enumerate(seq_r)] )
    M_G2 = sum([(2 * n - 1) * Cn / N ** 2 for n, Cn in enumerate(seq_r)])

    return 1 - M_G1 / M_G2


# --------------store in the dictionary
obj_diag_dict['indexes']['gini2'] = gini2
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def gini3(seq):
    """
    ref.:
        [18]	B. Hou, D. Wang, T. Yan, Y. Wang, Z. Peng, and K.-L. Tsui, “Gini Indices II
        and III: Two New Sparsity Measures and Their Applications to Machine Condition
        Monitoring,” IEEE/ASME Trans. Mechatronics, pp. 1–1, 2021, doi: 10.1109/TMECH.2021.3100532.
    """
    minValue = min(seq)
    if minValue <= 0:
        seq -= minValue  # non-negtives
        seq += np.spacing(1)  # non-zeros
    seq_r = np.sort(seq)  # sort in ascending order
    N = len(seq)
    M_x1 = np.mean(seq_r)
    M_G2 = sum([(2 * n - 1) * Cn / N ** 2 for n, Cn in enumerate(seq_r) ])

    return 1 - M_x1 / M_G2
# --------------store in the dictionary
obj_diag_dict['indexes']['gini3'] = gini3
################################################################################
# ------------------------------------------------------------------------------
# --------------the Lp/Lq norm without normalization, see ref. [1],[2]
# ------------------------------------------------------------------------------
def pq_norm(seq, p, q):
    """
    the Lp/Lq norm of a sequence

    inputs:
        -seq # 1-d sequence
        -p   # Lp norm
        -q   # Lq norm

    ref.:
        Wang, Dong. "Some further thoughts about spectral kurtosis, spectral L2/L1 norm,
        spectral smoothness index and spectral Gini index for characterizing repetitive
        transients." Mechanical Systems and Signal Processing 108 (2018): 360-368.
    """
    N = len(seq)
    if q == 0:
        return (N ** (-1 / p)) * norm(seq, p) / gmean(seq)
    else:
        return (N ** (1 / q - 1 / p)) * norm(seq, p) / norm(seq, q)

# --------------store in the dictionary
obj_diag_dict['indexes']['pq_norm'] = pq_norm
################################################################################
# ------------------------------------------------------------------------------
# --------------the Hoyer index, see ref. [4],[5]
# ------------------------------------------------------------------------------
def hoyer(seq):
    """
    the Hoyer index of a sequence

    input:
        seq # 1-d arrary_like sequence

    ref.:
        Hoyer, Patrik O. "Non-negative matrix factorization with sparseness constraints.
        " Journal of machine learning research 5.9 (2004).
    """
    N = len(seq)
    return (np.sqrt(N) - norm(seq, 1) / norm(seq, 2)) / (np.sqrt(N) - 1)
# --------------store in the dictionary
obj_diag_dict['indexes']['hoyer'] = hoyer

################################################################################
# ------------------------------------------------------------------------------
# --------------the entropy, see ref. [6]
# ------------------------------------------------------------------------------
def sEntropy(p):
    """
    the Shannon entropy
    input:
        -p  # 1-d arrary_like of probabilities
    ref.:
        Li, Yongbo, et al. "The entropy algorithm and its variants in the fault
        diagnosis of rotating machinery: A review." Ieee Access 6 (2018): 66723-66741.
    """
    return -sum([pxi * np.log2(pxi + no_zero) for pxi in p])


# --------------store in the dictionary
obj_diag_dict['indexes']['sEntropy'] = sEntropy

def sEntropy_energy(sub_sig):
    """
    the energy/power shannon entropy

    input:
        -sub_sig # list of subsignals e.g. sub_sig = [[sub_sig1], [sub_sig2], ...]

    ref.:
        Su, Houjun, et al. "New method of fault diagnosis of rotating machinery
        based on distance of information entropy." Frontiers of Mechanical Engineering
        6.2 (2011): 249.
    """
    e_list = [sum(sub_s ** 2) for sub_s in sub_sig]
    e_sum = sum(e_list) + no_zero
    p = [e / e_sum for e in e_list]

    return sEntropy(p)
# --------------store in the dictionary
obj_diag_dict['indexes']['sEntropy_energy'] = sEntropy_energy

################################################################################
def negetropy(seq):
    """
    the negetropy of a sequence
    """
    seq += no_zero
    return np.mean( (seq / np.mean(seq) ) * np.log( seq / np.mean(seq)) )
# --------------store in the dictionary
obj_diag_dict['indexes']['negetropy'] = negetropy

###############################################################################
#--------------fault indexes dependent of fault frequencies
###############################################################################
def harEstimation(seq, f_target, harN = 3, fs = 10e3, dev1 = 0.025, 
                  sig_len_original = None, fre2seq = None):
    '''
        inputs:
            -seq                    # half demodulation spectrum 
            -harN = 5               # number of considered harmonics
            -fs = 10e3              # sampling frequency (Hz)
            -f_target = 62.2631     # the target fault characteristic frequency (Hz)
            -dev1 = 0.05            # deviation percentage for estimating a local harmonic
            -dev2 = 0.5             # deviation percentage for estimating a fault harmonic to noise ratio
            -fre2seq: convert frequency to sequency index
        output:
            - amplitude and index estimation of fault characteristic frequency in sequence

        ref.:
            Mo, Zhenling, et al. "Weighted cyclic harmonic-to-noise ratio for rolling
            element bearing fault diagnosis." IEEE Transactions on Instrumentation and
            Measurement 69.2 (2019): 432-442.

        '''
    # ------- initialization
    fHarAmp = np.zeros([harN, 1])  # the amplitudes of the fault frequency harmonics
    fHarInd = np.zeros([harN, 1])  # the indexes of the fault frequency harmonics
    seq = seq.squeeze()
    #--fre2seq: convert frequency to sequency index
    
    if  fre2seq is not None:
        fre2seq =  fre2seq
    elif sig_len_original is not None:
        fre2seq = sig_len_original / fs
    else:
        fre2seq = len(seq) / fs

    fHar1Seq = f_target * fre2seq  # f_target in Hz to f_target in seq
    fHarInd_temp = fHar1Seq  # harmonic index in seq
    delta1 = int(np.round(fHar1Seq * dev1))

    for i in range(harN):
        # -- find the real position of each target harmonic
        f_lw = int(np.floor(fHarInd[0] + fHarInd_temp - delta1))
        f_up = int(np.ceil(fHarInd[0] + fHarInd_temp + delta1))
        # - get a sliced range of seq.
        seq_est = seq[f_lw:f_up]
        seq_len = len(seq_est)
        if seq_len > 1:
            [fHarAmp[i], fmaxIndex] = [np.max(seq_est), np.argmax(seq_est)]
            
        elif seq_len == 1:
            (fHarAmp[i], fmaxIndex) =(seq[f_lw ], 0) if seq[f_lw ]> seq[f_up] else (seq[f_up], 1)
            
        else:
            (fHarAmp[i], fmaxIndex) = (seq[f_lw], 0)
            
        fHarInd_temp = f_lw + fmaxIndex  
        fHarInd[i] = fHarInd_temp  # store the estimiated position

    return fHarAmp, fHarInd

def vanillaSNR(seq, f_target, harN = 3, fs = 10e3, dev1 = 0.025, dev2 = 0.5, 
               sig_len_original = None, fre2seq = None):
    # prevent zero denominator
    inf_preventor = np.spacing(1)
    fHarAmp, fHarInd = harEstimation(seq, f_target, harN , fs, dev1 ,
                                     sig_len_original,fre2seq)
    delta2 = int(np.round(fHarInd[0] * dev2))
    noise = sum(seq[int(fHarInd[0])-delta2 : int(fHarInd[-1]) + delta2]) - sum(fHarAmp)
    return 10 * np.log10( sum(fHarAmp) / (noise + inf_preventor) )

def harL2L1norm(seq, f_target, harN = 3, fs = 10e3, dev1 = 0.025, dev2 = 0.5,
                sig_len_original = None, fre2seq = None):
    _, fHarInd = harEstimation(seq, f_target, harN , fs, dev1 , 
                               sig_len_original,fre2seq)
    delta2 = int(np.round(fHarInd[0] * dev2))
    har_pq = np.zeros([harN, 1])
    for i in range(harN):
        har_seq = seq[int(fHarInd[i])-delta2 : int(fHarInd[i]) + delta2]
        har_pq[i]=pq_norm(har_seq, p=2, q=1)
    return np.mean(har_pq)

def harkurtosis(seq, f_target, harN = 3, fs = 10e3, dev1 = 0.025, dev2 = 0.5,
                sig_len_original = None,fre2seq = None):
    _, fHarInd = harEstimation(seq, f_target, harN , fs, dev1 ,
                               sig_len_original,fre2seq)
    delta2 = int(np.round(fHarInd[0] * dev2))
    har_pq = np.zeros([harN, 1])
    for i in range(harN):
        har_seq = seq[int(fHarInd[i])-delta2 : int(fHarInd[i]) + delta2]
        har_pq[i]=kurtosis(har_seq)
    return np.mean(har_pq)

#------------------------------------------------------------------------------
#--------------cyclic to harmonic ratio (CHNR)
#------------------------------------------------------------------------------
def CHNR(seq, f_target, harN = 3, fs = 10e3, dev1 = 0.025, dev2 = 0.5, 
         sig_len_original = None,fre2seq = None):
    '''
    The cycic harmonic to noise ratio without the white noise threshold
    inputs:
        -seq                    # the real-valued signal
        -harN = 5               # number of considered harmonics
        -fs = 10e3              # sampling frequency (Hz)
        -f_target = 62.2631     # the target fault characteristic frequency (Hz)
        -dev1 = 0.025           # half of the deviation percentage for estimating a local harmonic
        -dev2 = 0.5             # deviation percentage for estimating a fault harmonic to noise ratio
        -
    output:
        -the CHNR of the target frequency
    ref.:
        Mo, Zhenling, et al. "Weighted cyclic harmonic-to-noise ratio for rolling 
        element bearing fault diagnosis." IEEE Transactions on Instrumentation and
        Measurement 69.2 (2019): 432-442.
         
    '''
    inf_preventor = np.spacing(1)   # prevent zero denominator
    fHarAmp, fHarInd = harEstimation(seq, f_target, harN , fs, dev1 ,
                                     sig_len_original,fre2seq)
    delta1 = int(np.round(fHarInd[0] * dev1))
    kN = int(np.round((dev2 - dev1) / (2 * dev1)))  # the number of considered sub-bands around the target harmonic
    fHarUp = np.zeros([kN, harN])   # upper neighboring local maxima of the target harmonic
    fHarLow = np.zeros([kN, harN])  # lower neighboring local maxima of the target harmonic
    chnr = np.zeros([harN, 1])      # the cyclic harmonic to noise ratio

    #-- conisder the neighboring local maxima, i.e., sampling of local maxima
    for i, h_amp, h_ind in enumerate(zip(fHarAmp, fHarInd)):
        for k in range(kN):
            # k= 0, 1, 2, ..., kN
           fHarUp[k,i] = np.max(seq[int(np.floor( h_ind + (2*k + 1) * delta1)): int(np.ceil( h_ind + (2*k + 3) * delta1))])
           fHarLow[k,i] = np.max(seq[int(np.floor( h_ind + (-2*k - 3) * delta1)): int(np.ceil( h_ind + (-2*k -1) * delta1))])
        #-- the cycic harmonic to noise ratio in percentage
        chnr[i] = h_amp / ( (h_amp + sum(fHarUp[:,i]) + sum(fHarLow[:,i])) + inf_preventor )
    #-- other option
    # CHNR[i] = 20 * np.log10( h_amp / (sum(fHarUp[:,i]) + sum(fHarLow[:,i])) + inf_preventor)  )
    return np.mean(chnr)  
 
#--------------store in the dictionary
obj_diag_dict['indexes']['CHNR'] = CHNR

################################################################################
# ------------------------------------------------------------------------------
# -------------- index based on different signal domains
# ------------------------------------------------------------------------------
def fSigDomain(sig_real,  fBase = 'kurt', sigD ='env'):
    if sigD =='env':
        return obj_diag_dict['indexes'][fBase](sig_real_to_env(sig_real))
    elif sigD =='se':
        return obj_diag_dict['indexes'][fBase](sig_real_to_se(sig_real))
    elif sigD == 'ses':
        return obj_diag_dict['indexes'][fBase](sig_real_to_ses(sig_real))
    elif sigD == 'sess':
        return obj_diag_dict['indexes'][fBase](sig_real_to_sses(sig_real))
    elif sigD == 'se_ses':
        return ( obj_diag_dict['indexes'][fBase](sig_real_to_se(sig_real)) +
                obj_diag_dict['indexes'][fBase](sig_real_to_ses(sig_real)) ) / 2

###############################################################################
#-------------- the objective function for rotating machinery fault diagnosis
class Diagnosis():
    def __init__(self, sig=None, fs=1, filter_str='meyer', findex='CHNR', **kwargs):
        """
        the class that can do the followings:
            -provide the objective funtion for the optimizatin algorithm
            -record the optimal fault index and the optimal signal
            -provide the squared envolope spectrum of the optimal signal
            -load the signal from txt file 
        
        inputs:
            -sig            # the real valued signal
            -fs             # sampling frequency
            -filter_str     # filter name,string type
            -findex      # fault index name, string type
        """
        self.sig = sig # real-valued signal
        self.fs = fs # sampling frequency
        self.obj_dict = obj_diag_dict # store all the objective function
        self.filter_str = filter_str
        self.findex = findex
        self.filter = obj_diag_dict['filters'][filter_str] # specific filter
        self.index = obj_diag_dict['indexes'][findex] # specific fault index
        self.index_opt = 0 # record the optimal fault index
        self.sig_opt = [] # record the optimal signal 
        self.full_opt_info = [] # record the optimal variabes and the coressponding obj. values
        self.full_try_info = [] # record the tried variabes and the coressponding obj. values
        
        self.kwargs = kwargs # you may pass other key word arguments here
        
    def load_sig(self, fs=1, path = None):
        """
        load the signal
        support .txt
        
        inputs:
            -fs    # sampling frequency
            -path  # data path
        
        """
        self.fs = fs
        #-
        fileType = file_type(path)
        
        if fileType == '.txt':
            self.sig = np.loadtxt(path)
        else:
            raise ValueError('It is not a .txt file')
    
    def pso_vmd_benchmark(self, variables, tau=0.0, DC=0, init=1, tol=1e-07, maxIterNum=5, **kargs):
        """
        Mixed integer programming by rounding K 
        inputs:
            -sig_real           # the signal to be decomposed
            -alpha              # compactness factor, balancing factor,the penalty of the Tikhonov regularization of each mode
            -omega_init         # the initialized central frquency of each mode (Hz)
            -filter_str         # the filter name 
            -findex          # the fault index name
            -tau = 0.           # noise-tolerance (no strict fidelity enforcement)
            -K = 1              # the number of desired modes
            -DC = 0             # no DC part imposed
            -init = 1           # uniformly initialization
            -tol = 1e-7         # convergence criteria
            -maxIter = 5       # the maximum iteration number used as early stopping to save time
            -kargs             # key words arguments of fault index function
            
        outputs:
            -the largest fault index
        
        ref.:
            Wang, Xian-Bo, Zhi-Xin Yang, and Xiao-An Yan. "Novel particle swarm 
            optimization-based variational mode decomposition method for the 
            fault diagnosis of complex rotating machinery." IEEE/ASME Transactions
            on Mechatronics 23.1 (2017): 68-79.
        """
        #-- in the paper , they use alpha and K as scalar. However, it may be better to use customized values for each mode
        alpha = variables[0]
        K = round(variables[1])
        
        mode, _, _ = self.filter(self.sig, alpha, tau, K, DC, init, tol, maxIterNum)
        # the envelope moduel, which will be used to calcualte envelope energy entropy
        mode_env_list = [sig_real_to_env(mode[i,:]) for i in range(K)]
        # obtain the envelope energy entropy
        mode_en_entropy = self.index(mode_env_list)
        # the fitness
        # note that authors used spectral kurtosis to select the final mode
        # the optimal decomposition is defined by entropy then the optimal sig is selected from 
        # the optimal decompsotion by kurtosis
        #-- get the optimal sig of each call
        mode_kurt_list = [fSigDomain(mode[i,:], 'kurt', 'se') for i in range(K)]
        mode_kurt_max = max(mode_kurt_list)
        ind_max = mode_kurt_list.index(mode_kurt_max)
        
        # record the optimal index 
        if self.index_opt[-1][0] == -np.spacing(1): # if it is empty
            self.index_opt = [[mode_en_entropy, mode_kurt_max]]
            self.full_opt_info.append({'alpha': alpha, 'K': K, 'entropy_min':mode_en_entropy, 'kurt_max':mode_kurt_max})
        else:
            self.index_opt.append([min([self.index_opt[-1][0], mode_en_entropy]), mode_kurt_max])
            # record the optimal signal 
            if mode_en_entropy == self.index_opt[-1][0]:
                self.sig_opt = mode[ind_max,:] 
                self.full_opt_info.append({'alpha': alpha, 'K': K, 'entropy_min':mode_en_entropy, 'kurt_max':mode_kurt_max})
            else:
                self.full_opt_info.append(self.full_opt_info[-1])
        
        self.full_try_info.append({'alpha': alpha, 'K': K, 'entropy_min':mode_en_entropy, 'kurt_max':mode_kurt_max})
        
        
        return mode_en_entropy
        
    def vmd_alike_based(self, alpha, omega_init, tau=0.0, K=1, DC=0, init=3, tol=1e-07, maxIterNum=5, **kargs):
        """
        The vamd alike aglorithm based objective funtion for fault diagnosis
        
        inputs:
            -sig_real           # the signal to be decomposed
            -alpha              # compactness factor, balancing factor,the penalty of the Tikhonov regularization of each mode
            -omega_init         # the initialized central frquency of each mode (Hz)
            -filter_str         # the filter name 
            -findex          # the fault index name
            -tau = 0.           # noise-tolerance (no strict fidelity enforcement)
            -K = 1              # the number of desired modes
            -DC = 0             # no DC part imposed
            -init = 3           # initialize omegas uniformly if omega_init is not none
            -tol = 1e-7         # convergence criteria
            -maxIter = 10       # the maximum iteration number used as early stopping to save time
            -kargs             # key words arguments of fault index function
            
        outputs:
            -the largest fault index
        """
        omega_init = omega_init / self.fs #[0, 0.5fs] is normalized to [0, 0.5] 
        mode, _, _ = self.filter(self.sig, alpha, tau, K, DC, init, tol, maxIterNum, omega_init)
        # the higher the fault index, the more likely a fault has happened
        index_list = [ self.index( mode[i,:], **kargs ) for i in range(K) ] 
        index = max( index_list ) 
        ind_max = index_list.index(index)
        # record the optimal index 
        self.index_opt = max([self.index_opt, index])
        # record the optimal signal 
        self.sig_opt = mode[ind_max,:] if index == self.index_opt else self.sig_opt
        
        return index.item()
    
    def meyer3_based_mealpy(self, b):
        """
        The three meyer wavelet filter based objective funtion for fault diagnosis
        using mealpy algorithms
        inputs:
            -b                  # the boundaries for the filter bank
            
        outputs:
            -the largest fault index (negative value)
        """
        filter_num = 3
        
        #-- you may get other arguments from self.kwargs
        #-- it is convinient if other optimization algorithm only allows one argument

        kwargs_copy = copy.deepcopy(self.kwargs)
        minB = kwargs_copy.pop('minB') # pass minB and delete it
        kwargs = kwargs_copy
        
        # check the value of bounds regarding the bandwidth constraint
        if b[0] + minB > b[1]:
            b[0] = max([np.mean(b) - minB / 2, minB])
            b[1] = b[0] + minB
        
        #-- convert to point index (it will be further normized to [0, pi])
        b = np.array(b) * len(self.sig) / self.fs
        
        mode, _, _ = meyer(self.sig,  b, filter_num)
        # the higher the fault index, the more likely a fault has happened
        
        #--- calcualte indexes of three modes
        index_list= [ self.index( mode[:,i], **kwargs ) for i in range(filter_num) ] 
        index = max( index_list ) 
        ind_max = index_list.index(index)
        
        # record the optimal index 
        if self.index_opt == []: # if it is empty
            self.index_opt.append(index)
            self.full_opt_info.append({'b1': b[0], 'b2': b[1], self.findex:index}) # findex
        else:
            self.index_opt.append(max([self.index_opt[-1], index]))
            # record the optimal signal 
            if index == self.index_opt[-1]:
                self.sig_opt = mode[:,ind_max] 
                self.full_opt_info.append({'b1': b[0], 'b2': b[1], self.findex:index})
            else:
                self.full_opt_info.append(self.full_opt_info[-1])
        
        self.full_try_info.append({'b1': b[0], 'b2': b[1], self.findex:index})
        
        return index#.item()
    
    def meyer3_based(self, b, minB=None, **kwargs):
        """
        The three meyer wavelet filter based objective funtion for fault diagnosis
        used in dbtree search
        
        inputs:
            -b                  # the boundaries for the filter bank
            -filter_str         # the filter name 
            -findex          # the fault index name
            -filter_num         # the number of filters
            -kargs              # key words arguments of fault index function
        
        outputs:
            -the largest fault index
        """
        filter_num = 3
        
        #-- you may get other arguments from self.kwargs
        #-- it is convinient if other optimization algorithm only allows one argument
        
        # check the value of bounds regarding the bandwidth constraint
        if b[0] + minB > b[1]:
            b[0] = max([np.mean(b) - minB / 2, minB])
            b[1] = b[0] + minB
        
        #-- convert to point index (it will be further normized to [0, pi])
        b = np.array(b) * len(self.sig) / self.fs
        
        mode, _, _ = meyer(self.sig,  b, filter_num)
        # the higher the fault index, the more likely a fault has happened
        
        #--- calcualte indexes of three modes
        index_list= [ self.index( mode[:,i], **kwargs ) for i in range(filter_num) ] 
        index = max( index_list ) 
        ind_max = index_list.index(index)
        
        # #-- only care about the bandpass filter
        # ind_max = 1 
        # index = self.index( mode[:,ind_max], **kwargs )

        # record the optimal index 
        self.index_opt = max([self.index_opt, index])
        # record the optimal signal 
        self.sig_opt = mode[:,ind_max] if index == self.index_opt else self.sig_opt
        
        return index.item()
    
    def to_2pi(self, freq):
        """
        map a frequency in [0, fs/2] to the corresponding frequency in [0, 2pi]
        where 2pi = fs, pi = fs / 2
        """
        return freq * 2 * np.pi / self.fs
        
    def to_fs(self, omega):
        """
        map a frequency in [0, pi] to the corresponding frequency in [0, fs/2]
        where 2pi = fs, pi = fs / 2
        """
        return omega * self.fs / (2 * np.pi)
    
    def show_time(self, sig_real = None, title= '', xlabel = 'Time (s)',
                 ylabel = 'Normalized amplitude', figsize = (3.5, 1.8), dpi = 144,
                 fig_save_path= None, fig_format = 'png', fontsize = 8, linewidth = 1, non_text = False):
        '''
        show time domain waveform only
        '''
        data = sig_real if sig_real is not None else self.sig_opt
        fs = self.fs
        
        plt.figure(figsize=figsize, dpi=dpi)
        plt.rc('font', size = fontsize)
        
        data = data / max(abs(data))
        
        Ns = len(data)
        n = np.arange(Ns)  # start = 0, stop = Ns -1, step = 1
        t = n / fs  # time index
        plt.plot(t, data, color = 'xkcd:dark sky blue', linewidth = linewidth)
        
        if non_text:
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        else:
            plt.xlabel(xlabel, fontsize = fontsize + 0.5)
            plt.ylabel(ylabel, fontsize = fontsize + 0.5)
            plt.title(title, fontsize = fontsize + 1)
        
        plt.tight_layout()
        if fig_save_path is not None:
            plt.savefig(fig_save_path, format=fig_format)
        plt.show() 
    
         
    def show_freq(self, sig_real=None, fs = None, title='', xlabel = 'Frequency (Hz)',
                 ylabel = 'Normalized amplitude', f_target=None, figsize = (3.5, 1.8), dpi = 144,
                 fig_save_path= None, fig_format = 'png', fontsize = 8, linewidth = 1, non_text = False):
        """
        show frequency power spectrum only
        flag = half, 0 to fs // 2
        flag = full, -fs//2 to fs //2
        """
        data = sig_real if sig_real is not None else self.sig_opt
        fs = fs if fs is not None else self.fs 

        plt.figure(figsize=figsize, dpi=dpi)
        plt.rc('font', size = fontsize)
        
        Ns = len(data)
        n = np.arange(Ns)  # start = 0, stop = Ns -1, step = 1
        fn = n * fs / Ns # frequency index
        if np.iscomplexobj(data):
            Amp_fn = abs(data)
        else:
            Amp_fn = 2 * abs(np.fft.fft(data)) /Ns
        
        Amp_fn = Amp_fn / max(Amp_fn) 
        
        plt.plot(fn[:len(fn)//2], Amp_fn[:len(fn)//2], color = 'xkcd:dark sky blue', linewidth = linewidth) 
        
        if non_text:
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        else:
            plt.xlabel(xlabel, fontsize = fontsize + 0.5)
            plt.ylabel(ylabel, fontsize = fontsize + 0.5)
            plt.title(title, fontsize = fontsize + 1)

        plt.tight_layout()
        if fig_save_path is not None:
            plt.savefig(fig_save_path, format=fig_format)
        plt.show() 

    
    def show_sses(self, sig_real = None, f_target=None, SSES=True, title='', xlabel = 'Frequency (Hz)',
                 ylabel = 'Normalized amplitude',  figsize = (3.5, 1.8), dpi = 144,
                 fig_save_path= None, fig_format = 'png', fontsize = 8, linewidth = 1, non_text = False ):
        """
        show the square of the squared envelope spectrum
        """
        sig = sig_real if sig_real is not None else self.sig_opt
        ses = sig_real_to_ses(sig)
        #-- normalized by the maximum amplitude
        sesMax = max(ses)
        ses = ses / sesMax
        (sses, label)= (ses**2, 'SES') if SSES else (ses, 'SES') # you may use sses instead

        # plt.figure()
        plt.figure(figsize=figsize, dpi=dpi)
        
        plt.rc('font', size = fontsize)
        
        if f_target is not None:
            harN = 5
            harColor = ['r', 'r', 'r', 'm', 'm']
            harLine = ['--', '-.', ':', '--', '-.']
            point_num = 10
            targetHarAmp = [np.linspace(0, 1.1, point_num ) for i in range(harN) ]
            targetHar = [[f_target + i*f_target for j in range(point_num)] for i in range(harN) ]
            
            for i, (tar, tarAmp) in enumerate(zip(targetHar, targetHarAmp)):
                plt.plot(tar, tarAmp, harColor[i] + harLine[i],  label ='Har'+str(i+1), linewidth=linewidth + 0.3)
                # raise ValueError
                plt.ylim([0, 1.1])
                plt.xlim([0,  7* f_target]) # (harN + 1)
            
        
        Ns = len(sses)
        n = np.arange(Ns)  # start = 0, stop = Ns -1, step = 1
        fn = n * self.fs / Ns # frequency index, Fs / Ns = frequency resoluton
        plt.plot(fn[:Ns//2], sses[:Ns//2], 'b', label = label , linewidth=linewidth)
        
        
        if non_text:
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        else:
            plt.xlabel(xlabel, fontsize = fontsize + 0.5)
            plt.ylabel(ylabel, fontsize = fontsize + 0.5)
            plt.title(title, fontsize = fontsize + 1)    
        plt.legend(fontsize = fontsize - 1)
        
        plt.tight_layout()
        if fig_save_path is not None:
            plt.savefig(fig_save_path, format=fig_format)
        plt.show()  
 
    def plot_ffts(self, sig, sig_opt, opt_dict, mfb=None, boundaries=None, figsize = (3.5, 1.8), dpi = 144,
                 fig_save_path= None, fig_format = 'png', fontsize = 8, linewidth=1, non_text = False ):
        """
        plot the fft of the original signal and the filtered optimal signal
        
        inputs:
            -sig         # the original
            -sig_opt     # the optimal sigal
            -vars_opt    # the optimal decision variables
            -ffindex  # name of the fault index
            -findex_opt  # value of the optimal fault index
        """
        
        sig_fft_amp = abs(np.fft.fft(sig))
        #-- normalized by the maximum amplitude
        amp_max = max(sig_fft_amp)
        sig_fft_amp = sig_fft_amp / amp_max 
        sig_opt_fft_amp = abs(np.fft.fft(sig_opt)) / amp_max 
        
        Ns = len(sig_fft_amp)
        n = np.arange(Ns)  # start = 0, stop = Ns -1, step = 1
        fn = n * self.fs / Ns # frequency index, Fs / Ns = frequency resoluton
        
        # plt.figure()
        plt.figure(figsize=figsize, dpi=dpi)
        
        plt.plot(fn[:Ns//2], sig_fft_amp[:Ns//2], ':', color = 'xkcd:dark sky blue',  label = 'Original' , linewidth=linewidth)
        plt.plot(fn[:Ns//2], sig_opt_fft_amp[:Ns//2], 'b', label = 'Optimal' , linewidth=linewidth)
        
        #-- future use
        if mfb is not None:
            mode_len, mode_num = mfb.shape
            style = ['--', '-.', ':']
            for i in range(mode_num): #magenta, light salmon
                plt.plot(fn[:Ns//2],mfb[:Ns:2, i], style[i], color = 'xkcd:light purple', label = 'filter' + str(i+1), linewidth=linewidth )
                
        if boundaries is not None:
            style_b = ['r--', 'r-.']
            for i in range(len(boundaries)):
                b_x = boundaries[i] * np.ones(10)
                b_y = np.linspace(0, 1, 10)
                plt.plot(b_x, b_y, style_b[i], label = 'b' + str(i+1), linewidth=linewidth)
        
        plt.rc('font', size = fontsize)

        plt.xlim([0, fn[Ns//2]])
        
        if non_text: 
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        else:
            plt.xlabel('Frequency (Hz)', fontsize = fontsize + 0.5)
            plt.ylabel('Normalized amplitude', fontsize = fontsize + 0.5)
            plt.title(str(opt_dict), fontsize = fontsize + 0.5)
            plt.legend(fontsize = fontsize - 2) 
        
        plt.tight_layout()
        if fig_save_path is not None:
            plt.savefig(fig_save_path, format=fig_format)
        plt.show()   
 
if __name__ == '__main__':
    a = np.zeros((1000))
    f = obj_diag_dict['indexes']['gini'] 
    b = f(a)