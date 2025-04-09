import os
import glob
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import nibabel as nib
import hcp_utils as hcp
from nilearn import plotting as nlp
from nilearn import surface as nsf
from nilearn.connectome import ConnectivityMeasure
from nipype.interfaces.workbench import CiftiSmooth
from tqdm import tqdm
from scipy import stats
from scipy import signal
from nilearn import image
import scipy.fftpack
import threading
from nilearn.masking import compute_brain_mask, apply_mask, unmask
import multiprocessing

# location
save_vol_dir = '/home/yifeis/swin_bsp/hcp_data_prepro_6/'
loc_dir = '/home/yifeis/swin_bsp/hcp_data_prepro_6/'
loc_vol_dir = '/home/yifeis/swin_bsp/hcp_data_prepro_6/'

with open('../../HCP/unrelated_sub_hcp.txt') as f:
# with open('subjects.txt') as f:
    subjects = f.readlines()
subjects = [x.strip() for x in subjects]
print(f'Focus on {len(subjects)} subjects.')


'''
    Functions
'''
# read fMRI data
def read_fmri(dir):
    return nib.load(dir).get_fdata()

# spatial smoothing using ciftiSmooth
def spatial_smooth(raw_dir, surf_r_dir, surf_l_dir, ss_dir, fwhm=6, vol_size=2):
    sigma = fwhm/(np.sqrt(8 * np.log(2)) * vol_size)
    # rest
    smooth = CiftiSmooth(in_file = raw_dir,
                         sigma_surf = sigma,
                         sigma_vol = sigma,
                         direction = 'COLUMN',
                         right_surf = surf_r_dir,
                         left_surf = surf_l_dir,
                         out_file = ss_dir)
    os.system(smooth.cmdline)

# band-pass filter for temporal filtering
def bandpass_filter(img, TR):
    # data = img.T
    data = img.get_fdata(dtype='float32')
    hp_freq = 0.01 # Hz
    lp_freq = 0.1 # Hz
    fs = 1 / TR # sampling rate, TR in s, fs in Hz
    timepoints = data.shape[-1]
    F = np.zeros(timepoints)
    lowidx = timepoints // 2 + 1
    if lp_freq > 0: # map cutoff frequencies to corresponding index
        lowidx = int(np.round(lp_freq / fs * timepoints))
    highidx = 0
    if hp_freq > 0: # map cutoff frequencies to corresponding index
        highidx = int(np.round(hp_freq / fs * timepoints))
    F[highidx:lowidx] = 1 # let signal with freq between lower and upper bound pass and remove others
    F = ((F + F[::-1]) > 0).astype(int) ### need double-check
    filtered_data = np.zeros(data.shape)
    if np.all(F == 1):
        filtered_data = data
    else:
        filtered_data = np.real(np.fft.ifftn(np.fft.fftn(data) * F))

    filtered_data = filtered_data.astype('float32')
    # ensure the all zero voxels still have zeros
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                if np.all(data[i, j, k, :] == 0):
                    filtered_data[i, j, k, :] *= 0
    return filtered_data


def temporal_filter(ss_dir, tf_out, tr=0.72):
    ss_cifti = nib.load(ss_dir)
    mtx = bandpass_filter(ss_cifti, tr)
    print(mtx.shape)
    mtx = mtx.astype(np.float32)
    nib.save(nib.Cifti2Image(mtx, ss_cifti.header, ss_cifti.nifti_header, ss_cifti.extra, ss_cifti.file_map), tf_out)

# main
labels = hcp.mmp.labels
label_values = list(labels.values())
for sub in subjects:
    # get the fmri files for this subject
    fmri_files = glob.glob(loc_dir + sub+'/*.nii')
    fmri_files.sort()
    surf_files = glob.glob(loc_dir + sub+'/*.gii')
    surf_files.sort()
    if len(fmri_files)==0 or len(surf_files)==0:
        continue
    l_surf = surf_files[0]
    r_surf = surf_files[1]
    print(l_surf)
    print(r_surf)
    # preprocessing
    prepro_files = []
    progress_bar = tqdm(range(len(fmri_files)))
    for f in fmri_files:
        # spatial smoothing
        ss_out = f.split('_Atlas')[0] + '_smoothed.dtseries.nii'
        spatial_smooth(f, l_surf, r_surf, ss_out)
        # temporal filtering
        tf_out = f.split('_Atlas')[0] + '_temporal_filtered.dtseries.nii'
        temporal_filter(ss_out, tf_out)