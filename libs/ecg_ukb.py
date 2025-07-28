""" UKB ECG data loaders and related helpers """

__author__ = "Thomas Kaplan"

import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum
from scipy.io import loadmat
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    LabelBinarizer,
    OneHotEncoder,
)
from torch.utils.data import Dataset
from typing import Optional, List


# Original cohort variables
UKB_ECG_LVMASS_PATH = "ECG_LVmass.csv"
UKB_ECG_LVM_PATH = "ECG.orig_val.parquet.gzip"
# Hypertensive cohort variables
UKB_ECG_HC_PATH = "hypertension_CMR_metrics.all.20250303.csv"

# EID filter (following ECG QC)
UKB_ECG_ECG_IDS_PATH =  "feid.all.20250303.list"
# LVH beat filter (following QC, not using median beats)
UKB_ECG_LVH_BEATS_PATH = (
    "feid_ecg_i_lvh.all.20250303.csv"
)

# Internval validation cohort variables
UKB_ECG_VAL_LVM_PATH = (
    "ECG_LVM.validation.csv"
)
UKB_ECG_VAL_ECG_IDS_PATH = None


# SHIP data
SHIP_ECG_LVM_PATH = "SHIP.parquet.gzip"
SHIP_ECG_HC_PATH = "hypertension_CMR_metrics.ship.20250305.csv"
SHIP_ECG_ECG_IDS_PATH =  "feid.ship.20250305.list"

N_CHANNELS = 8
N_STEPS = 400
N_META = 16
N_LVH_GROUPS = 4

LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
LABELS = ["Normal", "Remodelling", "Eccentric", "Concentric"]


class ECGTarget(Enum):
    MASS = "LVM"
    INDEXED_MASS = "indexed.LVM"
    LVH = "LVM.group"
    HN = "HN.LVH.num"
    VOL = "LVEDV"
    INDEXED_VOL = "indexed.LVEDV"
    MASS_VOL_RATIO = "mass.volume.ratio"
    MASS_VOL = ["LVM", "LVEDV"]
    INDEXED_MASS_VOL = ["indexed.LVM", "indexed.LVEDV"]

    @property
    def is_binary(self):
        return self.value == self.LVH.value

    @property
    def is_categorical(self):
        return self.value in (self.LVH.value, self.HN.value)

    @property
    def is_multireg(self):
        return self.n_out > 1 and not self.is_categorical

    @property
    def labels(self):
        if not self.is_categorical:
            raise Error("Unable to return class labels for non-categorical variable")
        elif self.value == self.HN.value:
            return LABELS
        else:
            raise NotImplementedError(
                f"Unable to return class labels for: {self.value}"
            )

    @property
    def n_out(self):
        if self.value == self.HN.value:
            return len(self.labels)
        elif self.is_binary:
            return 1
        elif isinstance(self.value, list):
            return len(self.value)
        else:
            return 1

    @property
    def hypertension_sample(self):
        return self.value in (
            self.HN.value,
            self.VOL.value,
            self.INDEXED_VOL.value,
            self.MASS_VOL_RATIO.value,
            self.MASS_VOL.value,
            self.INDEXED_MASS_VOL.value,
        )

    def lvh_mass_threshold(self, male):
        """
        See Table 2 (men) and Table 3 (women):
            Petersen, S.E., Aung, N., Sanghvi, M.M. et al. Reference ranges for cardiac structure
            and function using cardiovascular magnetic resonance (CMR) in Caucasians from the UK
            Biobank population cohort. J Cardiovasc Magn Reson 19, 18 (2017).
        """
        if self.value == self.MASS.value:
            return 141 if male else 93
        elif self.value == self.INDEXED_MASS.value:
            return 70 if male else 55
        else:
            raise NotImplementedError(f"Unable to get LVH threshold for: {self.value}")


class ECGMode(Enum):
    H12 = "H12"
    HXYZ = "HXYZ"

def aug_amplitude_scaling(ecg, scale_range=(0.9, 1.1)):
    scale = torch.empty(1).uniform_(*scale_range)
    return ecg * scale
    
def aug_time_shift(ecg, max_shift=50):
    shift = torch.randint(-max_shift, max_shift, (1,)).item()
    return torch.roll(ecg, shifts=shift, dims=0)

def aug_add_gaussian_noise(ecg, std=0.005):
    lead_noise_std = torch.rand(1, ecg.shape[1]) * std
    noise = torch.randn_like(ecg) * lead_noise_std
    return ecg + noise

def aug_random_crop(ecg, max_crop=25, target_length=400):
    crop_start = torch.randint(0, max_crop + 1, (1,)).item()
    crop_end = torch.randint(0, max_crop + 1, (1,)).item()
    ecg = ecg[:, crop_start: ecg.shape[1] - crop_end].unsqueeze(0)
    ecg = F.interpolate(ecg, size=400, mode="linear", align_corners=False).squeeze(0)
    return ecg


class ECG_LVM(Dataset):
    def __init__(
        self,
        ids: list[int],
        mode: ECGMode,
        target: ECGTarget,
        meta_scaler=None,
        load_ecg_meta_kwargs={},
        load_ecg_kwargs={},
        valid_beats_path=UKB_ECG_LVH_BEATS_PATH,
        augment=False,
    ):
        self._ids = ids
        self._target = target
        self._mode = mode
        self._augment = augment

        self._df, self.meta_scaler = _load_ecg_lvm_df(
            filter_ids=ids,
            meta_scaler=meta_scaler,
            **load_ecg_meta_kwargs,
        )
        self._df_orig, _ = _load_ecg_lvm_df(
            filter_ids=ids, zscore_scale=False, **load_ecg_kwargs
        )

        # Exclude the 'f.eid', and our 7 target columns
        self.n_meta = None
        self.n_meta = self._df.shape[1] - 8
        assert (
            self.n_meta == N_META
        ), f"Expected {N_META} metadata columns, got {self.n_meta}"

        self._df['f.eid'] = self._df['f.eid'].astype(int)
        # For each LVH EID, we expand _df by valid beats; adjusting f.eid to FEID_i (i=index of beat)
        self._df_lvh_beats = None
        if valid_beats_path:
            self._df_lvh_beats = pd.read_csv(valid_beats_path)
            self._df_lvh_beats['f.eid'] = self._df_lvh_beats['f.eid'].astype(int)
            df_expanded = self._df.copy().merge(self._df_lvh_beats, on="f.eid", how="inner")
            df_expanded["f.eid"] = df_expanded.apply(
                lambda x: str(int(x["f.eid"])) + "_" + str(int(x["ecg_beat_i"])), axis=1
            )
            df_expanded = df_expanded.drop(columns=["ecg_beat_i"])
            df_result = pd.concat([self._df, df_expanded], ignore_index=True)
            # NOTE: Below reinstate if you don't want dupes per feid
            #df_result = df_result[~df_result['f.eid'].isin(self._df_lvh_beats['f.eid'].unique())]
            has_beat = [int(f.split('_')[0]) for f in df_result['f.eid'] if '_' in str(f)]
            self._df = df_result

        self._ecg_paths = {}

    def aug_random_ecg_transform(self, ecg):
        ecg = aug_random_crop(ecg)
        if torch.rand(1) > 0.5:
            ecg = aug_amplitude_scaling(ecg)
            if torch.rand(1) > 0.75:
                ecg = aug_add_gaussian_noise(ecg)
        else:
            if torch.rand(1) > 0.75:
                ecg = aug_add_gaussian_noise(ecg)
            ecg = aug_amplitude_scaling(ecg)
        return ecg

    def __getitem__(self, index):
        rec = self._df.iloc[index]

        feid = rec["f.eid"]
        ecg_beat_i = -1
        # For e.g. LVH cases, FEID is concatenated with a specific beat to process
        if isinstance(feid, str) and "_" in feid:
            feid, ecg_beat_i = feid.split("_")
            ecg_beat_i = int(ecg_beat_i)

        label = rec[self._target.value]
        if self._target.is_categorical:
            label = label.astype(np.int32)
        else:
            label = label.astype(np.float64)
        if isinstance(label, pd.Series):
            label = torch.Tensor(label.values)

        # Again, exclude leading 'f.eid' (ID) and 'LVM.group' (class)
        meta = rec.values[-self.n_meta :]

        if feid in self._ecg_paths:
            ecg_path = self._ecg_paths[feid]
        else:
            ecg_path = _get_ecg_path(feid)
            self._ecg_paths[feid] = ecg_path
        if ecg_path.endswith('.mat'):
            # SHIP
            mat = loadmat(ecg_path)
            ecg = resample(mat[self._mode.value], 400, axis=1)
            # Expect: ['I','II','III','aVF','aVR','aVF','V1','V2','V3','V4','V5','V6','X','Y','Z']
            #   and pick I, II, V1-V6
            ecg = ecg[[0, 1, 6, 7, 8, 9, 10, 11], :]
        else:
            # UKB
            ecg_files = np.load(ecg_path)
            # Expect: ['I','II','III','avr','avl','avf','V1','V2','V3','V4','V5','V6']
            #   and pick I, II, V1-V6
            ecg_signal = ecg_files[self._mode.value][[0, 1, 6, 7, 8, 9, 10, 11], :, :]
            if ecg_beat_i == -1:
                ecg = np.median(ecg_signal, axis=-1)
            else:
                ecg = ecg_signal[:, :, ecg_beat_i]
        assert ecg.shape == (N_CHANNELS, N_STEPS), f"Unexpected shape: {ecg.shape}"

        # NOTE: Simple scaling to get closer to unit range (as in minmax/stddev scaling)
        ecg = ecg / 1000.0

        ecg = torch.Tensor(ecg.astype(np.float32))
        if self._augment:
            ecg = self.aug_random_ecg_transform(ecg)

        return (
            ecg,
            meta.astype(np.float32),
            label,
            index,
        )

    def __len__(self):
        return self._df.shape[0]

    def get_record(self, index):
        df_rec = self._df.iloc[index]
        feid = df_rec['f.eid']
        if isinstance(feid, str) and "_" in feid:
            feid = feid.split('_')[0]
        feid = int(feid)
        orig_recs = self._df_orig[self._df_orig["f.eid"] == feid]
        return orig_recs.iloc[0]

    def get_weighted_sampler(self, undersample=False):
        if not self._target.is_categorical:
            raise ValueError(
                f"Unable to use weighted sampler for non-categorical target: {self._target.value}"
            )
        return _get_random_sampler(
            self._df[self._target.value], undersample=undersample
        )

    def get_prior(self, vals=None):
        if vals is None:
            if not self._target.is_categorical:
                raise ValueError(
                    f"Unable to get class prior for non-categorical target: {self._target.value}"
                )
            vals = self._df[self._target.value]
        counts = np.unique(vals, return_counts=True)[1]
        return counts / counts.sum()

    def get_lvh_prior(self):
        return self.get_prior(vals=self._df[ECGTarget.LVH.value])

    def get_class_weights(self):
        if not self._target.is_categorical:
            raise ValueError(
                f"Unable to get positive weight for non-categorical target: {self._target.value}"
            )
        return _get_class_weights(self._df[self._target.value])

    def get_norm_suff_stats(self):
        if self._target.is_categorical:
            raise ValueError(
                f"Unable to get sufficient statistics for categorical target: {self._target.value}"
            )
        vals = self._df[self._target.value]
        return np.mean(vals), np.std(vals, ddof=0)


def _get_class_weights(target):
    counts = np.unique(target, return_counts=True)[1]
    return 1.0 / counts


def _get_random_sampler(target, undersample=False):
    target = target.astype(int)
    weights = _get_class_weights(target)
    counts = 1.0 / weights
    samples_weight = weights[target]
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    if undersample:
        # Undersample
        num_samples = int(len(counts) * min(counts))
        replacement = False
    else:
        # Oversample
        num_samples = len(samples_weight)
        replacement = True
    print(
        f"Random sampler, undersample:{undersample}, replacement:{replacement}, weights:{weights}"
    )
    return torch.utils.data.WeightedRandomSampler(samples_weight, num_samples)


def _load_hn_df(path=UKB_ECG_HC_PATH):
    df_hn = pd.read_csv(path)
    df_hn["HN.LVH.num"] -= 1  # {1..4} -> {0..3}, for 0 indexing on predictive dist
    return df_hn[["f.eid", "HN.LVH.num", "LVEDV", "indexed.LVEDV", "mass.volume.ratio"]]


def _load_ecg_lvm_df(
    path=UKB_ECG_LVM_PATH,
    mass_path=None,#UKB_ECG_LVMASS_PATH,
    htn_path=UKB_ECG_HC_PATH,
    eids_path=UKB_ECG_ECG_IDS_PATH,
    filter_ids: Optional[set] = None,
    zscore_scale=True,
    meta_scaler=None,
    col_order: List = None,
):
    df_ = pd.read_parquet(path)
    df_["LVM.group"] = (df_["LVM.group"] == 1).astype(int)

    df_clin_markers = df_.copy()

    if not filter_ids:
        # By default, we load set of all eids with valid (QC'd) ECGs
        filter_ids = set(pd.read_csv(eids_path, header=None).loc[:, 0].values)
    # Filter by id, e.g. if this is a split/fold
    df_clin_markers = df_clin_markers[df_clin_markers["f.eid"].isin(filter_ids)]

    # One-hot encode for the only non-binary categorical column
    sm_df = pd.get_dummies(df_clin_markers["smoking.status"])
    sm_df.columns = [f"smoking.status.{int(i)}" for i in sm_df.columns.values]
    for i in range(3):
        sm_col = f"smoking.status.{i}"
        if not sm_col in sm_df.columns.values:
            sm_df[sm_col] = 0
        else:
            sm_df[sm_col] = sm_df[sm_col].astype(int)

    df_clin_markers = pd.concat([df_clin_markers, sm_df.astype(int)], axis=1).drop(
        columns=["smoking.status"]
    )

    binarizer = LabelBinarizer()
    binary_cols = [
        "htn.defined",
        "diabetes_final",
        "chol_final",
        "alcohol.status",
        "ethnicity",
    ]
    for col in binary_cols:
        if df_clin_markers[col].isna().any():
            print(f"{col} has NA values, mode imputing..")
            mode_value = df_clin_markers[col].mode()[0]
            df_clin_markers[col].fillna(mode_value, inplace=True)
        xs = df_clin_markers[col]
        df_clin_markers[col] = binarizer.fit_transform(xs)

    # z-score scaling
    if zscore_scale:
        cont_cols = [
            "Av_SBP",
            "Av_DBP",
            "Age",
            "BMI",
            "non.hdl.chol",
            "total.chol",
            "Ventricular_rate",
        ]
        df_to_scale = df_clin_markers[cont_cols]
        for col in cont_cols:
            df_to_scale[col] = df_to_scale[col].fillna(df_to_scale[col].median())
        if meta_scaler is None:
            meta_scaler = MinMaxScaler()
            meta_scaler.fit(df_to_scale)
        df_zs = pd.DataFrame(
            meta_scaler.transform(df_to_scale), columns=df_to_scale.columns
        )
        df_clin_markers = pd.concat([
            df_clin_markers.drop(columns=cont_cols).reset_index(drop=True),
            df_zs.reset_index(drop=True)
        ], axis=1)
        df_clin_markers = pd.concat([
            df_clin_markers.drop(columns=cont_cols).reset_index(drop=True),
            df_zs.reset_index(drop=True)
        ], axis=1)

    # Incorporate mass, otherwise assumed to be defined already
    if mass_path is not None:
        df_mass = pd.read_csv(mass_path)
        df_clin_markers = df_clin_markers.merge(
            df_mass.drop("LVM.group", axis=1), how="inner", on="f.eid"
        )

    # Incorporate HN class, otherwise assumed to be defined already
    if htn_path is not None:
        df_hn = _load_hn_df(htn_path).drop_duplicates(subset=['f.eid'])
        df_clin_markers = df_clin_markers.merge(df_hn, on="f.eid", how="left")
        df_clin_markers["HN.LVH.num"] = df_clin_markers["HN.LVH.num"].astype("Int64")

    if col_order is None:
        # Reorganise columns to keep outcomes/targets at the left
        cols = df_clin_markers.columns.tolist()
        #df_clin_markers = df_clin_markers[cols[:2] + cols[-6:] + cols[2:-6]]
        df_clin_markers = df_clin_markers[cols[:4] + cols[-4:] + cols[4:-4]]
    else:
        df_clin_markers = df_clin_markers[col_order]

    df_clin_markers['f.eid'] = df_clin_markers['f.eid'].astype(int)
    return df_clin_markers, meta_scaler


def _get_ecg_path(feid):
    feid = str(int(feid))
    orig_npz = f"processed/{feid[0]}0xxxxx/{feid}_20205_2_0_markers.npz",
    if os.path.isfile(orig_npz):
        return orig_npz
    int_val_npz = f"processed/{feid}_20205_2_0_markers.npz"
    if os.path.isfile(int_val_npz):
        return int_val_npz
    return f"MAT/{feid}.mat"

def create_ecg_splits(
    mode: ECGMode,
    target: ECGTarget,
    train_val_test_splits: tuple[int, int, int],
    seed: Optional[int] = None,
    verbose=False,
    external_val=False,
    external_tune=False,
    external_aug_train=True,
):

    assert (
        np.sum(train_val_test_splits) == 1
    ), "Invalid total of split ratios, expected 1"
    train_split, val_split, test_split = train_val_test_splits
    adj_val_split = val_split / (val_split + train_split)

    df_target = _load_ecg_lvm_df()[0]
    if target.hypertension_sample:
        # If HN, sub-sample accordingly
        df_target = df_target[~df_target[ECGTarget.HN.value].isna()]
    df_target = df_target[
        ["f.eid", *[e.value for e in ECGTarget if e.n_out == 1 or e.is_categorical]]
    ]

    stratify = lambda df: df[target.value] if target.is_categorical else None
    df_train_val, df_test = train_test_split(
        df_target,
        test_size=test_split,
        stratify=stratify(df_target),
        random_state=seed,
    )
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=adj_val_split,
        stratify=stratify(df_train_val),
        random_state=seed,
    )

    # TODO: Refactor below if we don't do split-specific stuff to Dataset, likely the case...
    _to_set = lambda ids: set(ids.unique())
    train_loader = ECG_LVM(_to_set(df_train["f.eid"]), mode, target)
    val_loader = ECG_LVM(
        _to_set(df_val["f.eid"]),
        mode,
        target,
        meta_scaler=train_loader.meta_scaler,
        valid_beats_path=None,
    )
    test_loader = ECG_LVM(
        _to_set(df_test["f.eid"]),
        mode,
        target,
        meta_scaler=train_loader.meta_scaler,
        valid_beats_path=None,
    )

    #get_cols_meta = lambda dset: dset._df.columns[-dset.n_meta :]
    get_cols_meta = lambda dset: dset._df.columns.values
    train_cols = get_cols_meta(train_loader)

    ext_val_loader = None
    load_ecg_meta_ext_val_kwargs = {
        # Validation CSV
        "path": SHIP_ECG_LVM_PATH,
        # Above contains all columns we need
        "mass_path": None,
        "htn_path": SHIP_ECG_HC_PATH,
        # Retain column ordering from our training set - very important
        "col_order": train_cols
    }
    load_ecg_ext_val_kwargs = {
        "eids_path": SHIP_ECG_ECG_IDS_PATH,
        "path": SHIP_ECG_LVM_PATH,
        "htn_path": SHIP_ECG_HC_PATH,
    }
    if external_val:
        print('external_val')
        df_val_target, _ = _load_ecg_lvm_df(**load_ecg_ext_val_kwargs)
        ext_val_loader = ECG_LVM(
            _to_set(df_val_target["f.eid"]),
            mode,
            target,
            meta_scaler=train_loader.meta_scaler,
            load_ecg_meta_kwargs=load_ecg_meta_ext_val_kwargs,
            load_ecg_kwargs=load_ecg_ext_val_kwargs,
            valid_beats_path=None,
        )
        assert (
            (ext_val_loader._df.columns == train_loader._df.columns).all()
        ), "Column mismatch between training/validation"
    elif external_tune:
        #train_split, val_split, test_split = 0.7, 0.15, 0.15
        train_split, val_split, test_split = 0.6, 0.2, 0.2
        adj_val_split = val_split / (val_split + train_split)

        df_ext_target, _ = _load_ecg_lvm_df(**load_ecg_ext_val_kwargs)
        if target.hypertension_sample:
            # If HN, sub-sample accordingly
            df_ext_target = df_ext_target[~df_ext_target[ECGTarget.HN.value].isna()]
        df_ext_target = df_ext_target[
            ["f.eid", *[e.value for e in ECGTarget if e.n_out == 1 or e.is_categorical]]
        ]
        ids_ext = _to_set(df_ext_target["f.eid"]),

        df_ext_train_val, df_ext_test = train_test_split(
            df_ext_target,
            test_size=test_split,
            stratify=stratify(df_ext_target),
            random_state=seed,
        )
        df_ext_train, df_ext_val = train_test_split(
            df_ext_train_val,
            test_size=adj_val_split,
            stratify=stratify(df_ext_train_val),
            random_state=seed,
        )
        train_loader = ECG_LVM(
            _to_set(df_ext_train["f.eid"]),
            mode,
            target,
            meta_scaler=train_loader.meta_scaler,
            load_ecg_meta_kwargs=load_ecg_meta_ext_val_kwargs,
            load_ecg_kwargs=load_ecg_ext_val_kwargs,
            valid_beats_path=None,
            augment=external_aug_train,
        )
        val_loader = ECG_LVM(
            _to_set(df_ext_val["f.eid"]),
            mode,
            target,
            meta_scaler=train_loader.meta_scaler,
            load_ecg_meta_kwargs=load_ecg_meta_ext_val_kwargs,
            load_ecg_kwargs=load_ecg_ext_val_kwargs,
            valid_beats_path=None,
        )
        test_loader = ECG_LVM(
            _to_set(df_ext_test["f.eid"]),
            mode,
            target,
            meta_scaler=train_loader.meta_scaler,
            load_ecg_meta_kwargs=load_ecg_meta_ext_val_kwargs,
            load_ecg_kwargs=load_ecg_ext_val_kwargs,
            valid_beats_path=None,
        )

    if verbose:
        print("Sanity checking meta columns...")
        print("Train:", train_cols)
        print("Val:", get_cols_meta(val_loader))
        print("Test:", get_cols_meta(test_loader))
        if not ext_val_loader is None:
            print("Ext. Val:", get_cols_meta(ext_val_loader))
        print("External,", "val?", external_val, "tune?", external_tune)

    return [train_loader, val_loader, test_loader, ext_val_loader]

