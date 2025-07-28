""" Training (including fine-tuning) and evaluation of of AI-ECG models for LVH in UKB/SHIP """

__author__ = "Thomas Kaplan"

import argparse
import collections
import datetime
import functools
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import sys
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, TypedDict

try:
    import ray.cloudpickle as pickle
    from ray import tune
    from ray import train as ray_train
    from ray.train import Checkpoint, get_checkpoint
    from ray.tune.schedulers import ASHAScheduler
except (ModuleNotFoundError, ImportError) as err:
    print("Unable to import Raytune (ray):", err, flush=True)

from libs.ecg_ukb import (
    create_ecg_splits,
    ECGMode,
    ECGTarget,
    N_STEPS,
    N_CHANNELS,
    N_META,
)
from libs.model import FCN1D, FCN1DConfig, LogCoshLoss, GaussInvCDFLoss
from libs.soto2022 import resnet18, resnet34, RESNET_LAYOUTS
from libs.eval import (
    print_classification_performance_summary,
    print_regression_performance_summary,
    lvh_threshold_learning,
    lvm_recalibration,
)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

SEED = 240302
torch.manual_seed(SEED)
TORCH_SEED = torch.Generator().manual_seed(SEED)
np.random.seed(SEED)

NUM_WORKERS = 4

SPLITS = (0.7, 0.15, 0.15)
BATCH_SIZE = 64
N_EPOCHS = 200
N_EPOCHS_TRANSF = 500

CHECKPOINT_FILE = "data.pkl"

CRITERION = {
    "LogCosh": LogCoshLoss,
    "Huber": functools.partial(torch.nn.HuberLoss, delta=1.5),
    "GaussInvCDF": GaussInvCDFLoss,
    "BCELogits": nn.BCEWithLogitsLoss,
    "CrossEntropy": nn.CrossEntropyLoss,
}

class EarlyStop:
    """Simple wrapper for early stopping on converging validation loss"""

    def __init__(self, patience: int = 7, delta: float = 0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.inf
        self.stop = False

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
            return True

        return False

    def reset_count(self):
        self.counter = 0


def load_data(
    batch_size: int,
    target: ECGTarget,
    splits: [int, int, int],
    oversample: Optional[bool] = False,
    undersample: Optional[bool] = False,
    external_val=False,
    external_tune=False,
    external_aug_train=False,
    num_workers=NUM_WORKERS
) -> [DataLoader, DataLoader, DataLoader]:
    print("Loading data...", "val?", external_val, "tune?", external_tune)
    train_set, val_set, test_set, external_val_set = create_ecg_splits(
        ECGMode.H12,
        target,
        splits,
        seed=SEED,
        verbose=True,
        external_val=external_val,
        external_tune=external_tune,
        external_aug_train=external_aug_train,
    )

    sampler = None
    if oversample:
        sampler = train_set.get_weighted_sampler(undersample=False)
    elif undersample:
        sampler = train_set.get_weighted_sampler(undersample=True)

    train_loader = DataLoader(
        train_set,
        batch_size,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=sampler is None,
    )
    val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(
        test_set, batch_size, shuffle=False, num_workers=num_workers
    )
    ext_val_loader = None
    if external_val:
        ext_val_loader = DataLoader(
            external_val_set, batch_size, shuffle=False, num_workers=num_workers
        )

    return train_loader, val_loader, test_loader, ext_val_loader


def evaluate(
    target: ECGTarget,
    model,
    test_loader: DataLoader,
    class_prior: Optional[list[float, float]] = None,
    thresh_target: ECGTarget = None,
    thresh_model = None,
    recalib_model = None,
):
    print("Target?", target,
          "Threshold?", thresh_model is not None,
          "Recalibration ?", recalib_model is not None)

    # Evaluate model on test set
    model.eval()
    out = []
    indices = []
    for X_batch, X_meta_batch, y_batch, inds in test_loader:
        with torch.no_grad():
            ys = model(X_batch, X_meta_batch)
        if target.is_binary:
            preds = nn.functional.sigmoid(ys).flatten()
            out.append(torch.vstack([preds, y_batch]))
        elif target.is_categorical:
            out.append(torch.concat([ys, y_batch.unsqueeze(1)], axis=1))
        elif target.n_out > 1:
            out.append(torch.concat([ys, y_batch], axis=1))
        else:
            preds = ys.flatten()
            out.append(torch.vstack([preds, y_batch]))
        indices.append(inds.detach().numpy())

    indices = list(itertools.chain(*indices))

    inds_is_m = [test_loader.dataset.get_record(i)['Sex'] == 1 for i in indices]
    inds_lvh = [test_loader.dataset.get_record(i)['LVM.group'] for i in indices]
    inds_ilvm = [test_loader.dataset.get_record(i)['indexed.LVM'] for i in indices]
    inds_eid = [test_loader.dataset.get_record(i)['f.eid'] for i in indices]

    if target.n_out > 1:
        predictions = torch.vstack(out).detach().numpy()
    else:
        predictions = torch.hstack(out).detach().numpy().T

    results = {'eid': inds_eid, 'is_m': inds_is_m, 'lvh_true': inds_lvh, 'ilvm_true': inds_ilvm}

    if target.value in (ECGTarget.MASS.value, ECGTarget.INDEXED_MASS.value):
        results['ilvm_pred'] = predictions[:, 0].copy()

    if recalib_model is not None:
        X_recalib = np.vstack([inds_is_m, predictions[:, 0]]).T
        recalib_predictions = recalib_model.predict(X_recalib)
        predictions[:, 0] = recalib_predictions
        results['ilvm_pred_recalib'] = recalib_predictions

    if not thresh_model is None:
        print("Adjusting predictions for threshold model")
        # Pipeline through the threshold model, using sex covariate, to update proba
        X_thresh = np.vstack([inds_is_m, predictions[:, 0]]).T
        thresh_preds = thresh_model.predict_proba(X_thresh)
        predictions[:, 0] = thresh_preds[:, 1]
        predictions[:, 1] = inds_lvh
        target = thresh_target

    # Results depend on target variable
    if target.is_binary:
        print("EVALUATION (binary, default proba)")
        y_pred_proba = predictions[:, 0]
        results['lvh_proba'] = y_pred_proba.copy()
        y_pred = (predictions[:, 0] > 0.5).astype(int)
        y_true = predictions[:, 1].astype(int)
        print_classification_performance_summary(
            y_true, y_pred, y_pred_proba, multi_class=False
        )
        if class_prior is not None:
            print("EVALUTION (prior-adjusted proba)")
            print("Prior:", class_prior)
            p_train = class_prior[1]
            logit_model = np.log(predictions[:, 0] / (1 - predictions[:, 0]))
            logit_prior = np.log(p_train / (1 - p_train))
            logit_adjusted = logit_model - logit_prior
            p_adjusted = 1 / (1 + np.exp(-logit_adjusted))
            predictions[:, 0] = p_adjusted
            results['lvh_proba_prioradj'] = p_adjusted.copy()

            y_pred_proba = predictions[:, 0]
            y_pred = (predictions[:, 0] > 0.5).astype(np.int32)
            y_true = predictions[:, 1].round()
            rocs = print_classification_performance_summary(
                y_true, y_pred, y_pred_proba, multi_class=False
            )

    elif target.is_categorical:
        print("EVALUATION (multi-cat, default proba)")
        y_pred_proba = predictions[:, :-1]
        y_pred = y_pred_proba.argmax(axis=1)
        y_true = predictions[:, -1]
        try:
            labels = target.labels
        except:
            # TODO: Don't swallow this error
            labels = []
        print_classification_performance_summary(
            y_true, y_pred, y_pred_proba, multi_class=True, labels=labels
        )

        if class_prior is not None:
            print("EVALUTION (prior-adjusted proba)")
            raise NotImplementedError("Unsupported prior adjustment for multi-classification")

    elif target.n_out > 1:
        print("EVALUATION")
        for i, v in enumerate(target.value):
            print(i, v)
            y_pred, y_true = predictions[:, i], predictions[:, i + 2]
            print_regression_performance_summary(y_true, y_pred)
    else:
        print("EVALUATION")
        y_pred, y_true = predictions[:, 0], predictions[:, 1]
        print_regression_performance_summary(y_true, y_pred)

        if target.value in (ECGTarget.MASS.value, ECGTarget.INDEXED_MASS.value):
            print("EVALUATION (using LVH cutoffs -> classification)")
            y_pred = [p >= target.lvh_mass_threshold(is_m) for p, is_m in zip(y_pred, inds_is_m)]
            y_pred = np.array(y_pred).astype(np.int32)
            results['lvh_pred_cutoff'] = y_pred
            y_true = np.array(inds_lvh).astype(np.int32)
            print_classification_performance_summary(
                y_true, y_pred, y_pred, multi_class=False
            )

    results_df = pd.DataFrame(results)
    for col in ('eid', 'lvh_true', 'is_m'):
        results_df[col] = results_df[col].astype(int)
    return results_df

def get_model(config: dict):

    n_meta = 0 if config["excl_meta"] else N_META
    n_out = config["target"].n_out
    multi_reg = n_out > 1 and not config["target"].is_categorical

    resnet_config = config.get("soto2022")
    if resnet_config is not None:
        resnet_kwargs = {
            "n_meta": n_meta,
            "num_classes": n_out,
            "conv_dropout": config["fcn_conv_dropout"],
            "lin_dropout": config["fcn_linear_dropout"],
            "multi_reg": multi_reg,
        }
        if resnet_config == "resnet18":
            return resnet18(**resnet_kwargs)
        elif resnet_config == "resnet34":
            return resnet34(**resnet_kwargs)
        else:
            raise ValueError(
                f"No ResNet configuration '{resnet_config}' from Soto et al. (2022)"
            )

    return FCN1D(
        N_CHANNELS,
        N_STEPS,
        n_meta,
        n_out,
        config["fcn_config"],
        config["fcn_batch_norm"],
        config["fcn_max_pool"],
        config["fcn_conv_dropout"],
        config["fcn_linear_dropout"],
        multi_reg=multi_reg,
    )


def train(
    target: ECGTarget,
    config: dict,
    checkpoint_path: Optional[str] = None,
    # NOTE: Keep early_stop_n a multiple of reduce_lr_n if not resetting count between LR steps
    early_stop_n: Optional[int] = 20,
    reduce_lr_n: Optional[int] = 10,
    n_epochs: Optional[int] = N_EPOCHS,
    verbose: bool = False,
    hyper: bool = False,
    existing_state_path: Optional[str] = None,
    existing_model = None,
    existing_optimizer = None,
    external_tune = False
):

    train_loader, val_loader, test_loader, _ = load_data(
        config["batch_size"],
        target,
        SPLITS,
        config.get("oversample", False),
        config.get("undersample", False),
        external_tune=external_tune
    )

    if target.is_binary:
        criterion_cls = config.get("criterion", nn.BCEWithLogitsLoss)
        pos_weight = None
        if config.get("loss_weighting", False):
            weights = train_loader.dataset.get_class_weights()
            pos_weight = weights[0] / weights[1]
            pos_weight = torch.tensor(pos_weight)
        criterion = criterion_cls(pos_weight=pos_weight)
    elif target.is_categorical:
        if config.get("loss_weighting", False):
            raise NotImplementedError(
                "Unsupported class weighting for multinomial target"
            )
        criterion_cls = config.get("criterion", nn.CrossEntropyLoss)
        criterion = criterion_cls()
    else:
        criterion_cls = config.get("criterion", LogCoshLoss)
        if criterion_cls == GaussInvCDFLoss:
            mean, std = train_loader.dataset.get_norm_suff_stats()
            criterion = criterion_cls(mean, std)
        else:
            criterion = criterion_cls()

    if existing_model is not None:
        model = existing_model
    else:
        model = get_model(config)
        if existing_state_path is not None:
            print("Loading:", existing_state_path)
            model.load_state_dict(torch.load(existing_state_path))

    best_optimizer_state = None
    if existing_optimizer is not None:
        optimizer = existing_optimizer
    elif isinstance(config.get("optimizer_params"), list):
        # Params pre-specified, so ensure to use these, not model.paramters()
        optimizer = config["optimizer_cls"](config["optimizer_params"])
    else:
        optimizer = config["optimizer_cls"](
            model.parameters(),
            lr=config["learning_rate"],
            **config.get("optimizer_params", {}),
        )

    prev_lr = None
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=reduce_lr_n,
        min_lr=1e-8,
    )
    early_stopper = EarlyStop(patience=early_stop_n)

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Training configuration:", flush=True)
        print(model, flush=True)
        print(f"Params, total={total_params}, trainable={train_params}", flush=True)
        print(criterion, flush=True)
        print(optimizer, flush=True)
        print(scheduler, flush=True)
        print("Saving weights to:", checkpoint_path, flush=True)
        print(
            f"Early stop after {early_stop_n}, reduce LR after {reduce_lr_n}",
            flush=True,
        )

    for epoch in tqdm(range(n_epochs)):
        model.train(True)
        losses = 0.0
        for X_batch, X_meta_batch, y_batch, _ in train_loader:
            # NOTE: Below is handy for debugging sampling distributions
            # if verbose:
            #   print(collections.Counter(y_batch.detach().numpy()))
            optimizer.zero_grad()
            ys = model(X_batch, X_meta_batch)
            if not target.is_binary and target.is_categorical:
                loss = criterion(ys, y_batch)
            elif target.is_multireg:
                loss = criterion(ys, y_batch.float())
            else:
                loss = criterion(ys, y_batch.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            losses += loss.item()

        loss = losses / len(train_loader)

        model.eval()
        vlosses = []
        for X_batch, X_meta_batch, y_batch, _ in val_loader:
            with torch.no_grad():
                ys = model(X_batch, X_meta_batch)
                if target.is_binary or not target.is_categorical:
                    y_batch = y_batch.float().unsqueeze(1)
                elif target.is_multireg:
                    y_batch = y_batch.float()
                vlosses.append(criterion(ys, y_batch).item())
        vloss = np.mean(vlosses)
        scheduler.step(vloss)
        best_vloss = early_stopper(vloss)

        epoch_lr = optimizer.param_groups[0]["lr"]
        # NOTE: We might re-introduce below if we truly expect to skip multiple LR steps
        # if prev_lr is not None and epoch_lr < prev_lr:
        #     if verbose:
        #         print("Resetting early-stopping count (LR has reduced)", flush=True)
        #     early_stopper.reset_count()
        prev_lr = epoch_lr
        if verbose:
            print(
                f"E[{epoch+1}/{n_epochs}], Loss: {loss:.4f}, V. Loss: {vloss:.4f}, LR: {epoch_lr}",
                flush=True,
            )

        if hyper:
            checkpoint_data = {
                "last_epoch": epoch,
                "model_state_dict": model.state_dict(),
            }
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / CHECKPOINT_FILE
                with open(data_path, "wb") as fp:
                    pickle.dump(checkpoint_data, fp)
                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                ray_train.report(
                    {"loss": loss.detach().numpy(), "vloss": vloss},
                    checkpoint=checkpoint,
                )
        elif checkpoint_path is not None and best_vloss:
            if verbose:
                print(f"Saving weights to: {checkpoint_path}", flush=True)
            torch.save(model.state_dict(), checkpoint_path)
            # Save optimizer weights
            best_optimizer_state = optimizer.state_dict()

        if not hyper and early_stopper.stop:
            if verbose:
                print("Early stopping criteria met!", flush=True)
            break

    if not hyper and checkpoint_path is not None:
        if verbose:
            print(
                f"Before any evaluation, reloading best state: {checkpoint_path}",
                flush=True,
            )
        model.load_state_dict(torch.load(checkpoint_path))
        optimizer.load_state_dict(best_optimizer_state)

    return model, optimizer


def main(args):

    if args.verbose:
        print("Arguments:", args, flush=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "; cuda available?", torch.cuda.is_available(), flush=True)

    target = args.target
    excl_meta = args.excl_meta
    outpath = f"{args.save}.{timestamp}.pth" if args.save is not None else None

    model_config = {
        "target": target,
        "excl_meta": args.excl_meta,
        "batch_size": 32,
        "optimizer_cls": optim.Adam,
        "learning_rate": 0.0005,
        "oversample": args.oversample,
        "undersample": args.undersample,
        "loss_weighting": args.loss_weighting,
        # FCN is the primary model
        "fcn_config": FCN1DConfig.WANG2016, # FCN1DConfig.ZHOU2024,
        "fcn_batch_norm": True,
        "fcn_max_pool": True,
        "fcn_conv_dropout": args.conv_dropout if args.conv_dropout is not None else 0.4,
        "fcn_linear_dropout": args.lin_dropout if args.lin_dropout is not None else 0.6,
        "soto2022": args.soto2022,
    }
    if args.criterion is not None:
        model_config["criterion"] = CRITERION[args.criterion]

    if args.hyper:
        # Hyperparameter search space
        tune_config = {
            "target": target,
            "optimizer_cls": tune.choice([optim.Adam, optim.SGD]),
            "learning_rate": tune.choice(1e-4, 1e-3, 1e-2),
            "batch_size": tune.choice([16, 32, 64, 128]),
            "fcn_config": tune.choice([FCN1DConfig.ZHOU2024, FCN1DConfig.WANG2016]),
            "fcn_batch_norm": tune.choice([False, True]),
            "fcn_max_pool": tune.choice([False, True]),
            "fcn_conv_dropout": tune.quniform(0.0, 0.8, 0.1),
            "fcn_linear_dropout": tune.quniform(0.0, 0.8, 0.1),
            "excl_meta": excl_meta,
        }
        if target.is_categorical:
            tune_config["oversample"] = tune.choice([False, True])
            tune_config["loss_weighting"] = tune.choice([False, True])

        scheduler = ASHAScheduler(
            metric="vloss",
            mode="min",
            max_t=125,
            grace_period=1,
            reduction_factor=2,
        )
        result = tune.run(
            partial(train, target, hyper=True),
            resources_per_trial={"cpu": 4, "gpu": 0},
            config=tune_config,
            num_samples=50,
            max_concurrent_trials=4,
            scheduler=scheduler,
        )
        print("Finished search!", flush=True)

        best_trial = result.get_best_trial("vloss", "min", "all")
        print("Best config:", best_trial.config)
        print("Best result:", best_trial.last_result)

        print("Reloading best model...")
        # Configure
        model = get_model(best_trial.config)
        # Load respective weights
        best_checkpoint = result.get_best_checkpoint(
            trial=best_trial, metric="vloss", mode="min"
        )
        with best_checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / CHECKPOINT_FILE
            with open(data_path, "rb") as fp:
                best_checkpoint_data = pickle.load(fp)
            model.load_state_dict(best_checkpoint_data["model_state_dict"])

        if args.save:
            print(f"Writing best model to: {outpath}")
            torch.save(model.state_dict(), outpath)

    elif args.train and args.ext_tune:
        if not args.load:
            sys.exit("Unable to fine-tune model without existing state specified")
        existing_state_path = args.load
        print("Fine tuning:", existing_state_path)

        # Load existing model
        model = get_model(model_config)
        model.load_state_dict(torch.load(existing_state_path))
        model.train()
        # Pre-freeze everything and disable gradients
        for param in model.parameters():
            param.requires_grad = False  
        # Keep batch-norm in inference mode
        def _set_bn_eval(m, requires_grad=False):
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                if requires_grad:
                    m.train()
                else:
                    m.eval()
                for param in m.parameters():
                    param.requires_grad = requires_grad 
        model.apply(_set_bn_eval)
        # For selective enablement
        def _get_params(model, just_bn=False):
            params = []
            for module in model.modules():
                # TODO: Fix, rather messy 
                is_bn = isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d)
                if is_bn and just_bn:
                    params.extend(module.parameters())
                elif is_bn and not just_bn:
                    params.extend(module.parameters())
                elif not just_bn:
                    params.extend(module.parameters())
            return params
        # Incremental transfer learning
        def _train_partial(partial_model, partial_optimizer, model_config, cp, **kwargs):
            return train(
                target,
                model_config,
                checkpoint_path=cp,
                n_epochs=N_EPOCHS_TRANSF,
                verbose=args.verbose,
                existing_model=partial_model,
                existing_optimizer=partial_optimizer,
                existing_state_path=None,
                external_tune=True,
                # Custom early stopping
                early_stop_n = 20,
                reduce_lr_n = 10,
                **kwargs
            )

        if model_config.get('soto2022') is None:
            print('Tuning out...')
            for param in model.out.parameters():
                param.requires_grad = True
            model_config['optimizer_params'] = [
                {'params': model.out.parameters(), 'lr': 0.00005, 'weight_decay': 1e-4}
            ]
            model, opt = _train_partial(model, None, model_config, outpath)
            print('Tuning in3...')
            for param in model.in3.parameters():
                param.requires_grad = True
            opt.add_param_group({'params': model.in3.parameters(), 'lr': 0.0001, 'weight_decay': 1e-5})
            model, opt = _train_partial(model, opt, model_config, outpath)
            print('Tuning in2...')
            for param in model.in2.parameters():
                param.requires_grad = True
            opt.add_param_group({'params': model.in2.parameters(), 'lr': 0.0001, 'weight_decay': 1e-5})
            model, opt = _train_partial(model, opt, model_config, outpath)
            print('Tuning in1...')
            for param in model.in1.parameters():
                param.requires_grad = True
            opt.add_param_group({'params': model.in1.parameters(), 'lr': 0.0001, 'weight_decay': 1e-5})
            model, opt = _train_partial(model, opt, model_config, outpath)
        elif model_config.get('soto2022') == 'resnet34':
            print('Tuning out...')
            for param in model.classifier.parameters():
                param.requires_grad = True
            model_config['optimizer_params'] = [
                {'params': model.classifier.parameters(), 'lr': 0.00005, 'weight_decay': 1e-4}
            ]
            model, opt = _train_partial(model, None, model_config, outpath)
            print('Tuning layer4...')
            for param in model.layer4.parameters():
                param.requires_grad = True
            opt.add_param_group({'params': model.layer4.parameters(), 'lr': 0.0001, 'weight_decay': 1e-5})
            model, opt = _train_partial(model, opt, model_config, outpath)
            print('Tuning layer3...')
            for param in model.layer3.parameters():
                param.requires_grad = True
            opt.add_param_group({'params': model.layer3.parameters(), 'lr': 0.0001, 'weight_decay': 1e-5})
            model, opt = _train_partial(model, opt, model_config, outpath)
            print('Tuning layer2...')
            for param in model.layer2.parameters():
                param.requires_grad = True
            opt.add_param_group({'params': model.layer2.parameters(), 'lr': 0.0001, 'weight_decay': 1e-5})
            model, opt = _train_partial(model, opt, model_config, outpath)
            print('Tuning layer1...')
            for param in model.layer1.parameters():
                param.requires_grad = True
            opt.add_param_group({'params': model.layer1.parameters(), 'lr': 0.0001, 'weight_decay': 1e-5})
            model, opt = _train_partial(model, opt, model_config, outpath)
        else:
            for param in model.parameters():
                param.requires_grad = True
            model_config['optimizer_params'] = [
                {'params': model.parameters(), 'lr': 0.00001, 'weight_decay': 1e-4}
            ]
            model, opt = _train_partial(model, None, model_config, outpath)

    elif args.train:
        print("Training...")
        model, _ = train(
            target,
            model_config,
            checkpoint_path=outpath,
            verbose=args.verbose,
            existing_state_path=None
        )

    elif args.load is not None:
        print(f"Loading weights from: {args.load}", flush=True)
        model = get_model(model_config)
        model.load_state_dict(torch.load(args.load))
        if args.verbose:
            total_params = sum(p.numel() for p in model.parameters())
            train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("Loaded model:", flush=True)
            print(model, flush=True)
            print(f"Params, total={total_params}, trainable={train_params}", flush=True)
    else:
        sys.exit("No model specified to load, or train, so exiting instead.")

    train_loader, val_loader, test_loader, ext_val_loader = load_data(
        BATCH_SIZE, target, SPLITS, args.oversample, args.undersample,
        external_val=args.ext_val, external_tune=args.ext_tune,
    )
    target_loader = test_loader if not args.ext_val else ext_val_loader
    if args.save_splits:
        for split, loader in zip(('train', 'val', 'test'), ((train_loader, val_loader, test_loader))):
            feid_out_fname = f'ukb_feids.{split}_split.list'
            loader.dataset._df_orig['f.eid'].sort_values().to_csv(feid_out_fname, index=False)

    if not target.is_categorical:
        class_prior = None
    else:
        class_prior = train_loader.dataset.get_prior()
    results_df = evaluate(target, model, target_loader, class_prior)

    recalib_model, thresh_model = None, None
    if target.value in (ECGTarget.MASS.value, ECGTarget.INDEXED_MASS.value):
        if args.recalib:
            print("Recalibrating LVM predictions (pipeline)... ")
            #recalib_model = lvm_recalibration(model, val_loader, col=target.value)
            recalib_model = lvm_recalibration(model, train_loader, col=target.value)
            recal_results_df = evaluate(target, model, target_loader,
                                        train_loader.dataset.get_lvh_prior(),
                                        recalib_model=recalib_model)
            if args.save_results:
                sys.exit('Unsupported result export handling when using recalibration model')

        print("Learning an LVH classifier from LVM predictions (pipeline)...")
        thresh_model = lvh_threshold_learning(model, val_loader)
        thresh_results_df = evaluate(target, model, target_loader,
                                     train_loader.dataset.get_lvh_prior(),
                                     thresh_target = ECGTarget.LVH,
                                     thresh_model = thresh_model,
                                     recalib_model=recalib_model)
        results_df['lvh_proba_lr'] = thresh_results_df['lvh_proba']
        results_df['lvh_proba_prioradj_lr'] = thresh_results_df['lvh_proba_prioradj']

    if args.save_results:
        if outpath is not None:
            # Something has been trained (including fine-tuning)
            res_outpath = outpath.replace('.pth', '.results.csv')
        elif args.load:
            # Loaded for results regeneration
            res_outpath = args.load.replace('.pth', '.results.csv')
        print(f"Writing model results to: {res_outpath}")
        results_df.to_csv(res_outpath, index=False)
        
        if thresh_model is not None:
            lr_outpath = res_outpath.replace('.results.csv', '.LR.pkl')
            print(f"Writing thresh. model to: {lr_outpath}")
            with open(lr_outpath, "wb") as fp:
                pickle.dump(thresh_model, fp, protocol=5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose printing",
    )
    parser.add_argument(
        "--hyper",
        action="store_true",
        help="Optimise hyperparameters",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a new model",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save model weights, dated from training start time, with this custom ID",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save predictions, and any pipelined models, to file.",
    )
    parser.add_argument(
        "--ext_val",
        action="store_true",
        help="Evaluate external dataset instead of test set",
    )
    parser.add_argument(
        "--ext_tune",
        action="store_true",
        help="Fine tune and evaluate external dataset",
    )
    parser.add_argument(
        "--load",
        help="Load existing model weights",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot model performance, requires interactive use",
    )

    def _parse_ecg_target(value):
        for target in ECGTarget:
            if target.name == value:
                return target
        choices = [e.name for e in ECGTarget]
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a valid target variable: {choices}"
        )

    parser.add_argument(
        "--target",
        type=_parse_ecg_target,
        required=True,
        help="ECG outcome to predict",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        choices=list(sorted(CRITERION.keys())),
        default=None,
        help="Loss function"
    )
    parser.add_argument(
        "--oversample",
        action="store_true",
        help="Oversample the minority class",
    )
    parser.add_argument(
        "--undersample",
        action="store_true",
        help="Undersample the majority class",
    )
    parser.add_argument(
        "--loss_weighting",
        action="store_true",
        help="Weight criterion (either categorical imbalance or extreme outcomes)",
    )
    parser.add_argument(
        "--soto2022",
        type=str,
        choices=RESNET_LAYOUTS,
        help="ResNet configuration from Soto et al. (2022)",
    )
    parser.add_argument(
        "--conv_dropout",
        type=float,
        help="Dropout after convolutional layers",
        default=None,
    )
    parser.add_argument(
        "--lin_dropout",
        type=float,
        help="Dropout between final linear layers",
        default=None,
    )
    parser.add_argument("--excl_meta", action="store_true", help="Ignore metadata")
    parser.add_argument("--recalib", action="store_true", help="Recalibrate predictions")
    parser.add_argument("--save_splits", action="store_true", help="Save dataset splits")
    args = parser.parse_args()

    if not args.target.is_categorical and (args.oversample or args.undersample):
        print(f"Unable to over/undersample non-categorical target: {args.target.value}")
        sys.exit(1)
    if args.oversample and args.undersample:
        print(
            f"Unable to over- and undersample the categorical target: {args.target.value}"
        )
        sys.exit(1)

    main(args)
