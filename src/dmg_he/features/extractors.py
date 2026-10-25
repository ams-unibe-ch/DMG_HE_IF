"""Tile feature extractors.

Every extractor maps a list of RGB PIL images to a 2-D array
(n_images x n_features). Heavy dependencies (torch, timm, tensorflow and the
model-specific repositories) are imported only when an extractor is built.

| name          | model                                   | features |
|---------------|-----------------------------------------|----------|
| retccl        | RetCCL ResNet-50 (Wang et al. 2023)     | 2048     |
| ctranspath    | CTransPath Swin-T (Wang et al. 2022)    | 768      |
| remedis       | REMEDIS ResNet (Azizi et al. 2023)      | 4096     |
| uni           | UNI ViT-L/16 (Chen et al. 2024)         | 1024     |
| virchow       | Virchow ViT-H/14 (Vorontsov et al. 2024)| 2560     |
| provgigapath  | Prov-GigaPath tile encoder (Xu 2024)    | 1536     |
| rgb_stats     | colour statistics (baseline / testing)  | 15       |
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from ..utils import log

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

REGISTRY: dict[str, Callable] = {}


def register(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


class Extractor:
    """Wraps a model and its preprocessing. Call with a list of PIL images."""

    def __init__(self, name: str, fn: Callable[[list], np.ndarray], batch_size: int = 64):
        self.name, self._fn, self.batch_size = name, fn, batch_size

    def __call__(self, images: list) -> np.ndarray:
        out = [self._fn(images[i:i + self.batch_size]) for i in range(0, len(images), self.batch_size)]
        return np.concatenate(out, axis=0) if out else np.empty((0, 0))


def build(name: str, settings: dict, resolve: Callable[[str], Path]) -> Extractor:
    if name not in REGISTRY:
        raise KeyError(f"unknown feature extractor {name!r}; choose from {sorted(REGISTRY)}")
    model_cfg = settings.get(name, {}) or {}
    log.info("loading feature extractor %s", name)
    fn = REGISTRY[name](model_cfg, settings, resolve)
    return Extractor(name, fn, batch_size=int(settings.get("batch_size", 64)))


# --------------------------------------------------------------------- torch
def _device(settings):
    import torch

    dev = settings.get("device", "auto")
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("using device %s", dev)
    return torch.device(dev)


def _torch_runner(model, transform, device, postprocess=None):
    import torch

    model.eval().to(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    def run(images):
        x = torch.stack([transform(im) for im in images]).to(device)
        with torch.inference_mode():
            y = model(x)
            if postprocess is not None:
                y = postprocess(y)
        return y.float().cpu().numpy()

    return run


def _imagenet_transform(resize, crop=None, bicubic=False):
    from torchvision import transforms

    interp = transforms.InterpolationMode.BICUBIC if bicubic else transforms.InterpolationMode.BILINEAR
    steps = [transforms.Resize(resize, interpolation=interp)]
    if crop:
        steps.append(transforms.CenterCrop(crop))
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


def _weights(model_cfg, resolve, name):
    p = model_cfg.get("weights")
    if not p:
        raise ValueError(f"feature_extractors.{name}.weights is not set in the config")
    p = resolve(p)
    if not p.exists():
        raise FileNotFoundError(f"{name} weights not found at {p}")
    return p


@register("retccl")
def _retccl(model_cfg, settings, resolve):
    import torch
    import torch.nn as nn

    try:
        from RetCCL import ResNet  # https://github.com/Xiyue-Wang/RetCCL on PYTHONPATH
    except ImportError as e:
        raise ImportError("RetCCL code not found: clone https://github.com/Xiyue-Wang/RetCCL "
                          "and add its parent folder to PYTHONPATH") from e
    model = ResNet.resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
    model.fc = nn.Identity()
    state = torch.load(_weights(model_cfg, resolve, "retccl"), map_location="cpu")
    model.load_state_dict(state, strict=True)
    return _torch_runner(model, _imagenet_transform(256), _device(settings))


@register("ctranspath")
def _ctranspath(model_cfg, settings, resolve):
    import torch
    import torch.nn as nn

    try:
        from ctran import ctranspath  # https://github.com/Xiyue-Wang/TransPath
    except ImportError:
        try:
            from ctranspath.ctran import ctranspath
        except ImportError as e:
            raise ImportError("CTransPath code not found: clone https://github.com/Xiyue-Wang/TransPath, "
                              "install its modified timm and add the folder to PYTHONPATH") from e
    model = ctranspath()
    model.head = nn.Identity()
    state = torch.load(_weights(model_cfg, resolve, "ctranspath"), map_location="cpu")
    state = state.get("model", state)
    res = model.load_state_dict(state, strict=False)
    missing = [k for k in res.missing_keys if not k.startswith("head")]
    if missing:
        raise RuntimeError(f"CTransPath weights do not match the model ({len(missing)} missing keys, "
                           f"e.g. {missing[:3]}) - check the timm version used with TransPath")
    return _torch_runner(model, _imagenet_transform(224), _device(settings))


def _timm_hub(name, **kwargs):
    import timm

    return timm.create_model(name, pretrained=True, **kwargs)


def _timm_transform(model):
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    return create_transform(**resolve_data_config(model.pretrained_cfg, model=model))


@register("uni")
def _uni(model_cfg, settings, resolve):
    model = _timm_hub("hf-hub:MahmoodLab/uni", init_values=1e-5, dynamic_img_size=True)
    return _torch_runner(model, _timm_transform(model), _device(settings))


@register("virchow")
def _virchow(model_cfg, settings, resolve):
    import torch
    from timm.layers import SwiGLUPacked

    model = _timm_hub("hf-hub:paige-ai/Virchow", mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)

    def embed(tokens):  # class token + mean of patch tokens -> 2 x 1280
        return torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=-1)

    return _torch_runner(model, _timm_transform(model), _device(settings), postprocess=embed)


@register("provgigapath")
def _provgigapath(model_cfg, settings, resolve):
    model = _timm_hub("hf_hub:prov-gigapath/prov-gigapath")
    return _torch_runner(model, _imagenet_transform(224, crop=224, bicubic=True), _device(settings))


# ---------------------------------------------------------------- tensorflow
@register("remedis")
def _remedis(model_cfg, settings, resolve):
    import tensorflow as tf
    import tensorflow_hub as hub

    model = hub.load(str(_weights(model_cfg, resolve, "remedis")))

    def run(images):
        x = np.stack([np.asarray(im.resize((224, 224)), dtype=np.float32) / 255.0 for im in images])
        y = model(tf.convert_to_tensor(x)).numpy()
        if y.ndim == 4:  # feature map -> global average pooling
            y = y.mean(axis=(1, 2))
        return y

    return run


# ------------------------------------------------------------------ baseline
@register("rgb_stats")
def _rgb_stats(model_cfg, settings, resolve):
    """Colour statistics per RGB channel; a fast, dependency-free baseline used in the tests."""

    def run(images):
        out = []
        for im in images:
            a = np.asarray(im, dtype=np.float32).reshape(-1, 3) / 255.0
            out.append(np.concatenate([a.mean(0), a.std(0), *np.percentile(a, [10, 50, 90], axis=0)]))
        return np.stack(out)

    return run
