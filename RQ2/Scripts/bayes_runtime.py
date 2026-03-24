from __future__ import annotations

import logging
import os
from typing import Optional

import jax
import numpyro


logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def detect_slurm_environment() -> dict:
    slurm_vars = [
        "SLURM_JOB_ID",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NTASKS",
        "SLURM_JOB_NODELIST",
    ]
    return {k: os.environ.get(k) for k in slurm_vars if k in os.environ}


def log_jax_runtime_info(assert_backend: Optional[str] = None) -> None:
    backend = jax.default_backend()
    devices = jax.devices()

    logger.info("JAX backend: %s", backend)
    logger.info("JAX device count: %d", len(devices))

    for i, d in enumerate(devices):
        logger.info("Device %d: %s", i, d)

    if assert_backend is not None and backend != assert_backend:
        raise RuntimeError(
            f"Expected JAX backend '{assert_backend}', but got '{backend}'."
        )


def log_numpyro_runtime_info() -> None:
    logger.info("NumPyro version: %s", numpyro.__version__)
    # logger.info("NumPyro platform: %s", numpyro.util.platform)


def log_slurm_info() -> None:
    slurm_info = detect_slurm_environment()
    if slurm_info:
        logger.info("Running under SLURM:")
        for k, v in slurm_info.items():
            logger.info("  %s = %s", k, v)
    else:
        logger.info("Not running under SLURM.")
