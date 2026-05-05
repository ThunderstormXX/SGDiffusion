"""
Preset experiment configurations for SGD -> GD -> SGD tunnel experiments.

Available presets:
    MNIST:
        - mnist_small:  5 runs,   SGD(100)  -> GD(10)  -> SGD(1000)
        - mnist_medium: 20 runs,  SGD(1000) -> GD(100) -> SGD(1000)
        - mnist_large:  1000 runs, SGD(1000) -> GD(100) -> SGD(3000)

    Shakespeare:
        - shakespeare_small:  5 runs,   SGD(100)  -> GD(10)  -> SGD(1000)
        - shakespeare_medium: 20 runs,  SGD(1000) -> GD(100) -> SGD(1000)
        - shakespeare_large:  1000 runs, SGD(1000) -> GD(100) -> SGD(3000)

Tracking strategy:
    - Stage 1 (SGD): weights only
    - Stage 2 (GD):  metrics only (+ weights for visualization)
    - Stage 3 (SGD): weights + hessians
"""

from src.datamodelopt.core.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    StageConfig,
    TrackerConfig,
    TunnelConfig,  # NEW: for tunnel-based experiments
)

# =============================================================================
# AVAILABLE PRESETS
# =============================================================================

AVAILABLE_PRESETS = [
    "mnist_small",
    "mnist_medium",
    "mnist_large",
    "shakespeare_small",
    "shakespeare_medium",
    "shakespeare_large",
    # Many SGD from single point
    "mnist_manysgd_small",
    "mnist_manysgd_medium",
    "mnist_manysgd_large",
    "shakespeare_manysgd_small",
    "shakespeare_manysgd_medium",
    "shakespeare_manysgd_large",
]


def get_preset(name: str) -> ExperimentConfig:
    """
    Get a preset configuration by name.

    Args:
        name: Preset name.

    Returns:
        An ExperimentConfig instance.
    """
    presets = {
        "mnist_small": get_mnist_small,
        "mnist_medium": get_mnist_medium,
        "mnist_large": get_mnist_large,
        "shakespeare_small": get_shakespeare_small,
        "shakespeare_medium": get_shakespeare_medium,
        "shakespeare_large": get_shakespeare_large,
        # Many SGD from single point
        "mnist_manysgd_small": get_mnist_manysgd_small,
        "mnist_manysgd_medium": get_mnist_manysgd_medium,
        "mnist_manysgd_large": get_mnist_manysgd_large,
        "shakespeare_manysgd_small": get_shakespeare_manysgd_small,
        "shakespeare_manysgd_medium": get_shakespeare_manysgd_medium,
        "shakespeare_manysgd_large": get_shakespeare_manysgd_large,
    }

    if name not in presets:
        raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")

    return presets[name]()


# =============================================================================
# MNIST PRESETS
# =============================================================================

def get_mnist_small() -> ExperimentConfig:
    """
    MNIST Small: Quick testing experiment.

    - N_RUNS: 5 (handled by bash script)
    - Stage 1 SGD: 100 steps, track weights
    - Stage 2 GD: 10 epochs, track metrics
    - Stage 3 SGD: 1000 steps, track weights + hessians
    """
    return ExperimentConfig(
        run_dir="src/scripts/exp5/exp_results/mnist_small",
        seed=42,
        device="cpu",
        dtype="float32",
        data=DataConfig(
            name="mnist",
            kwargs={
                "batch_size": 64,
                "replacement": True,
                "sample_size": 6400,  # Standard MNIST subset
            },
        ),
        model=ModelConfig(
            name="class",
            kwargs={
                "class_name": "FlexibleMLP",
                "hidden_dim": 8,
                "num_hidden_layers": 1,
                "input_downsample": 6,  # 28->6, so 6x6=36 input dims -> 386 params
            },
        ),
        stages=[
            # Stage 1: SGD - track WEIGHTS only
            StageConfig(
                name="stage1_sgd",
                mode="steps",
                steps=100,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                ],
                eval_every=20,
                save_checkpoint="stage1_sgd.pt",
            ),
            # Stage 2: GD - track METRICS only (+ weights for viz)
            StageConfig(
                name="stage2_gd",
                mode="epochs",
                epochs=10,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.01}),
                dataloader_mode="fullbatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                ],
                eval_every=5,
                save_checkpoint="stage2_gd.pt",
            ),
            # Stage 3: SGD - track WEIGHTS + HESSIANS
            StageConfig(
                name="stage3_sgd",
                mode="steps",
                steps=1000,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                    TrackerConfig(name="hessian", kwargs={"every": 1}),
                ],
                eval_every=100,
                save_checkpoint="stage3_sgd.pt",
            ),
        ],
    )


def get_mnist_medium() -> ExperimentConfig:
    """
    MNIST Medium: Standard experiment.

    - N_RUNS: 20 (handled by bash script)
    - Stage 1 SGD: 1000 steps, track weights
    - Stage 2 GD: 100 epochs, track metrics
    - Stage 3 SGD: 1000 steps, track weights + hessians
    """
    return ExperimentConfig(
        run_dir="src/scripts/exp5/exp_results/mnist_medium",
        seed=42,
        device="cpu",
        dtype="float32",
        data=DataConfig(
            name="mnist",
            kwargs={
                "batch_size": 64,
                "replacement": True,
                "sample_size": 6400,  # Standard MNIST subset
            },
        ),
        model=ModelConfig(
            name="class",
            kwargs={
                "class_name": "FlexibleMLP",
                "hidden_dim": 8,
                "num_hidden_layers": 1,
                "input_downsample": 6,  # 28->6, so 6x6=36 input dims -> 386 params
            },
        ),
        stages=[
            # Stage 1: SGD - track WEIGHTS every step
            StageConfig(
                name="stage1_sgd",
                mode="steps",
                steps=1000,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                ],
                eval_every=100,
                save_checkpoint="stage1_sgd.pt",
            ),
            # Stage 2: GD - track METRICS and WEIGHTS every step
            StageConfig(
                name="stage2_gd",
                mode="epochs",
                epochs=100,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.01}),
                dataloader_mode="fullbatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                ],
                eval_every=20,
                save_checkpoint="stage2_gd.pt",
            ),
            # Stage 3: SGD - track WEIGHTS and HESSIANS every step
            StageConfig(
                name="stage3_sgd",
                mode="steps",
                steps=1000,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                    TrackerConfig(name="hessian", kwargs={"every": 1}),
                ],
                eval_every=100,
                save_checkpoint="stage3_sgd.pt",
            ),
        ],
    )


def get_mnist_large() -> ExperimentConfig:
    """
    MNIST Large: Comprehensive experiment for thorough analysis.

    - N_RUNS: 1000 (handled by bash script)
    - Stage 1 SGD: 1000 steps, track weights
    - Stage 2 GD: 100 epochs, track metrics
    - Stage 3 SGD: 3000 steps, track weights + hessians
    """
    return ExperimentConfig(
        run_dir="src/scripts/exp5/exp_results/mnist_large",
        seed=42,
        device="cpu",
        dtype="float32",
        data=DataConfig(
            name="mnist",
            kwargs={
                "batch_size": 64,
                "replacement": True,
                "sample_size": 6400,  # Standard MNIST subset
            },
        ),
        model=ModelConfig(
            name="class",
            kwargs={
                "class_name": "FlexibleMLP",
                "hidden_dim": 8,
                "num_hidden_layers": 1,
                "input_downsample": 6,  # 28->6, so 6x6=36 input dims -> 386 params
            },
        ),
        stages=[
            # Stage 1: SGD - track WEIGHTS only
            StageConfig(
                name="stage1_sgd",
                mode="steps",
                steps=1000,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                ],
                eval_every=100,
                save_checkpoint="stage1_sgd.pt",
            ),
            # Stage 2: GD - track METRICS only (+ weights for viz)
            StageConfig(
                name="stage2_gd",
                mode="epochs",
                epochs=100,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.01}),
                dataloader_mode="fullbatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                ],
                eval_every=20,
                save_checkpoint="stage2_gd.pt",
            ),
            # Stage 3: SGD - track WEIGHTS + HESSIANS
            StageConfig(
                name="stage3_sgd",
                mode="steps",
                steps=3000,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                    TrackerConfig(name="hessian", kwargs={"every": 1}),
                ],
                eval_every=300,
                save_checkpoint="stage3_sgd.pt",
            ),
        ],
    )


# =============================================================================
# SHAKESPEARE PRESETS
# =============================================================================

def get_shakespeare_small() -> ExperimentConfig:
    """
    Shakespeare Small: Quick testing experiment with NanoGPT.

    - N_RUNS: 5 (handled by bash script)
    - Stage 1 SGD: 100 steps, track weights
    - Stage 2 GD: 10 epochs, track metrics
    - Stage 3 SGD: 1000 steps, track weights + hessians
    """
    return ExperimentConfig(
        run_dir="src/scripts/exp5/exp_results/shakespeare_small",
        seed=42,
        device="cpu",
        dtype="float32",
        data=DataConfig(
            name="shakespeare",
            kwargs={
                "batch_size": 64,
                "replacement": True,
                "train_path": "src/data/shakespeare_train.pt",
                "val_path": "src/data/shakespeare_val.pt",
                "meta_path": "src/data/shakespeare_meta.pt",
            },
        ),
        model=ModelConfig(
            name="nanogpt",
            kwargs={
                "n_embd": 8,
                "n_head": 1,
                "n_layer": 1,
                "mlp_ratio": 1,
                "meta_path": "src/data/shakespeare_meta.pt",
            },
        ),
        stages=[
            # Stage 1: SGD - track WEIGHTS only
            StageConfig(
                name="stage1_sgd",
                mode="steps",
                steps=100,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.01}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                ],
                eval_every=20,
                save_checkpoint="stage1_sgd.pt",
            ),
            # Stage 2: GD - track METRICS only (+ weights for viz)
            StageConfig(
                name="stage2_gd",
                mode="epochs",
                epochs=10,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.001}),
                dataloader_mode="fullbatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                ],
                eval_every=5,
                save_checkpoint="stage2_gd.pt",
            ),
            # Stage 3: SGD - track WEIGHTS + HESSIANS
            StageConfig(
                name="stage3_sgd",
                mode="steps",
                steps=1000,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.01}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                    TrackerConfig(name="hessian", kwargs={"every": 1}),
                ],
                eval_every=100,
                save_checkpoint="stage3_sgd.pt",
            ),
        ],
    )


def get_shakespeare_medium() -> ExperimentConfig:
    """
    Shakespeare Medium: Standard experiment with NanoGPT.

    - N_RUNS: 20 (handled by bash script)
    - Stage 1 SGD: 1000 steps, track weights
    - Stage 2 GD: 100 epochs, track metrics
    - Stage 3 SGD: 1000 steps, track weights + hessians
    """
    return ExperimentConfig(
        run_dir="src/scripts/exp5/exp_results/shakespeare_medium",
        seed=42,
        device="cpu",
        dtype="float32",
        data=DataConfig(
            name="shakespeare",
            kwargs={
                "batch_size": 64,
                "replacement": True,
                "train_path": "src/data/shakespeare_train.pt",
                "val_path": "src/data/shakespeare_val.pt",
                "meta_path": "src/data/shakespeare_meta.pt",
            },
        ),
        model=ModelConfig(
            name="nanogpt",
            kwargs={
                "n_embd": 8,
                "n_head": 1,
                "n_layer": 1,
                "mlp_ratio": 1,
                "meta_path": "src/data/shakespeare_meta.pt",
            },
        ),
        stages=[
            # Stage 1: SGD - track WEIGHTS only
            StageConfig(
                name="stage1_sgd",
                mode="steps",
                steps=1000,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.01}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                ],
                eval_every=100,
                save_checkpoint="stage1_sgd.pt",
            ),
            # Stage 2: GD - track METRICS only (+ weights for viz)
            StageConfig(
                name="stage2_gd",
                mode="epochs",
                epochs=100,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.001}),
                dataloader_mode="fullbatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                ],
                eval_every=20,
                save_checkpoint="stage2_gd.pt",
            ),
            # Stage 3: SGD - track WEIGHTS + HESSIANS
            StageConfig(
                name="stage3_sgd",
                mode="steps",
                steps=1000,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.01}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                    TrackerConfig(name="hessian", kwargs={"every": 1}),
                ],
                eval_every=100,
                save_checkpoint="stage3_sgd.pt",
            ),
        ],
    )


def get_shakespeare_large() -> ExperimentConfig:
    """
    Shakespeare Large: Comprehensive experiment with NanoGPT.

    - N_RUNS: 1000 (handled by bash script)
    - Stage 1 SGD: 1000 steps, track weights
    - Stage 2 GD: 100 epochs, track metrics
    - Stage 3 SGD: 3000 steps, track weights + hessians
    """
    return ExperimentConfig(
        run_dir="src/scripts/exp5/exp_results/shakespeare_large",
        seed=42,
        device="cpu",
        dtype="float32",
        data=DataConfig(
            name="shakespeare",
            kwargs={
                "batch_size": 64,
                "replacement": True,
                "train_path": "src/data/shakespeare_train.pt",
                "val_path": "src/data/shakespeare_val.pt",
                "meta_path": "src/data/shakespeare_meta.pt",
            },
        ),
        model=ModelConfig(
            name="nanogpt",
            kwargs={
                "n_embd": 8,
                "n_head": 1,
                "n_layer": 1,
                "mlp_ratio": 1,
                "meta_path": "src/data/shakespeare_meta.pt",
            },
        ),
        stages=[
            # Stage 1: SGD - track WEIGHTS only
            StageConfig(
                name="stage1_sgd",
                mode="steps",
                steps=1000,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.01}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                ],
                eval_every=100,
                save_checkpoint="stage1_sgd.pt",
            ),
            # Stage 2: GD - track METRICS only (+ weights for viz)
            StageConfig(
                name="stage2_gd",
                mode="epochs",
                epochs=100,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.001}),
                dataloader_mode="fullbatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                ],
                eval_every=20,
                save_checkpoint="stage2_gd.pt",
            ),
            # Stage 3: SGD - track WEIGHTS + HESSIANS
            StageConfig(
                name="stage3_sgd",
                mode="steps",
                steps=3000,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.01}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                    TrackerConfig(name="hessian", kwargs={"every": 1}),
                ],
                eval_every=300,
                save_checkpoint="stage3_sgd.pt",
            ),
        ],
    )


# =============================================================================
# MNIST "MANY SGD FROM SINGLE POINT" PRESETS
# =============================================================================
# Strategy: 1 run SGD -> 1 run GD -> MANY runs SGD from that single point
# This explores distribution within final SGD stage, not across full pipeline

def get_mnist_manysgd_small() -> ExperimentConfig:
    """
    MNIST Many-SGD Small: 5 runs of final SGD from single point.

    - Stage 1: 1 run SGD (1000 steps)
    - Stage 2: 1 run GD (100 epochs)
    - Stage 3: 5 runs SGD (100 steps each), tracks weights + Hessians
    """
    return ExperimentConfig(
        run_dir="src/scripts/exp5/exp_results/mnist_manysgd_small",
        seed=42,
        device="cpu",
        dtype="float32",
        data=DataConfig(
            name="mnist",
            kwargs={
                "batch_size": 64,
                "replacement": True,
                "sample_size": 6400,
            },
        ),
        model=ModelConfig(
            name="class",
            kwargs={
                "class_name": "FlexibleMLP",
                "hidden_dim": 8,
                "num_hidden_layers": 1,
                "input_downsample": 6,
            },
        ),
        stages=[
            # Stage 1: SGD (single run, track weights for viz)
            StageConfig(
                name="stage1_sgd",
                mode="steps",
                steps=1000,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                ],
                eval_every=100,
                save_checkpoint="stage1_sgd.pt",
            ),
            # Stage 2: GD (single run, track weights for viz)
            StageConfig(
                name="stage2_gd",
                mode="epochs",
                epochs=100,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.01}),
                dataloader_mode="fullbatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                ],
                eval_every=10,
                save_checkpoint="stage2_gd.pt",
            ),
            # Stage 3: SGD (MANY runs from single point)
            StageConfig(
                name="stage3_sgd",
                mode="steps",
                steps=100,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                    TrackerConfig(name="hessian", kwargs={"every": 1}),
                ],
                eval_every=20,
                save_checkpoint="stage3_sgd.pt",
            ),
        ],
    )


def get_mnist_manysgd_medium() -> ExperimentConfig:
    """MNIST Many-SGD Medium: 50 runs of final SGD from single point."""
    return ExperimentConfig(
        run_dir="src/scripts/exp5/exp_results/mnist_manysgd_medium",
        seed=42,
        device="cpu",
        dtype="float32",
        data=DataConfig(
            name="mnist",
            kwargs={
                "batch_size": 64,
                "replacement": True,
                "sample_size": 6400,
            },
        ),
        model=ModelConfig(
            name="class",
            kwargs={
                "class_name": "FlexibleMLP",
                "hidden_dim": 8,
                "num_hidden_layers": 1,
                "input_downsample": 6,
            },
        ),
        stages=[
            StageConfig(
                name="stage1_sgd",
                mode="steps",
                steps=1000,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                ],
                eval_every=100,
                save_checkpoint="stage1_sgd.pt",
            ),
            StageConfig(
                name="stage2_gd",
                mode="epochs",
                epochs=100,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.01}),
                dataloader_mode="fullbatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                ],
                eval_every=10,
                save_checkpoint="stage2_gd.pt",
            ),
            StageConfig(
                name="stage3_sgd",
                mode="steps",
                steps=1000,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                    TrackerConfig(name="hessian", kwargs={"every": 1}),
                ],
                eval_every=100,
                save_checkpoint="stage3_sgd.pt",
            ),
        ],
    )


def get_mnist_manysgd_large() -> ExperimentConfig:
    """MNIST Many-SGD Large: 1000 runs of final SGD from single point."""
    return ExperimentConfig(
        run_dir="src/scripts/exp5/exp_results/mnist_manysgd_large",
        seed=42,
        device="cpu",
        dtype="float32",
        data=DataConfig(
            name="mnist",
            kwargs={
                "batch_size": 64,
                "replacement": True,
                "sample_size": 6400,
            },
        ),
        model=ModelConfig(
            name="class",
            kwargs={
                "class_name": "FlexibleMLP",
                "hidden_dim": 8,
                "num_hidden_layers": 1,
                "input_downsample": 6,
            },
        ),
        stages=[
            StageConfig(
                name="stage1_sgd",
                mode="steps",
                steps=1000,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                ],
                eval_every=100,
                save_checkpoint="stage1_sgd.pt",
            ),
            StageConfig(
                name="stage2_gd",
                mode="epochs",
                epochs=100,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.01}),
                dataloader_mode="fullbatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                ],
                eval_every=10,
                save_checkpoint="stage2_gd.pt",
            ),
            StageConfig(
                name="stage3_sgd",
                mode="steps",
                steps=3000,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                    TrackerConfig(name="hessian", kwargs={"every": 1}),
                ],
                eval_every=300,
                save_checkpoint="stage3_sgd.pt",
            ),
        ],
    )


# =============================================================================
# SHAKESPEARE "MANY SGD FROM SINGLE POINT" PRESETS
# =============================================================================

def get_shakespeare_manysgd_small() -> ExperimentConfig:
    """Shakespeare Many-SGD Small: 5 runs of final SGD from single point."""
    return ExperimentConfig(
        run_dir="src/scripts/exp5/exp_results/shakespeare_manysgd_small",
        seed=42,
        device="cpu",
        dtype="float32",
        data=DataConfig(
            name="shakespeare",
            kwargs={
                "batch_size": 64,
                "replacement": True,
                "train_path": "src/data/shakespeare_train.pt",
                "val_path": "src/data/shakespeare_val.pt",
                "meta_path": "src/data/shakespeare_meta.pt",
            },
        ),
        model=ModelConfig(
            name="nanogpt",
            kwargs={
                "n_embd": 8,
                "n_head": 1,
                "n_layer": 1,
                "mlp_ratio": 1,
                "meta_path": "src/data/shakespeare_meta.pt",
            },
        ),
        stages=[
            StageConfig(
                name="stage1_sgd",
                mode="steps",
                steps=1000,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.01}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                ],
                eval_every=100,
                save_checkpoint="stage1_sgd.pt",
            ),
            StageConfig(
                name="stage2_gd",
                mode="epochs",
                epochs=100,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.001}),
                dataloader_mode="fullbatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                ],
                eval_every=10,
                save_checkpoint="stage2_gd.pt",
            ),
            StageConfig(
                name="stage3_sgd",
                mode="steps",
                steps=100,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.01}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                    TrackerConfig(name="hessian", kwargs={"every": 1}),
                ],
                eval_every=20,
                save_checkpoint="stage3_sgd.pt",
            ),
        ],
    )


def get_shakespeare_manysgd_medium() -> ExperimentConfig:
    """Shakespeare Many-SGD Medium: 50 runs of final SGD from single point."""
    return ExperimentConfig(
        run_dir="src/scripts/exp5/exp_results/shakespeare_manysgd_medium",
        seed=42,
        device="cpu",
        dtype="float32",
        data=DataConfig(
            name="shakespeare",
            kwargs={
                "batch_size": 64,
                "replacement": True,
                "train_path": "src/data/shakespeare_train.pt",
                "val_path": "src/data/shakespeare_val.pt",
                "meta_path": "src/data/shakespeare_meta.pt",
            },
        ),
        model=ModelConfig(
            name="nanogpt",
            kwargs={
                "n_embd": 8,
                "n_head": 1,
                "n_layer": 1,
                "mlp_ratio": 1,
                "meta_path": "src/data/shakespeare_meta.pt",
            },
        ),
        stages=[
            StageConfig(
                name="stage1_sgd",
                mode="steps",
                steps=1000,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.01}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                ],
                eval_every=100,
                save_checkpoint="stage1_sgd.pt",
            ),
            StageConfig(
                name="stage2_gd",
                mode="epochs",
                epochs=100,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.001}),
                dataloader_mode="fullbatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                ],
                eval_every=10,
                save_checkpoint="stage2_gd.pt",
            ),
            StageConfig(
                name="stage3_sgd",
                mode="steps",
                steps=1000,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.01}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                    TrackerConfig(name="hessian", kwargs={"every": 1}),
                ],
                eval_every=100,
                save_checkpoint="stage3_sgd.pt",
            ),
        ],
    )


def get_shakespeare_manysgd_large() -> ExperimentConfig:
    """Shakespeare Many-SGD Large: 1000 runs of final SGD from single point."""
    return ExperimentConfig(
        run_dir="src/scripts/exp5/exp_results/shakespeare_manysgd_large",
        seed=42,
        device="cpu",
        dtype="float32",
        data=DataConfig(
            name="shakespeare",
            kwargs={
                "batch_size": 64,
                "replacement": True,
                "train_path": "src/data/shakespeare_train.pt",
                "val_path": "src/data/shakespeare_val.pt",
                "meta_path": "src/data/shakespeare_meta.pt",
            },
        ),
        model=ModelConfig(
            name="nanogpt",
            kwargs={
                "n_embd": 8,
                "n_head": 1,
                "n_layer": 1,
                "mlp_ratio": 1,
                "meta_path": "src/data/shakespeare_meta.pt",
            },
        ),
        stages=[
            StageConfig(
                name="stage1_sgd",
                mode="steps",
                steps=1000,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.01}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                ],
                eval_every=100,
                save_checkpoint="stage1_sgd.pt",
            ),
            StageConfig(
                name="stage2_gd",
                mode="epochs",
                epochs=100,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.001}),
                dataloader_mode="fullbatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                ],
                eval_every=10,
                save_checkpoint="stage2_gd.pt",
            ),
            StageConfig(
                name="stage3_sgd",
                mode="steps",
                steps=3000,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.01}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                    TrackerConfig(name="hessian", kwargs={"every": 1}),
                ],
                eval_every=300,
                save_checkpoint="stage3_sgd.pt",
            ),
        ],
    )


# ============================================================================
# NEW: Tunnel-based Presets (Clean Architecture)
# ============================================================================

def get_mnist_manysgd_small_v2() -> ExperimentConfig:
    """
    MNIST Many-SGD Small (Tunnel-based architecture v2).

    Strategy: 1 SGD -> 1 GD -> 5 SGD from single point

    Tunnel 0: 1 run,  1000 steps SGD, lr=0.1
    Tunnel 1: 1 run,  100 epochs GD, lr=0.01 (continues from tunnel_0/run_0)
    Tunnel 2: 5 runs, 100 steps SGD, lr=0.1 (all from tunnel_1/run_0 - cartesian!)

    All learning rates and optimizer configs stored in JSON metadata.
    """
    return ExperimentConfig(
        run_dir="src/scripts/exp5/exp_results/mnist_manysgd_small_v2",
        seed=42,
        device="cpu",
        dtype="float32",
        data=DataConfig(
            name="mnist",
            kwargs={
                "batch_size": 64,
                "replacement": True,
                "sample_size": 6400,
            },
        ),
        model=ModelConfig(
            name="class",
            kwargs={
                "class_name": "FlexibleMLP",
                "hidden_dim": 8,
                "num_hidden_layers": 1,
                "input_downsample": 6,
            },
        ),
        tunnels=[
            TunnelConfig(
                tunnel_index=0,
                description="Initial SGD training, 1000 steps",
                mode="steps",
                steps=1000,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                ],
                eval_every=100,
                save_checkpoint=True,
                source_mode=None,
            ),
            TunnelConfig(
                tunnel_index=1,
                description="Full-batch GD refinement, 100 epochs",
                mode="epochs",
                epochs=100,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.01}),
                dataloader_mode="fullbatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                ],
                eval_every=10,
                save_checkpoint=True,
                source_mode="1to1",
            ),
            TunnelConfig(
                tunnel_index=2,
                description="Multiple SGD runs from single point, 100 steps",
                mode="steps",
                steps=100,
                optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
                dataloader_mode="minibatch",
                trackers=[
                    TrackerConfig(name="metrics", kwargs={"every": 1}),
                    TrackerConfig(name="weights", kwargs={"every": 1}),
                    TrackerConfig(name="hessian", kwargs={"every": 1}),
                ],
                eval_every=20,
                save_checkpoint=True,
                source_mode="cartesian",
                source_run_index=0,
            ),
        ],
    )
