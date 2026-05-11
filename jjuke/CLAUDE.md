# CLAUDE.md

Guidance for Claude Code working inside `jjuke/`. This is a thin, opinionated
training scaffold around HuggingFace `Accelerate` + `OmegaConf` + `EasyDict`.
The framework provides config-driven object instantiation, distributed
orchestration, mixed precision, EMA, checkpointing, logging, and a Rich
progress bar; it leaves model / loss / dataset bodies to subclasses.

**This is a skeleton.** Wiring is fully populated, but `train_epoch`,
`train_step`, `valid_step`, `BasePreprocessor.__call__`, `build_network`,
`prepare_objects`, `save_pipeline`, `final_process`, and the example
`Dataset` body are all empty stubs. See §"Skeleton vs populated" below.

## Migration note (post-restructure)

`jjuke/` has been reorganized into `core / dl / rl` subpackages. The
shared base classes (`BaseTrainer`, `BasePreprocessor`, `RunningMeanStd`,
`BaseObsNormalizer`) live under `jjuke.core`; the DL-specific
`dataset` example moved to `jjuke.dl.dataset`; the RL skeletons stay
at `jjuke.rl`. Old → new import map:

| Old | New |
|---|---|
| `from jjuke.model import trainer` | `from jjuke.core import trainer` |
| `from jjuke.model.trainer import BaseTrainer` | `from jjuke.core.trainer import BaseTrainer` |
| `from jjuke.model.trainer import BasePreprocessor` | `from jjuke.core.trainer import BasePreprocessor` |
| `from jjuke.model.normalizers import RunningMeanStd, BaseObsNormalizer` | `from jjuke.core.normalizers import RunningMeanStd, BaseObsNormalizer` |
| `from jjuke import trainer` (top-level alias) | unchanged — `jjuke.__init__` re-exports `trainer` from the new `core` location |
| `from jjuke.dataset import ...` | `from jjuke.dl.dataset import ...` |
| `from jjuke.rl import BaseRLTrainer, ...` | unchanged |
| `python -m jjuke.main` (or `jjuke/main.py`) | `python -m jjuke.dl.main` (DL launcher: `accelerate launch` subprocess pattern) |
| `python -m jjuke.train` (or `jjuke/train.py`) | `python -m jjuke.dl.train` (DL worker) |
| (no equivalent — env-driven RL had no library launcher) | `python -m jjuke.rl.main` (in-process launcher; env bootstrap goes inside `jjuke.rl.train.train`) |
| `jjuke/config/model_to_experiment.yaml` (DL example) | `jjuke/dl/config/exp_dl.yaml` (DL example, unchanged content) + `jjuke/rl/config/exp_rl.yaml` (NEW — RL example) |

Consumers (MambaDance, the InterMimic clean project) need to update
their import sites accordingly. Env-driven RL consumers (Isaac Sim,
Mujoco, etc.) can model their launcher on `jjuke/rl/{main,train}.py`
or write their own that wraps `jjuke.options.get_config(...)` +
`options.instantiate_from_config(args.trainer, args).fit()`.

## Layout

```
jjuke/
  __init__.py            re-exports core, dl, rl, util, logger, options, trainer (alias to core.trainer), progress_bar, vis
  core/                  cross-cutting base layer (shared by DL + RL)
    __init__.py          re-exports trainer, normalizers
    trainer.py           BaseTrainer (ABCMeta) + BasePreprocessor (ABCMeta)
    normalizers.py       RunningMeanStd (concrete) + BaseObsNormalizer (skeleton)
  dl/                    DL-specific subpackage (DataLoader-driven training)
    __init__.py          re-exports dataset
    main.py              DL launcher → spawns `accelerate launch jjuke/dl/train.py`
    train.py             DL worker → load yaml, build trainer, call fit()
    config/
      exp_dl.yaml        DL experiment schema template (every cfg block)
    dataset/
      __init__.py        empty
      dataloader.py      ExampleDataset (stub) + load_dataloaders factory (stub)
  rl/                    RL-specific skeletons (env-driven training)
    __init__.py          re-exports BaseRLTrainer, BaseRolloutBuffer, BasePolicy
    main.py              RL launcher (in-process; env bootstrap happens in train)
    train.py             RL worker (env-bootstrap-aware; comments for Sim/gym/etc.)
    config/
      exp_rl.yaml        RL experiment schema template (env factory + trainer + optim)
    base_trainer.py      BaseRLTrainer(BaseTrainer) — env-loop scaffold
    buffers.py           BaseRolloutBuffer + GAE + episode-length helper
    policies.py          BasePolicy + Gaussian static helpers
  util/
    __init__.py          public surface
    options.py           load_yaml, get_config, instantiate_from_config, get_obj_from_str
    logger.py            CustomLogger + module-level singleton, basic_config, get_logger
    progress_bar.py      ProgressBar (Rich), RichProgressBarTheme
    vis.py               plot_image_grid, get_wandb_image, get_wandb_video
    info/                fork of `torchinfo` (model_summary, ColumnSettings, Mode, etc.)
```

## Entrypoint flow

There are two launcher patterns, mirroring the `dl/` vs `rl/`
subpackage split. Both end at the same place:
`options.instantiate_from_config(args.trainer, args).fit()`. Pick the
one that matches your training mode.

### DL launcher pattern (`jjuke/dl/main.py` + `jjuke/dl/train.py`)

DataLoader-driven training with multi-process DDP via
`accelerate launch`.

**Step 1 — `jjuke/dl/main.py`** (argparse launcher, runs in user shell):
1. Parse `--config_file --gpus --debug`.
2. Call `options.get_config(cfg_path, gpus, debug, save=True)` — loads YAML,
   resolves `__parent__` includes, expands `__pycall__` / `__pyobj__`
   markers, timestamps `exp_path` from `<exp_dir>/<YYMMDD_HHMM>_<stem>[_<memo>][_debug]`,
   creates the dir, dumps `args.yaml` next to it.
3. Build an Accelerate config dict from `args.gpus` (`multi_gpu`,
   `num_processes`, `num_cpu_threads_per_process`, `main_process_port` from
   `find_free_port()`).
4. `subprocess.run(["accelerate", "launch", ...flat_accel_args,
   "jjuke/dl/train.py", "--args_file", yaml_path])`.

**Step 2 — `jjuke/dl/train.py`** (the worker each Accelerate process runs):
```python
args = EasyDict(OmegaConf.load(opt.args_file))
if len(args.gpus) == 1:
    torch.cuda.set_device(args.gpus[0])
logger.basic_config(Path(args.exp_path) / "train.log")
args.log = logger.get_logger()
trainer = options.instantiate_from_config(args.trainer, args)
trainer.fit()
```

### RL launcher pattern (`jjuke/rl/main.py` + `jjuke/rl/train.py`)

Env-driven training. **In-process** dispatch (no subprocess) because
env-specific bootstrap (e.g. Isaac Sim's `AppLauncher`, MuJoCo model
load) typically must run in the same Python process that imports the
env. Single-process by default — multi-process DDP for env-driven RL
needs additional wiring outside this skeleton's scope.

**Step 1 — `jjuke/rl/main.py`**:
1. Parse `--config_file --gpus --debug --headless` (the `--headless`
   flag is added because RL envs frequently control a renderer).
2. Same `options.get_config(...)` call as the DL launcher.
3. **Direct in-process call**: `from jjuke.rl.train import train;
   train(args_file=..., headless=...)` — NO subprocess.

**Step 2 — `jjuke/rl/train.py`**:
1. Load `args.yaml`, set CUDA device.
2. **Env-specific bootstrap goes HERE** — BEFORE importing any env
   module. The skeleton has commented examples for Isaac Sim
   (`AppLauncher`), Gym, MuJoCo. Reference implementation lives at
   `isaaclab/train.py` (Isaac Sim Lab + InterMimic env).
3. Import jjuke `logger` + `options`, init logger.
4. `trainer = options.instantiate_from_config(args.trainer, args)` —
   typically a `BaseRLTrainer` subclass.
5. `trainer.fit()`.
6. Cleanup (close env / Sim app).

**Step 3 — `BaseTrainer.fit()`** (`core/trainer.py:768`):
- Calls `prepare_train()` → returns `(global_step, global_epoch)`,
  auto-resumes from `args.resume_ckpt` if present, builds `self.pbar`.
- Branches on `self.trainer_type`:
  - **EpochTrainer** (`hasattr(args, "train_epochs")`): outer loop over
    epochs → `pbar.start(...)` → `with self.accel.autocast(): self.train_epoch()`
    → `pbar.stop()` → periodic `valid_step()` and `save()`.
  - **StepTrainer** (`hasattr(args, "train_steps")`): single `pbar.start()`,
    nested epoch/batch loops, `with self.accel.autocast(): self.train_step(batch)`
    → periodic `valid_step()` and `save()`, breaks when `global_step >= total_steps`.
- Final: `unwrap_model()`, `ema.copy_to(model)` if EMA, `save_pipeline()`,
  `final_process()`, `accel.end_training()`.

The `trainer_type` is auto-selected from the cfg — exactly one of
`train_epochs` / `train_steps` MUST be set (asserted in `__init__`,
`core/trainer.py:89`).

## Config convention — `target` + `params`, no registry

Every block that needs to become a Python object uses string-based dynamic
import; there is no registry decorator. A canonical block:

```yaml
trainer:
  target: model.foo.MyTrainer        # importlib resolves "module.ClassName"
  params:                            # kwargs for the constructor
    num_saves: 5
    save_period: 30000
  argums:                            # optional positional args (rare)
    - 42
```

`instantiate_from_config(cfg, *args, **kwargs)` (`util/options.py:63`):
- Requires `target` (KeyError otherwise).
- Reads optional `argums` (list) and `params` (dict).
- Returns `get_obj_from_str(cfg["target"])(*argums, *args, **params, **kwargs)`.
- **Caller-provided positionals come AFTER `argums`**; kwargs from `params`
  and `kwargs` are merged (caller's `kwargs` win on collision via Python's
  normal kwarg-dup error).

### Recursive markers (resolved during `load_yaml` / `get_config`)

| Marker | Where | Effect |
|---|---|---|
| `__parent__: <path>` or `[<path>, <dot.path.to.subkey>]` | any node | Loads another YAML, optionally selects a subkey, then `OmegaConf.merge(parent, current)` so the current cfg overrides. |
| `__pycall__: { target, params }` | any node | Replaced with the result of `instantiate_from_config(...)`. |
| `__pyobj__: <module.ClassName>` or `{ target: <...> }` | any node | Replaced with the class object itself (no instantiation). |
| `__pyinstance__: { target, params }` | inside a `params` block | Same as `__pycall__` but resolved inside `instantiate_from_config` rather than at load time. |

### Top-level cfg keys consumed by `BaseTrainer`

| Key | Purpose |
|---|---|
| `exp_dir` | Root for the timestamped `exp_path`. Required by `get_config` to create the run directory. |
| `train_epochs` XOR `train_steps` | Selects EpochTrainer vs StepTrainer. Exactly one. |
| `seed` | Passed to `accelerate.utils.set_seed`. |
| `memo` | Optional suffix in the run-directory name. |
| `logging.use_wandb` / `logging.project_name` / `logging.push_to_hub` | Use either wandb or push_to_hub, not both. |
| `accel.dl_cfg` / `accel.deepspeed` / `accel.ddp_kwargs` | Each is a `target+params` block instantiated and passed to `Accelerator(...)`. |
| `model` | A `target+params` block (instantiated by the subclass's `build_network`). |
| `preprocessor` | `target+params`; instantiated by `build_preprocessor` with `device=self.device`. |
| `dataset` | `target+params`; the target MUST be callable returning `(dl_train, dl_valid)` or `(dl_train, dl_valid, dl_test)`. |
| `optim` | `target+params`; passed `model.parameters()` by the subclass. |
| `sched` | `target+params`; if `target` contains `"get_scheduler"`, the diffusers scheduler factory is used with `num_warmup_steps`/`num_training_steps` auto-scaled by `accel.num_processes`. Otherwise instantiated with `self.optim`. |
| `trainer` | `target+params` for the subclass itself. `params` are forwarded to `BaseTrainer.__init__` (`num_saves`, `save_period`, `valid_period`, `mixed_precision`, `clip_grad`, `grad_acc_steps`) plus subclass-specific kwargs. |
| `trainer.ema` | `target+params` instantiated as `self.ema_model` if `use_ema=True`. Reads `trainer.ema.offload` for CPU offload. |
| `resume_ckpt` | Path to a `checkpoint-N` directory; auto-resumes optimizer/model/scheduler state plus `ema_state_dict.pth`. |

## `BaseTrainer` subclass contract

### MUST override (enforced by `@abstractmethod`)

| Method | Where | Notes |
|---|---|---|
| `build_network(**kwargs)` | `core/trainer.py:242` | Instantiate `self.model`, `self.optim`, etc. End with `self.config_network(**kwargs)` to wire EMA / xformers / TF32 / grad-checkpoint / `scale_lr`. |
| `prepare_objects()` | `core/trainer.py:344` | Wrap with `self.model, self.optim, self.dl_train, self.sched, ... = self.accel.prepare(...)`. Pick which loaders go through Accelerator. |
| `BasePreprocessor.__call__(batch, augmentation=False)` | `core/trainer.py:61` | Move tensors to device (use `self._to(...)` / `self.batch_to_device(...)`), normalize, return a structured batch (the MambaDance-style pattern is `EasyDict`). |

### SHOULD override (stubbed `pass` — the loop is a no-op otherwise)

| Method | Where | Called from |
|---|---|---|
| `train_epoch()` | `core/trainer.py:613` | `fit()` EpochTrainer branch; takes no `batch` arg — iterate `self.dl_train` yourself. |
| `train_step(batch)` | `core/trainer.py:667` | `fit()` StepTrainer branch; receives `batch` from the outer for-loop. (The base def takes no `batch` arg, but `fit()` calls `self.train_step(batch)` at line 791 — overrides MUST accept it.) |
| `valid_step()` | `core/trainer.py:710` | Decorated with `@torch.no_grad()`; iterate `self.dl_valid`, end with `self.log_loss(...)`. |

### MAY override (stubbed `pass`)

| Method | Where | When |
|---|---|---|
| `save_pipeline()` | `core/trainer.py:762` | After fit, on main process only — emit a deployment-ready artifact (e.g. `diffusers` `save_pretrained` dir) separate from the resume checkpoint. |
| `final_process()` | `core/trainer.py:765` | After fit — final sampling / metrics / cleanup. |

### Provided by the base — do NOT reimplement

`build_accelerator`, `build_dataset`, `build_preprocessor`, `config_network`,
`prepare_accelerator`, `prepare_train`, `unwrap_model`, `save`,
`get_total_loss`, `gather_loss`, `clip_gradient`, `ema_step`, `log_loss`,
`fit`, plus the `trainer_type` / `device` / `ddp` / `log` / `model_params`
properties. `__init__` orchestrates the whole build chain in this order:
`build_accelerator` → seed → `build_network` → `build_dataset` →
`build_preprocessor` → `prepare_accelerator` (which calls `prepare_objects`).

## Public utility surface

What a subclass can call.

**`self.accel`** (HuggingFace `Accelerator`):
`prepare(...)`, `accumulate(model)`, `autocast()`, `backward(loss)`,
`clip_grad_norm_(params, max)`, `sync_gradients` (bool), `wait_for_everyone()`,
`is_main_process` / `is_local_main_process`, `save_state(path)`,
`load_state(path)`, `unwrap_model(model)`, `gather(tensor)`,
`log(dict_, step=...)`, `init_trackers(...)`, `end_training()`,
`num_processes`, `mixed_precision`.

**`self`** (BaseTrainer instance):
`unwrap_model()`, `save()`, `get_total_loss(losses_dict, verbose=False)`,
`gather_loss(batch_size, losses_dict)` (validation-time multi-process gather),
`clip_gradient(model)` (no-op when `clip_grad <= 0`), `ema_step()`,
`log_loss(data_dict, phase, images=None, caption=None)` where `phase ∈ {"train", "valid"}`,
plus properties `device`, `dtype` (alias of `weight_dtype`), `ddp`, `log`,
`model_params`, `trainer_type`. State: `self.global_step`, `self.global_epoch`,
`self.steps_per_epoch`, `self.total_steps`, `self.total_epochs`,
`self.weight_dtype`, `self.use_ema`, `self.ema_model`, `self.preprocessor`,
`self.dl_train` / `self.dl_valid` / `self.dl_test`, `self.pbar`.

**`jjuke.util`** (`util/__init__.py`):
`load_yaml(path)`, `get_config(path, gpus, debug=False, save=False)`,
`instantiate_from_config(cfg, *args, **kwargs)`, `get_obj_from_str(string)`,
`ProgressBar(is_main_process, trainer_type)`, `model_summary(...)`. The
`info/` submodule re-exports `ColumnSettings`, `Mode`, `RowSettings`,
`Units`, `Verbosity`, `ModelStatistics` (torchinfo enums).

**`jjuke.logger`**:
`basic_config(filename, lock=False)` (initializes the module-level singleton),
`get_logger() -> CustomLogger`, `timenow(braket=False)`. `CustomLogger` has
`debug / info / warn / error / fatal / flush` — all colored, with both stdout
and file output if a path was passed to `basic_config`.

**`jjuke.util.vis`**:
`plot_image_grid(images, plot_size=3, save_dir=None, filename=None, return_figure=False)`,
`get_wandb_image(pred_image, gt_image=None)`,
`get_wandb_video(video, fps=30)`.

**`ProgressBar`**:
`start(total_steps, current_step, total_epochs, current_epoch, steps_per_epoch, msg)`
(EpochTrainer needs `total_epochs/current_epoch/steps_per_epoch`; StepTrainer
needs `total_steps/current_step`), `update(step, metrics_dict)`, `stop()`.
No-ops on non-main processes.

## DataLoader wiring

`build_dataset()` (`core/trainer.py:326`) calls
`instantiate_from_config(self.args.dataset)` and unpacks the return:
- length-2 tuple → `self.dl_train, self.dl_valid`
- length-3 tuple → `self.dl_train, self.dl_valid, self.dl_test`
- anything else → `NotImplementedError`

So the cfg `dataset.target` MUST be a callable (typically a top-level
factory in `dataset/dataloader.py`) returning that tuple. The skeleton's
`load_dataloaders(batch_size, num_workers, ..., **kwargs)` is the canonical
signature shape — keep it for new consumers.

`build_preprocessor()` calls `instantiate_from_config(self.args.preprocessor,
device=self.device)` — so a `preprocessor:` block is required even if the
preprocessor is trivial. Subclass `BasePreprocessor` and implement
`__call__`.

## Distributed / mixed precision / checkpointing

- All multi-process routing goes through Accelerator. Do not call
  `torch.distributed` directly. The `accel:` cfg block configures
  `DataLoaderConfiguration`, `DeepSpeedPlugin`, and
  `DistributedDataParallelKwargs` via `target+params`.
- Mixed precision: set in the trainer's `params.mixed_precision: 'no' | 'fp16' | 'bf16'`.
  `False` is silently coerced to `"no"`. The `weight_dtype` property and
  `self.accel.autocast()` follow from this.
- Checkpoints are Accelerator's native `save_state(...)` directory format —
  written to `<exp_path>/checkpoint-{epoch|step}/`, NOT `.pt` files. The base
  also writes `ema_state_dict.pth` inside that directory if EMA is on. The
  `num_saves` parameter rotates the oldest out.
- `resume_ckpt` (path to a `checkpoint-N` dir) auto-resumes everything
  including EMA state, infers `(global_step, global_epoch)` from the dir
  name, and rewrites `exp_path` to point at the resumed run.

## Skeleton vs populated

**Populated** (use as-is):
- `BaseTrainer.fit`, `prepare_train`, `build_accelerator`, `build_dataset`,
  `build_preprocessor`, `prepare_accelerator`, `config_network`.
- All training utilities: `unwrap_model`, `save`, `get_total_loss`,
  `gather_loss`, `clip_gradient`, `ema_step`, `log_loss`.
- Resume logic (`__init__` + `prepare_train`).
- EMA wiring (`config_network` + `ema_step` + final `copy_to` in `fit`).
- `util/options.py` (config loading + instantiation).
- `util/logger.py`, `util/progress_bar.py`, `util/vis.py`, `util/info/`.
- `main.py` Accelerate launcher, `train.py` worker.

**Stubbed** — bodies are `pass` or commented examples; subclasses fill in:
- `BaseTrainer.train_epoch`, `BaseTrainer.train_step`, `BaseTrainer.valid_step`.
- `BaseTrainer.save_pipeline`, `BaseTrainer.final_process`.
- `BaseTrainer.build_network`, `BaseTrainer.prepare_objects` (both `@abstractmethod`).
- `BasePreprocessor.__call__` (`@abstractmethod`).
- `dataset/dataloader.py:ExampleDataset` (`__init__` / `__len__` /
  `__getitem__` are all `pass`); `load_dataloaders(...)` references
  `ExampleDataset(...)` placeholders that will fail at runtime.
- `config/model_to_experiment.yaml` is a schema template, not a runnable cfg
  (most `params:` use `...`).

Don't waste time chasing a missing body if it's listed above — that's
intentional, and the consumer is expected to provide it.

## Example consumer codebases

These are **examples**, not prescriptions — jjuke imposes the cfg
convention (`target` + `params`) and the subclass contracts
(`BaseTrainer` / `BaseRLTrainer`), but the surrounding directory
layout is the consumer's call. The shapes below are what's known to
work today and what the existing reference consumers
(`MambaDance` for DL, `isaaclab` for RL) follow.

### DL example codebase

DataLoader-driven training (supervised / self-supervised / generative
/ diffusion / contrastive — anything that iterates a `Dataset`).
Reference impl: `MambaDance` (a generative-diffusion project that
vendors `jjuke` and exercises every convention).

```
my_dl_project/
├── config/
│   └── exp.yaml              extends `jjuke/dl/config/exp_dl.yaml` via `__parent__:`
├── dataset/
│   ├── __init__.py
│   └── dataloader.py         custom `Dataset` + `load_dataloaders(...)` factory
│                             (returns `(dl_train, dl_valid)` — see jjuke/dl/dataset/dataloader.py)
├── model/
│   ├── __init__.py
│   ├── network.py            custom `nn.Module` (the model being trained)
│   └── trainer.py            subclass of `jjuke.core.trainer.BaseTrainer`;
│                             implements `build_network`, `prepare_objects`,
│                             `train_epoch` (or `train_step`), `valid_step`
├── main.py                   project's own launcher OR call `python -m jjuke.dl.main`
└── train.py                  project's own worker OR call `python -m jjuke.dl.train`
```

Smallest invocation (no project-side `main.py` / `train.py` needed —
use the library's launchers directly):
```bash
python -m jjuke.dl.main --config_file config/exp.yaml --gpus 0
```

### RL example codebase

Env-driven training (PPO, DAgger, off-policy methods). Reference
impl: `isaaclab` (Isaac Sim Lab + InterMimic Teacher PPO
+ Student DAgger).

```
my_rl_project/
├── config/
│   └── exp.yaml              extends `jjuke/rl/config/exp_rl.yaml` via `__parent__:`
├── env/
│   ├── __init__.py
│   ├── my_env.py             custom env class (Gym-style `step` / `reset` interface)
│   └── env_factory.py        `load_env(...)` factory called by the cfg's `dataset.target`
├── model/
│   ├── __init__.py
│   ├── network.py            custom policy (subclass `jjuke.rl.BasePolicy` or standalone)
│   ├── rollout_buffer.py     subclass of `jjuke.rl.BaseRolloutBuffer` (PPO / DAgger / etc.)
│   └── ppo_trainer.py        subclass of `jjuke.rl.BaseRLTrainer`;
│                             implements `_build_policy`, `update_from_rollout`,
│                             plus the env-loop wiring (see RLBaseTrainer skeleton)
├── scripts/
│   └── train.sh              shell wrapper that exports env vars + invokes the launcher
├── main.py                   project's own launcher (often a thin in-process variant of jjuke.rl.main
│                             that adds env-bootstrap before importing env modules)
└── train.py                  project's own worker (replaces the `# Env-specific bootstrap` block
                              in jjuke/rl/train.py with the actual env init, e.g. `AppLauncher` for Sim)
```

For envs that don't need in-process bootstrap (most Gym/MuJoCo envs)
the project can skip its own `main.py` / `train.py` and use the
library directly:
```bash
python -m jjuke.rl.main --config_file config/exp.yaml --gpus 0
```

For envs that need pre-import setup (Isaac Sim, certain GPU-physics
backends), copy `jjuke/rl/{main,train}.py` into the project and
fill the env-bootstrap block with the env's actual initialization.

## RL skeletons

### Layout

```
jjuke/
  core/
    normalizers.py       RunningMeanStd (concrete) + BaseObsNormalizer (skeleton)
  rl/                    NEW SUBPACKAGE
    __init__.py
    base_trainer.py      BaseRLTrainer(BaseTrainer) — env-loop scaffold
    buffers.py           BaseRolloutBuffer + GAE math (concrete) + episode-length helper
    policies.py          BasePolicy + Gaussian static helpers (log_prob / entropy / gaussian_kl)
```

`from jjuke.rl import BaseRLTrainer, BaseRolloutBuffer, BasePolicy` and
`from jjuke.core.normalizers import RunningMeanStd, BaseObsNormalizer`
are now part of the public surface (`jjuke.__all__` includes `rl`).

### Subclass contracts

#### `jjuke.rl.BaseRLTrainer(BaseTrainer)` — env-driven RL trainer

Same skeleton style as `BaseTrainer`: substantial scaffold + a few
abstract methods + commented-out reference impls.

| Method | Status | Notes |
|---|---|---|
| `_build_policy(obs_dim, action_dim) → nn.Module` | **abstract** | Construct the actor-critic network. |
| `update_from_rollout(flat: dict) → dict[str, float]` | **abstract** | Algorithm-specific update over the flat rollout buffer (PPO, DAgger, etc.). |
| `build_network`, `build_dataset`, `build_preprocessor`, `prepare_objects` | stub-with-comments | Subclasses fill in env construction + RMS init + Accelerator prep. Reference: `isaaclab/model/rl_base_trainer.py`. |
| `_collect_rollout`, `train_epoch`, `valid_step` | stub-with-comments | Env-step loop, GAE call, log_loss. |
| `save` | stub-with-comments | Drops `rl_state.pth` (RMS + curriculum-state extras) next to the Accelerator checkpoint. |
| `_save_state_extras() / _load_state_extras(state)` | concrete (default no-op) | Subclasses extend to checkpoint algorithm-specific state (DAgger curriculum). |
| `_PlaceholderDataset` / `_PlaceholderLoader` | concrete (private) | Length-1 stand-ins so `BaseTrainer.fit()`'s DataLoader-iterating loop runs in env-driven mode. |

Ctor signature (forwarded to `BaseTrainer`):
```python
BaseRLTrainer(args,
              gamma=0.99, tau=0.95, horizon_length=32,
              clip_actions=1.0, normalize_input=True,
              num_saves=None, save_period=500, valid_period=500,
              mixed_precision="no", clip_grad=1.0, grad_acc_steps=1,
              **kwargs)
```

#### `jjuke.rl.BaseRolloutBuffer` — fixed-size on-policy buffer

| Member | Status | Notes |
|---|---|---|
| `__init__` | concrete | Declares standard slots: obs / actions_raw / mus / log_sigmas / log_probs / values / rewards / dones / returns / advantages. |
| `reset()` | concrete | Resets the write index. |
| `append(obs, action_raw, mu, log_sigma, log_prob, value, reward, done)` | **abstract** | Subclass writes into the slots and bumps `_step`. |
| `get_flat() → dict[str, Tensor]` | **abstract** | Flatten `(horizon, num_envs, ...)` → `(horizon * num_envs, ...)`. Add domain-specific keys (e.g. `expert_mus`, `expert_mask` for DAgger). |
| `compute_gae(last_value, gamma, lam)` | concrete | Standard GAE with done-masking + bootstrap. Writes `returns` and `advantages`. |
| `episode_length_estimate()` | concrete | Approximate steps-until-first-done across envs. Diagnostic; rl_games' `episode_lengths/iter` is more exact. |

#### `jjuke.rl.BasePolicy(nn.Module)` — diagonal-Gaussian continuous policy

| Member | Status | Notes |
|---|---|---|
| `forward(obs) → (mu, log_sigma, value)` | **abstract** | Subclass builds the actor + critic forward. |
| `act(obs, deterministic=False)` | **abstract** | Sample (or take mu) + return diagnostics. Subclass should decorate with `@torch.no_grad()`. |
| `log_prob(action, mu, log_sigma)` | concrete static | Diagonal-Gaussian log-prob, summed over action dim. |
| `entropy(log_sigma)` | concrete static | Differential entropy of `N(mu, exp(log_sigma))`. |
| `gaussian_kl(mu_old, log_sigma_old, mu_new, log_sigma_new)` | concrete static | Analytic Gaussian KL — matches rl_games' `info/kl` convention (signed). |

These statics are **bit-identical** to the project's
`model.network.ActorCritic` static methods (verified in CI), so
consumers can call `jjuke.rl.BasePolicy.log_prob(...)` directly without
re-deriving the math.

#### `jjuke.core.normalizers.RunningMeanStd(nn.Module)` — concrete

`update(x)` (Welford), `normalize(x, eps)`. Buffers (`mean`, `var`,
`count`) round-trip through `state_dict()` / `load_state_dict()`.

#### `jjuke.core.normalizers.BaseObsNormalizer(BasePreprocessor)` — skeleton

Abstract `__call__(obs, augmentation=False)` + `state_dict()` /
`load_state_dict(state)`. The `state_dict` contract is what the trainer
hooks (`_save_state_extras` / `_load_state_extras`) rely on for RMS
checkpoint persistence.

### Reference impl (project consumer)

`isaaclab/model/` ships a fully working PPO + DAgger stack
that mirrors every skeleton above:

| Skeleton | Reference impl |
|---|---|
| `jjuke.rl.BaseRLTrainer` | `model.rl_base_trainer.RLBaseTrainer` |
| `jjuke.rl.BaseRolloutBuffer` | `model.rollout_buffer.RolloutBuffer` (+ `DAggerRolloutBuffer`) |
| `jjuke.rl.BasePolicy` | `model.network.ActorCritic` (MLP) and `model.network.TransformerActorCritic` |
| `jjuke.core.normalizers.RunningMeanStd` | `model.preprocessor.RunningMeanStd` (byte-identical copy) |
| `jjuke.core.normalizers.BaseObsNormalizer` | `model.preprocessor.ObsNormalizer` |

The project's classes do **NOT** currently subclass the jjuke skeletons
— they're independent reference implementations that mirror the
skeletons' structure. A future refactor could introduce subclassing
(the names and signatures are designed to make that drop-in); for now
the skeletons stand alone for future jjuke library users.

### Pip install (distant future)

The `isaaclab/jjuke/` package is structured to mirror
`/root/dev/jjuke/jjuke/` exactly so a future
`cp -r isaaclab/jjuke /root/dev/jjuke/jjuke` is a clean
diff. The `setup.py` at `/root/dev/jjuke/setup.py` will then expose
the new RL surface to anyone who runs `pip install
git+https://github.com/JJukE/JJuk_E.git`.
