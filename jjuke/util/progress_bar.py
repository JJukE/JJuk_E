from dataclasses import dataclass
from typing import Dict, Union
from time import time

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.style import Style

__all__ = ["ProgressBar"]


@dataclass
class RichProgressBarTheme:
    """Styles to associate to different base components.

    Args:
        description: Style for the progress bar description. For eg., Epoch x, Testing, etc.
        progress_bar: Style for the bar in progress.
        progress_bar_finished: Style for the finished progress bar.
        progress_bar_pulse: Style for the progress bar when `IterableDataset` is being processed.
        batch_progress: Style for the progress tracker (i.e 10/50 batches completed).
        time: Style for the processed time and estimate time remaining.
        processing_speed: Style for the speed of the batches being processed.
        metrics: Style for the metrics

    https://rich.readthedocs.io/en/stable/style.html
    """

    description: Union[str, Style] = "white"
    progress_bar: Union[str, Style] = "#6206E0"
    progress_bar_finished: Union[str, Style] = "#228B22" # green
    progress_bar_pulse: Union[str, Style] = "#FFA500" # orange
    batch_progress: Union[str, Style] = "white"
    time: Union[str, Style] = "grey54"
    processing_speed: Union[str, Style] = "grey70"
    metrics: Union[str, Style] = "white"


class ProgressBar:
    """Create a progress bar with `rich text formatting <https://github.com/Textualize/rich>`_.

    Install it with pip:

    .. code-block:: bash

        pip install rich

    .. code-block:: python

        from jjuke.util import RichProgressBar

        pbar = ProgressBar()

    Args:
        refresh_rate: Determines at which rate (in number of batches) the progress bars get updated.
            Set it to ``0`` to disable the display.
        leave: Leaves the finished progress bar in the terminal at the end of the epoch. Default: False
        theme: Contains styles used to stylize the progress bar.
        console_kwargs: Args for constructing a `Console`
    """
    def __init__(self, is_main_process, trainer_type = "StepTrainer", theme: RichProgressBarTheme = RichProgressBarTheme()):
        assert trainer_type in ["EpochTrainer", "StepTrainer", None]
        self._progress_stopped = False
        self.is_main_process = is_main_process
        self.console = Console()
        self.theme = theme
        self.progress = None
        self.task_id = None
        self.start_time = None
        self.trainer_type = trainer_type

    def start(self, total_steps=None, current_step=None, total_epochs=None, current_epoch=None, steps_per_epoch=None, msg=""):
        """ Restart the progress bar at the beginning of each epoch. """
        msg = "[" + msg + "]" if msg != "" else ""
        
        assert (self.trainer_type == "EpochTrainer" and (total_epochs is not None and current_epoch is not None and steps_per_epoch is not None)) \
            or ((self.trainer_type == "StepTrainer" or self.trainer_type == None) and (total_steps is not None and current_step is not None))
        
        if not self.is_main_process:
            return
        
        if self.progress:
            self.progress.stop()
        
        if self.trainer_type == "EpochTrainer":
            self.progress = Progress(
                TextColumn(f"{msg} [{self.theme.description}]Epoch {{task.fields[current_epoch]}}/{{task.fields[total_epochs]}}"),
                BarColumn(
                    complete_style=self.theme.progress_bar,
                    finished_style=self.theme.progress_bar_finished,
                    pulse_style=self.theme.progress_bar_pulse,
                ),
                TextColumn(f"[{self.theme.batch_progress}]{{task.completed}}/{{task.total}}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn(f"[{self.theme.processing_speed}]{{task.fields[speed]}} it/s"),
                TextColumn(f"[{self.theme.metrics}]{{task.fields[metrics]}}"),
                transient=True,
            )
            self.task_id = self.progress.add_task(
                "[cyan]Training",
                total=steps_per_epoch, # total steps per epoch
                metrics="--",
                speed="--",
                current_step=0 if current_step is None else current_step,
                current_epoch=current_epoch,
                total_epochs=total_epochs,
            )
        else:
            self.progress = Progress(
                TextColumn(f"{msg} [{self.theme.description}]Step"), # {{task.fields[current_step]}}/{{task.fields[total_steps]}}"),
                BarColumn(
                    complete_style=self.theme.progress_bar,
                    finished_style=self.theme.progress_bar_finished,
                    pulse_style=self.theme.progress_bar_pulse,
                ),
                TextColumn(f"[{self.theme.batch_progress}]{{task.completed}}/{{task.total}}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn(f"[{self.theme.processing_speed}]{{task.fields[speed]}} it/s"),
                TextColumn(f"[{self.theme.metrics}]{{task.fields[metrics]}}"),
                transient=True,
            )
            self.task_id = self.progress.add_task(
                "[cyan]Training" if self.trainer_type is not None else "",
                total=total_steps,
                metrics="--",
                speed="--",
                current_step=current_step,
                # total_steps=total_steps,
            )
        self.progress.start()
        self.start_time = time()
    
    def update(self, step, metrics_dict: Dict[str, float]):
        """ Update progress bar with losses. """
        
        if not self.is_main_process or self.progress is None:
            return
        
        elapsed_time = time() - self.start_time
        speed = f"{step / elapsed_time:.2f}" if elapsed_time > 0 else "--"
        
        metrics = ", ".join([f"{k}: {v:.4f}" for k, v in metrics_dict.items()]) # convert metrics to readable format
        
        # the progress bar is completely filled when current_step == total
        self.progress.update(self.task_id, advance=1, metrics=metrics, speed=speed, current_step=step)
        # if self.trainer_type == "EpochTrainer":
        #     self.progress.update(self.task_id, advance=1, metrics=metrics, speed=speed, current_step=step)
        # else:
        #     self.progress.update(self.task_id, advance=1, metrics=metrics, speed=speed, current_step=step)
    
    def stop(self):
        """ Stop the progress bar after training is completed. """
        if self.is_main_process and self.progress is not None:
            self.progress.stop()

