import csv
import logging
import os
import stat
import time
from datetime import datetime

import torch
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger("benchmark")


class BenchmarkCallback(Callback):
    def __init__(
        self,
        model_name: str,
        precision: str,
        workers: int,
        warmup_steps: int = 50,
    ):
        self.warmup_steps = warmup_steps
        self.start_time = 0
        self.end_time = 0
        self.precision = precision
        self.model = model_name
        self.workers = workers

    def on_fit_start(self, trainer, pl_module):
        logger.info(
            f"Benchmark started. Number of warmup iterations: {self.warmup_steps}"
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int):
        if batch_idx == self.warmup_steps:
            logger.info(
                f"Completed {self.warmup_steps} warmup steps. Benchmark timer started."
            )
            self.start_time = time.time()

    def on_fit_end(self, trainer, pl_module):
        self.end_time = time.time()
        logger.info("Fit function finished")

        dataset = trainer.train_dataloader.dataset
        batch_size = trainer.train_dataloader.batch_size
        in_w, in_h = dataset.width, dataset.height

        benchmark_steps = trainer.global_step - self.warmup_steps
        processed_megapixels = (
            trainer.world_size * in_w * in_h * batch_size * benchmark_steps / 1e6
        )

        elapsed_time = (
            self.end_time - self.start_time
        ) + 1e-7  # for numerical stability
        mpx_s = processed_megapixels / (elapsed_time)

        processed_imgs = batch_size * benchmark_steps * trainer.world_size
        images_s = processed_imgs / (elapsed_time)

        batches_s = benchmark_steps * trainer.world_size / elapsed_time

        logger.info(f"Benchmark finished in {elapsed_time:.1f} seconds")
        logger.info(
            f"Average training throughput: {mpx_s:.2f} MPx/s (megapixels per second) | "
            + f"{images_s:.2f} images/s | {batches_s:.2f} batches/s"
        )

        os.makedirs("./benchmarks", exist_ok=True)
        csv_path = os.path.join("./benchmarks", "benchmark.csv")
        file_exists = os.path.isfile(csv_path) and os.stat(csv_path).st_size >= 0
        with open(csv_path, "a") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(
                    [
                        "Datetime",
                        "GPU",
                        "cuDNN version",
                        "N GPUs",
                        "Data Loader workers",
                        "Model",
                        "Precision",
                        "Minibatch",
                        "Input width [px]",
                        "Input height [px]",
                        "Warmup steps",
                        "Benchmark steps",
                        "MPx/s",
                        "images/s",
                        "batches/s",
                    ]
                )

            data = [
                datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                torch.cuda.get_device_name(0),
                torch.backends.cudnn.version(),
                trainer.world_size,
                self.workers,
                self.model,
                self.precision,
                batch_size,
                in_w,
                in_h,
                self.warmup_steps,
                benchmark_steps,
                mpx_s,
                images_s,
                batches_s,
            ]
            writer.writerow(data)
            logger.info(
                "Written benchmark data to a CSV file. "
                + "See 'Logging Results to a Persisent CSV File' section to "
                + "save the file on your disk: "
                + "https://github.com/tensorpix/benchmarking-cv-models#logging-results-to-a-persistent-csv-file"
            )

        try:
            os.chmod(
                csv_path,
                stat.S_IRUSR
                | stat.S_IRGRP
                | stat.S_IWUSR
                | stat.S_IROTH
                | stat.S_IWOTH,
            )
        except Exception as e:
            logger.error(f"Failed to change csv permissions: {e}")
