import logging
import time

from lightning.pytorch.callbacks import Callback

logger = logging.getLogger("benchmark")


class BenchmarkCallback(Callback):
    def __init__(self, warmup_steps: int = 50):
        self.warmup_steps = warmup_steps
        self.start_time = 0
        self.end_time = 0

    def on_fit_start(self, trainer, pl_module):
        logger.info(
            f"Benchmark started. Number of warmup iterations: {self.warmup_steps}"
        )
# AAAAAAAAAAAAAAAAAAAAAAAAAAASDASDASDASDASDAAAAAAAAAAAAAAAAAAAAAAAAAAASDASDASDASDASDAAAAAAAAAAAAAAAAAAAAAAAAAAASDASDASDASDASDAAAAAAAAAAAAAAAAAAAAAAAAAAASDASDASDASDASDAAAAAAAAAAAAAAAAAAAAAAAAAAASDASDASDASDASDAAAAAAAAAAAAAAAAAAAAAAAAAAASDASDASDASDASDAAAAAAAAAAAAAAAAAAAAAAAAAAASDASDASDASDASDAAAAAAAAAAAAAAAAAAAAAAAAAAASDASDASDASDASDAAAAAAAAAAAAAAAAAAAAAAAAAAASDASDASDASDASD
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
            f"Average training throughput: {mpx_s:.2f} Mpx/s (megapixels per second) | "
            + f"{images_s:.2f} images/s | {batches_s:.2f} batches/s"
        )
