"""Ordered, bounded CPU episode preparation using CUDA-free spawn workers."""
from __future__ import annotations

import multiprocessing as mp
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import AbstractContextManager

import torch

from looplora.data.relations import EpisodeArrays, TaskSpec, generate
from looplora.data.render import load_name_pool
from .batching import TokenBatch, tokenize_batch


def _initialize(spec, split, batch_size, seed, tokenizer):
    global _settings
    # Spawn never inherits a CUDA context. Tokenization and tensor assembly use
    # one CPU thread per worker, independently of the training process's settings.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    _settings = spec, split, batch_size, seed, tokenizer, load_name_pool()


def _produce(step):
    spec, split, batch_size, seed, tokenizer, pool = _settings
    arrays = generate(spec, split, step * batch_size, batch_size, seed)
    tokens = tokenize_batch(arrays, tokenizer, device="cpu", name_pool=pool)
    return arrays, tokens


class BatchStream(AbstractContextManager):
    """Yield absolute steps [start_step, steps), with at most 2*workers pending.

    Zero workers keeps the original synchronous generator and device-side
    tokenization path (tokens are None). Nonzero workers return CPU TokenBatch
    objects. Use a context manager so early exits and training exceptions join
    workers. The tokenizer is copied into spawned workers, never the model.
    """

    def __init__(self, spec: TaskSpec, split: str, batch_size: int, seed: int,
                 steps: int, *, tokenizer, workers: int = 0, start_step: int = 0):
        for name, value, minimum in (("workers", workers, 0), ("steps", steps, 0),
                                     ("start_step", start_step, 0), ("batch_size", batch_size, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if start_step > steps:
            raise ValueError("start_step must not exceed steps")
        self.spec, self.split, self.batch_size, self.seed = spec, split, batch_size, seed
        self.steps, self.workers, self.tokenizer = steps, workers, tokenizer
        self._step = self._submit_step = start_step
        self._pool = None
        self._pending = deque()
        self._closed = False

    def __enter__(self):
        if self._closed:
            raise RuntimeError("batch stream is closed")
        if self.workers and self._step < self.steps and self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=mp.get_context("spawn"),
                initializer=_initialize,
                initargs=(self.spec, self.split, self.batch_size, self.seed, self.tokenizer),
            )
            try:
                self._fill()
            except BaseException:
                self.close()
                raise
        return self

    def _fill(self):
        while self._submit_step < self.steps and len(self._pending) < 2 * self.workers:
            self._pending.append(self._pool.submit(_produce, self._submit_step))
            self._submit_step += 1

    def __iter__(self):
        return self

    def __next__(self) -> tuple[EpisodeArrays, TokenBatch | None]:
        if self._closed or self._step >= self.steps:
            self.close()
            raise StopIteration
        try:
            if self.workers:
                self.__enter__()
                result = self._pending.popleft().result()
                self._fill()
            else:
                result = (generate(self.spec, self.split, self._step * self.batch_size,
                                   self.batch_size, self.seed), None)
            self._step += 1
            return result
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self._closed:
            self._closed = True
            if self._pool is not None:
                self._pool.shutdown(wait=True, cancel_futures=True)
            self._pending.clear()

    def __exit__(self, *exc):
        self.close()
        return False
