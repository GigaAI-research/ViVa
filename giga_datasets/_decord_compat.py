"""decord-compatible VideoReader, backed by torchcodec when decord is absent.

If `decord` is importable, it is re-exported unchanged so behavior matches the
original codebase. Otherwise a thin wrapper around `torchcodec.decoders.VideoDecoder`
exposes the subset of the decord API actually used here:
``len(vr)``, ``vr.get_avg_fps()``, ``vr.get_batch(indices)``, ``vr.next()``,
``vr[idx]`` / ``vr[slice]``, and iteration.

`get_batch` and `next` return CPU ``torch.Tensor`` (NHWC uint8), which matches
decord's behavior when the torch bridge is enabled — call sites in this repo
already handle that case via ``isinstance(frame, torch.Tensor)``.
"""

from __future__ import annotations

import importlib.util
import io

_HAS_DECORD = importlib.util.find_spec("decord") is not None

if _HAS_DECORD:
    from decord import VideoReader  # type: ignore  # noqa: F401
else:
    import torch
    from torchcodec.decoders import VideoDecoder

    class VideoReader:
        def __init__(
            self,
            uri,
            ctx=None,
            width: int = -1,
            height: int = -1,
            num_threads: int = 0,
            fault_tol: int = -1,
        ) -> None:
            if isinstance(uri, io.BytesIO):
                uri = uri.getvalue()
            self._decoder = VideoDecoder(uri, dimension_order="NHWC")
            self._iter_idx = 0

        def __len__(self) -> int:
            return int(self._decoder.metadata.num_frames)

        def get_avg_fps(self) -> float:
            return float(self._decoder.metadata.average_fps)

        def get_batch(self, indices) -> "torch.Tensor":
            if hasattr(indices, "tolist"):
                indices = indices.tolist()
            indices = [int(i) for i in indices]
            return self._decoder.get_frames_at(indices=indices).data

        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return self.get_batch(list(range(*idx.indices(len(self)))))
            if isinstance(idx, (list, tuple)) or (
                hasattr(idx, "__iter__") and not isinstance(idx, (str, bytes))
            ):
                return self.get_batch(idx)
            return self._decoder.get_frame_at(int(idx)).data

        def next(self):
            if self._iter_idx >= len(self):
                raise StopIteration
            frame = self._decoder.get_frame_at(self._iter_idx).data
            self._iter_idx += 1
            return frame

        def __iter__(self):
            self._iter_idx = 0
            return self

        def __next__(self):
            return self.next()
