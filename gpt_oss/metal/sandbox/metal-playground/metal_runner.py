import ctypes
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np
from Foundation import NSRange

# from Foundation import NSBundle, NSData
from Metal import (
    MTLCompileOptions,
    MTLCreateSystemDefaultDevice,
    MTLDataTypeUInt,
    MTLFunctionConstantValues,
    MTLResourceStorageModeShared,
    MTLSizeMake,
)


def build_fc_from_ordered(ordered: OrderedDict[str, int]):
    """
    ordered: OrderedDict of name->value where the order matches the function_constant indices.
             e.g., OrderedDict([("Bm",64), ("Bn",64), ("Bk",16), ("Sg_Bm",2), ("Sg_Bn",2)])
    Returns: MTLFunctionConstantValues with indices 0..len-1 set to those values.
    """
    if not isinstance(ordered, OrderedDict):
        raise TypeError("Pass an OrderedDict so order matches function_constant indices.")

    # Pack Python ints into a contiguous C array of uint32 (indices 0..N-1)
    vals = list(ordered.values())
    arr = (ctypes.c_uint * len(vals))(*map(int, vals))

    fc = MTLFunctionConstantValues.alloc().init()
    fc.setConstantValues_type_withRange_(memoryview(arr), MTLDataTypeUInt, NSRange(0, len(vals)))
    return fc


class MetalRunner:
    def __init__(self):
        self.device = MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal device found. Are you on Apple Silicon/macOS?")
        self.queue = self.device.newCommandQueue()
        if self.queue is None:
            raise RuntimeError("Failed to create MTLCommandQueue")

    def compile_library_from_source(
        self,
        source_str: str,
        label: str = "inmem",
        constants: Optional[OrderedDict[str, int]] = None,
    ) -> object:
        opts = MTLCompileOptions()  # default compile options
        if constants is not None:
            opts = MTLCompileOptions.alloc().init()
            print(constants)
            opts.setPreprocessorMacros_(constants)
        lib, err = self.device.newLibraryWithSource_options_error_(source_str, opts, None)
        if lib is None:
            raise RuntimeError(f"Metal compile failed for {label}: {err}")
        return lib

    def compile_library_from_file(
        self, path: str, constants: Optional[OrderedDict[str, int]] = None
    ) -> object:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        return self.compile_library_from_source(src, label=path, constants=constants)

    def pipeline(self, library: object, func_name: str) -> object:
        fn = library.newFunctionWithName_(func_name)
        if fn is None:
            raise RuntimeError(f"Function '{func_name}' not found in library.")
        pipeline, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if pipeline is None:
            raise RuntimeError(f"Failed to create pipeline for '{func_name}': {err}")
        return pipeline

    def make_buffer(self, arr_or_bytes, label: Optional[str] = None) -> object:
        if isinstance(arr_or_bytes, (bytes, bytearray, memoryview)):
            data = bytes(arr_or_bytes)
            length = len(data)
            buf = self.device.newBufferWithBytes_length_options_(
                data, length, MTLResourceStorageModeShared
            )
        else:
            arr = np.ascontiguousarray(arr_or_bytes)
            buf = self.device.newBufferWithBytes_length_options_(
                arr.tobytes(), arr.nbytes, MTLResourceStorageModeShared
            )
        if label:
            buf.setLabel_(label)
        return buf

    def run(
        self,
        pipeline: object,
        buffers: Dict[int, object],
        grid: Tuple[int, int, int],
        threads_per_tg: Tuple[int, int, int],
    ) -> Dict[str, float]:
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pipeline)
        for idx, buf in buffers.items():
            enc.setBuffer_offset_atIndex_(buf, 0, idx)
        grid_size = MTLSizeMake(*grid)
        tg_size = MTLSizeMake(*threads_per_tg)

        # This dispatch assumes that the grid_size is in terms of threads
        # not threadgroups.
        # enc.dispatchThreads_threadsPerThreadgroup_(grid_size, tg_size)

        # This dispatch assumes that the grid_size is in terms of threadgroups
        enc.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        t_start = cmd.GPUStartTime()
        t_end = cmd.GPUEndTime()
        assert t_start is not None and t_end is not None
        assert t_start >= 0 and t_end >= 0
        assert t_end >= t_start
        return {"gpu_ms": (t_end - t_start) * 1000.0}

    def max_threads_per_threadgroup(self) -> int:
        return self.device.maxThreadsPerThreadgroup().width
