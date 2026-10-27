#!/usr/bin/env python
"""
kprofile.py
============

See CSGOptiX/CSGOptiX7.cu WITH_KERNEL_PROFILE


"""
import numpy as np

k = np.load("kprofile.npy")

# RTX 5000 Ada boost clock frequency in Hz (2.55 GHz)
clock_freq_hz = 2.55e9

# Extract columns
sample_counts = k[:, 0]
trace_cycles = k[:, 1]
prop_cycles = k[:, 2]

# Filter out empty/trailing rows where no photons reached that bounce
valid = sample_counts > 0

# Convert cycles to milliseconds per sampled photon
# Time (ms) = (cycles / frequency) * 1000 / sample_count
trace_ms = (trace_cycles[valid] / clock_freq_hz * 1000.0) / sample_counts[valid]
prop_ms = (prop_cycles[valid] / clock_freq_hz * 1000.0) / sample_counts[valid]

for b, (t, p, count) in enumerate(zip(trace_ms, prop_ms, sample_counts[valid])):
    ratio = t / p if p > 0 else float('inf')
    print(f"Bounce {b:2d} | Trace: {t:6.4f} ms | Propagate: {p:6.4f} ms | Ratio (T/P): {ratio:5.1f}x")


total_trace_s = trace_ms.sum()*255/1000
total_prop_s = prop_ms.sum()*255/1000


print(f" total_trace_s {total_trace_s:8.2f} ")
print(f" total_prop_s  {total_prop_s:8.2f} ")

