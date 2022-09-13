# Stable Diffusion with 🤗 Diffusers

## Inference Performance Comparison between GPU, CPU, and Apple M1/M2

**Our results: Average time for one (diffusion) step (measured in seconds)**

| Hardware                   | (512, 512) | (768, 768) |
|----------------------------|------------|------------|
| Nvidia RTX 8000            | 0.23s      | 0.81       |
| Apple M1 Pro (16‑Core GPU) | 1.30s      | 6.67       |
| Apple M1 Pro (10‑Core CPU) | 4.77s      | 15.41      |

Average total time to generate one image using 200 diffusion steps:
- Nvidia RTX 8000: ~45s / ~2.5 min
- Apple M1 Pro (16‑Core GPU): ~4.3min / ~22min
- Apple M1 Pro (10‑Core CPU): ~15min / ~51min

**Details**
- We run the diffusion process for 50, 150 and 200 diffusion steps calculated the average time for one step
- Each run was repeated 5 times and results were again averages
- We did not use half-float precision, since it is not (yet) available for MPS at the moment (should also mainly affect memory allocation)
- Going beyond 768 was not able for the M1 with 32GB shared memory