# Stable Diffusion with 🤗 Diffusers

## Inference Performance Comparison between GPU, CPU, and Apple M1/M2

**Our results: Average time for one (diffusion) step (measured in seconds)**

| Hardware                   | (512, 512) | (768, 768) |
|----------------------------|------------|------------|
| Nvidia RTX 8000            |            |            |
| Apple M1 Pro (10‑Core CPU) |            |            |
| Apple M1 Pro (16‑Core GPU) |            |            |

**Details**
- We run the diffusion process for 50, 150 and 200 steps calculated the average time for one step
- Each run was repeated 5 times and results were averages
- We did not use half-float precision, since it is not available for MPS at the moment
