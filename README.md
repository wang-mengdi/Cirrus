# *Cirrus*: Adaptive Hybrid Particle-Grid Flow Maps on GPU

[**Mengdi Wang** ](https://wang-mengdi.github.io/), [Fan Feng](https://sking8.github.io/), Junlin Li, [Bo Zhu](https://www.cs.dartmouth.edu/~bozhu/)

[![webpage](https://img.shields.io/badge/Project-Homepage-green)](https://wang-mengdi.github.io/proj/25-cirrus/)
[![paper](https://img.shields.io/badge/Paper-Preprint-red)](https://wang-mengdi.github.io/proj/25-cirrus/cirrus-preprint.pdf)
[![code](https://img.shields.io/badge/Source_Code-Github-blue)](https://github.com/wang-mengdi/Cirrus)

This repo stores the source code of our SIGGRAPH 2025 paper ***Cirrus*: Adaptive Hybrid Particle-Grid Flow Maps on GPU.**

<figure>
  <img src="./representative-image.jpg" align="left" width="100%" style="margin: 0% 5% 2.5% 0%">
  <figcaption>Left: smoke (large) passing a racing car and its vorticity field (small). Right: smoke (large) passing an aircraft with 4 rotating propellers at a 15-degree angle of attack and its vorticity field (small). The wingtip vortices are captured by our algorithm in the vorticity field. Effective resolutions are 512x512x1024 on our adaptive grid implemented on GPU.</figcaption>
</figure>
<br />

## Abstract

We propose the adaptive hybrid particle-grid flow map method, a novel flow-map approach that leverages Lagrangian particles to simultaneously transport impulse and guide grid adaptation, introducing a fully adaptive flow map-based fluid simulation framework. The core idea of our method is to maintain flow-map trajectories separately on grid nodes and particles: the grid-based representation tracks long-range flow maps at a coarse spatial resolution, while the particle-based representation tracks both long and short-range flow maps, enhanced by their gradients, at a fine resolution. This hybrid Eulerian-Lagrangian flow-map representation naturally enables adaptivity for both advection and projection steps. We implement this method in Cirrus, a GPU-based fluid simulation framework designed for octree-like adaptive grids enhanced with particle trackers. The efficacy of our system is demonstrated through numerical tests and various simulation examples, achieving up to $512\times 512\times 2048$ effective resolution on an RTX 4090 GPU. We achieve a 1.5 to 2 $\times$ speedup with our GPU optimization over the Particle Flow Map method on the same hardware, while the adaptive grid implementation offers efficiency gains of one to two orders of magnitude by reducing computational resource requirements. The source code has been made publicly available at: [https://wang-mengdi.github.io/proj/25-cirrus/](https://wang-mengdi.github.io/proj/25-cirrus/).

## Usage

First, install the following dependencies:

- [xmake](https://xmake.io/)
- A C++ build environment (e.g., Visual Studio on Windows)
- NVIDIA CUDA Toolkit (for `nvcc`)

To build the project, run:

```bash
xmake -v
```

To run a simulation:

```bash
xmake r scenes/smokesphere.json
```

That runs for 67 seconds on a desktop with Intel i9-14900KF CPU, RTX 4090 (24G dedicated VRAM) GPU and 64G memory.