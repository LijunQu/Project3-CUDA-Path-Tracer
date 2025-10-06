CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Lijun Qu
  * [LinkedIn](https://www.linkedin.com/in/lijun-qu-398375251/), [personal website](www.lijunqu.com).
* Tested on: Windows 11, i7-14700HX (2.10 GHz) 32GB, Nvidia GeForce RTX 4060 Laptop

* [Intro](#Introduction)
* [Features](#Features)
  * [BSDFs](#BSDFs)
  * [Integrators](#Integrators)
  * [Mesh and Texture Loading](#Mesh-Loading)
  * [Denoiser](#OIDN)
* [Perf Analysis](#Perf-Analysis)
* [Extras and Bloopers](#Extras-and-Bloopers)
* [Credits](#Credits)
-----

##### Example Renders:
<p align="center">
  <img src="img/matilda.png" width="1000" />
</p>

###### [Source](https://sketchfab.com/3d-models/matilda-7ddedfb652bd4ea091bc3de27f98fc02)

<p align="center">
  <img src="img/cathedral.png" width="1000" />
</p>

###### [Source](https://sketchfab.com/3d-models/cathedral-faed84a829114e378be255414a7826ca)

<p align="center">
  <img src="img/sliksong.png" width="1000" />
  </p>

###### [Source](https://sketchfab.com/3d-models/shaw-hornet-hollow-knight-silksong-670a87a9234c40bc9c2a4f274f6d8cc1)

<p align="center">
  <img src="img/abeautifulgame.png" width="1000" />
</p>