# Vehicle re-Identification

## Introduction

Vehicle  re-Identification  aims  to  identify  the  samevehicle  across  the  different  cameras.  It  has  useful  applicationsin  surveillance  and  intelligent  transport  systems.  One  of  thefundamental  challenges  of  vehicle  re-identification  is  how  tolearn  robust  and  discriminative  visual  features  given  in  smallinter-class  similarity  and  large  intra-class  differences  that  don’tfollow the same distribution. In this project we propose to treatthe  re-identification  problem  as  a  domain  adaptation  task.  We implemented  the  Deep  Joint  Domain  Adaptation  algorithm  totrain  a  model  that  could  robustly  detect  the  same  classes  evenwhen  the  images  were  obtained  in  different  conditions.

## Preparation

### Dependencies
- Python 3
- PyTorch
- PIL
- Matplotlib

### Dataset

- VeRi-776
- Download the data and save in `./data/VeRi`  folder

## Authors

- [Israfel Salazar](https://github.com/israfelsr)
- [Yujin Cho](https://github.com/nuniniyujin)