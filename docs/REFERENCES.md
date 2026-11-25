# References — RoadVision-PDC

## Dataset

```bibtex
@article{arya2024rdd2022,
  title   = {RDD2022: A multi-national image dataset for automatic Road Damage Detection},
  author  = {Arya, Deeksha and Maeda, Hiroya and Ghosh, Sanjay Kumar and Toshniwal, Durga and Sekimoto, Yoshihide},
  journal = {Geoscience Data Journal},
  volume  = {11},
  number  = {4},
  pages   = {846--862},
  year    = {2024},
  publisher = {Wiley Online Library}
}
```

- Official repository: https://github.com/sekilab/RoadDamageDetector
- arXiv preprint (CRDDC'22 / RDD2022): https://arxiv.org/abs/2209.08538
- Challenge summary paper:
  `arya2022crowdsensing` — CRDDC'2022, IEEE Big Data 2022, pp. 6378–6386.
- Image license: **CC BY-SA 4.0**; repository code: MIT.

## Models

- SSDLite: Liu, W. et al. *SSD: Single Shot MultiBox Detector* (ECCV 2016).
  https://arxiv.org/abs/1512.02325
- MobileNetV3: Howard, A. et al. *Searching for MobileNetV3*
  (ICCV 2019). https://arxiv.org/abs/1905.02244
- Torchvision SSDLite320 docs:
  https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html
- Torchvision reference detection training recipe:
  https://github.com/pytorch/vision/blob/main/references/detection/

## Evaluation

- COCO mAP: Lin, T.-Y. et al. *Microsoft COCO: Common Objects in Context*
  (ECCV 2016). https://arxiv.org/abs/1405.0312
- pycocotools: https://github.com/cocodataset/cocoapi

## Parallel & distributed computing context

- CUDA C Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- PyTorch performance tuning:
  https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- Colab runtime docs: https://colab.research.google.com/notebooks/intro.ipynb

## Course material

- Assignment brief supplied by project owner (no project PDF available in
  repo root at setup time; course OpenMP Assignment #3 reviewed separately).
