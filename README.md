# Code for running test-time adaptation on sensor drift dataset

Python codes for running test-time adaptation algorithms, and semi-supervised adaptation on the sensor drift dataset [1]. Three methods are run for unlabelled adaptation, based on TENT [2], CoTTA [3] and Pseudolabeling [4].

How to run:
 1. Simply run the python code "src/main_batch.py". This loads a pre-trained base model (provided for reproducability), but code is available to run the training from scratch.

 [1] Vergara, A. (2012). Gas Sensor Array Drift Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5RP6W (licensed under CC BY 4.0 https://creativecommons.org/licenses/by/4.0)

 [2] Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2020). Tent: Fully test-time adaptation by entropy minimization. arXiv preprint arXiv:2006.10726.
 
 [3] Wang, Q., Fink, O., Van Gool, L., & Dai, D. (2022). Continual test-time domain adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 7201-7211).

 [4] Lee, D. H. (2013, June). Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. In Workshop on challenges in representation learning, ICML (Vol. 3, No. 2, p. 896).