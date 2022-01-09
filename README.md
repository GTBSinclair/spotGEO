# spotGEO

## Introduction

spotGEO was a 2020 ESA competition to develop an algorithm that could detect geo-stationary satellites from ground-based telescope images. Depending on parameter settings, 'Main algorithm v2' can achieve an accuracy of 99% in a test dataset of true and false GEO sats, however the kappa score is still low due to poor performance in classifying true GEO sats. To achieve the best results, the localising aspect of the algorithm has to be developed and tested on the dataset of ground-based telescope images.

## Dataset

The spotGEO training dataset (available at https://kelvins.esa.int/spot-the-geo-satellites/dataset/) is very large and can be used to train AI neural networks. In this algorithm, however, it is only used as a testing dataset. A subset of this dataset containing 5000 images of false sats and 15 images of true sats is used.

<p align="center">
  <img src="https://github.com/P9981/spotGEO/blob/main/images/True_sat.png" width="250" title="hover text">
  <img src="https://github.com/P9981/spotGEO/blob/main/images/False_sat.png" width="250" alt="accessibility text">
  <img src="https://github.com/P9981/spotGEO/blob/main/images/Area1_Area2.png" width="250" alt="accessibility text">
</p>

## Main algorithm

The algorithm starts by identifying two regions of pixels in each image, i.e. Area 1 and Area 2. Area 1 contains the inner 9 pixels and Area 2 contains the outer 40 pixels. A key fact is that Area 1 and Area 2 will significantly differ in brightness when a true satellite is present in the image. 
The algorithm calculates the average of pixel brightness for both areas, i.e. avg. Area 1 and avg. Area 2. The difference between the two is then calculated, so avg. Area 1 - avg. Area 2. If the area difference is below the threshold it means that there isn't enough brightness discrepancy between the areas and the image is labelled '0' for false satellite. Alternatively, if the difference in areas is above the threshold then its likely a true satellite and its labelled '1'. 
The above process is then iterated over all images in the dataset.

<p align="center">
  <img src="https://github.com/P9981/spotGEO/blob/main/images/spotGEO_main_algorithm_v2.png" width="350" title="hover text">
</p>

## Results

The figure below shows the value of avg. Area 1 - avg. Area 2 for every image in the dataset. Through a process of optimization, the optimal threshold to identify true sats from false sats was found at y = 0.012, indicated by the red line. This threshold allowed the algorithm to correctly identify 46% of true sats and 99% of false sats.

<p align="center">
  <img src="https://github.com/P9981/spotGEO/blob/main/images/threshold.png" width="550" title="hover text">
</p>

The threshold performance is analysed in the figure below using the Kappa score as an indicator of success (a higher kappa score means better performance)

<p align="center">
  <img src="https://github.com/P9981/spotGEO/blob/main/images/threshold_performance3.png" width="550" title="hover text">
</p>

## Improvements

A major improvement is to lower the algorithm threshold so that it catches all the true sats and modify the algorithm to work on the much larger ground-based telescope images (available at https://kelvins.esa.int/spot-the-geo-satellites/problem/). By locating the possible satellites in each image and quantifying their movement in the same 'sequence' it will be possible to filter all the false positives and be left with only the true satellites. The true satellites will be visible in all the images in the sequence whereas the false positives won't be.
