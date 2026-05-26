# S40: 200x100 Failure Diagnostics

## 1. S39 Overall Conclusion

S40 reads only the completed S39 metrics and PNG artifacts. It does not run new training.

| run | avg defect_iou | avg defect_area_pred | avg mu_mse | avg mu_mae |
| --- | ---: | ---: | ---: | ---: |
| baseline | 7.472148e-02 | 1.095210e+04 | 5.040013e+05 | 5.185692e+02 |
| temp25_lambda5_outer60 | 7.595984e-01 | 1.333170e+03 | 6.412482e+04 | 1.794309e+02 |

Key conclusion: `temp25_lambda5_outer60` improves IoU over baseline on all 100/100 S39 samples and remains the current 200x100 default semi-supervised BCE candidate. The remaining weakness is concentrated in false-positive diffusion, area overprediction, centroid shift, and local shape-detail mismatch. This is still a semi-supervised BCE mask-prior upper bound, not unsupervised weak-form success.

## 2. Representative Successful Samples

Top-5 samples by final `temp25_lambda5_outer60` IoU:

| sample | final IoU | defect_area_pred / label | centroid offset | figure |
| ---: | ---: | ---: | ---: | --- |
| 5 | 0.993639 | 782 / 785 | 0.0029 | sample_005_top_iou_5_iou_0.9936.png |
| 45 | 0.987934 | 660 / 658 | 0.0088 | sample_045_top_iou_5_iou_0.9879.png |
| 81 | 0.987179 | 616 / 624 | 0.0072 | sample_081_top_iou_5_iou_0.9872.png |
| 50 | 0.983636 | 548 / 543 | 0.0023 | sample_050_top_iou_5_iou_0.9836.png |
| 77 | 0.981386 | 961 / 955 | 0.0150 | sample_077_top_iou_5_iou_0.9814.png |

## 3. Clear Weak Samples

Lowest-10 samples by final `temp25_lambda5_outer60` IoU:

| sample | final IoU | defect_area_pred / label | centroid offset | baseline IoU | improvement | failure type |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 40 | 0.132256 | 8975 / 1187 | 1.2791 | 0.074890 | 0.057367 | low IoU; area overprediction; centroid shift; false positive diffusion; smallest baseline-to-final improvement |
| 47 | 0.134349 | 4252 / 662 | 2.6909 | 0.040092 | 0.094257 | low IoU; area overprediction; centroid shift; false positive diffusion; smallest baseline-to-final improvement |
| 17 | 0.179130 | 5143 / 959 | 3.9780 | 0.066941 | 0.112189 | low IoU; area overprediction; centroid shift; false positive diffusion; smallest baseline-to-final improvement |
| 57 | 0.190597 | 5108 / 995 | 3.5255 | 0.077462 | 0.113135 | low IoU; area overprediction; centroid shift; false positive diffusion; smallest baseline-to-final improvement |
| 73 | 0.201920 | 2971 / 660 | 1.5880 | 0.051785 | 0.150135 | low IoU; area overprediction; centroid shift; false positive diffusion; smallest baseline-to-final improvement |
| 95 | 0.226625 | 4813 / 1130 | 2.0836 | 0.113192 | 0.113433 | low IoU; area overprediction; centroid shift; false positive diffusion; smallest baseline-to-final improvement |
| 54 | 0.239952 | 6654 / 1645 | 2.8492 | 0.133155 | 0.106797 | low IoU; area overprediction; centroid shift; false positive diffusion; smallest baseline-to-final improvement |
| 29 | 0.267585 | 3391 / 1024 | 3.0916 | 0.067156 | 0.200429 | low IoU; area overprediction; centroid shift; false positive diffusion; smallest baseline-to-final improvement |
| 49 | 0.332601 | 4512 / 1550 | 1.6522 | 0.109984 | 0.222617 | low IoU; area overprediction; centroid shift; false positive diffusion; smallest baseline-to-final improvement |
| 43 | 0.337901 | 3135 / 1454 | 2.4229 | 0.117991 | 0.219910 | low IoU; area overprediction; centroid shift; false positive diffusion; smallest baseline-to-final improvement |

## 4. Ranking Summary

- Highest IoU 5 samples: 5, 45, 81, 50, 77
- Lowest IoU 10 samples: 40, 47, 17, 57, 73, 95, 54, 29, 49, 43
- Largest absolute area-error 10 samples: 40, 54, 17, 57, 95, 47, 49, 41, 29, 73
- Largest centroid-offset 10 samples: 17, 57, 29, 54, 47, 31, 43, 86, 41, 1
- Largest baseline-to-final improvement 10 samples: 98, 5, 81, 45, 75, 35, 26, 82, 58, 89
- Smallest baseline-to-final improvement 10 samples: 40, 47, 54, 17, 57, 95, 73, 29, 43, 49

The complete ranking table is saved in `diagnostics_rankings.csv`.

## 5. Representative Figures

Copied 20 representative PNGs from S39 `temp25_lambda5_outer60` sample directories into `figures/`. No `.npy`, model weights, checkpoints, or temporary files are copied.

| sample | source ranking | copied PNG |
| ---: | --- | --- |
| 5 | top_iou_5 | figures/sample_005_top_iou_5_iou_0.9936.png |
| 45 | top_iou_5 | figures/sample_045_top_iou_5_iou_0.9879.png |
| 81 | top_iou_5 | figures/sample_081_top_iou_5_iou_0.9872.png |
| 50 | top_iou_5 | figures/sample_050_top_iou_5_iou_0.9836.png |
| 77 | top_iou_5 | figures/sample_077_top_iou_5_iou_0.9814.png |
| 40 | low_iou_10 | figures/sample_040_low_iou_10_iou_0.1323.png |
| 47 | low_iou_10 | figures/sample_047_low_iou_10_iou_0.1343.png |
| 17 | low_iou_10 | figures/sample_017_low_iou_10_iou_0.1791.png |
| 57 | low_iou_10 | figures/sample_057_low_iou_10_iou_0.1906.png |
| 73 | low_iou_10 | figures/sample_073_low_iou_10_iou_0.2019.png |
| 95 | low_iou_10 | figures/sample_095_low_iou_10_iou_0.2266.png |
| 54 | low_iou_10 | figures/sample_054_low_iou_10_iou_0.2400.png |
| 29 | low_iou_10 | figures/sample_029_low_iou_10_iou_0.2676.png |
| 49 | low_iou_10 | figures/sample_049_low_iou_10_iou_0.3326.png |
| 43 | low_iou_10 | figures/sample_043_low_iou_10_iou_0.3379.png |
| 41 | area_abs_error_10 | figures/sample_041_area_abs_error_10_iou_0.3802.png |
| 31 | centroid_offset_10 | figures/sample_031_centroid_offset_10_iou_0.4506.png |
| 86 | centroid_offset_10 | figures/sample_086_centroid_offset_10_iou_0.3658.png |
| 1 | centroid_offset_10 | figures/sample_001_centroid_offset_10_iou_0.4035.png |
| 98 | improvement_largest_10 | figures/sample_098_improvement_largest_10_iou_0.9771.png |

## 6. Current Judgment

`temp25_lambda5_outer60` is overall effective for 200x100 / 100 samples. Weak samples should be analyzed from shape-detail mismatch, area control, local false positives, and centroid shift rather than by more blind sweeps over radius, center mode, or area-prior strength.

## 7. Suggested Next Step

If this side branch continues, prioritize failure-sample taxonomy and final report consolidation. Do not treat the S39/S40 result as unsupervised weak-form inversion success. For improving weak 200x100 samples, prioritize stronger false-positive suppression or a structural modeling change over more radius / centers / area-prior scans.
