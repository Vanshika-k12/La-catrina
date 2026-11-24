# Colour-naming calibration (development / test split)

Images with an exact palette colour label: 869. Stratified 50/50 split (seed 0): **434 development** images used to learn reference colours, **435 test** images never seen during calibration.

Calibrated 19 colours (>= 3 development images each); the others keep the ideal swatch.

## Held-out test results

| Naming reference | Exact name | 95% CI | Same family | 95% CI |
|---|---|---|---|---|
| ideal swatches (before) | 88/435 = 20.2% | 16.7% - 24.3% | 158/435 = 36.3% | 31.9% - 40.9% |
| calibrated (after) | 171/435 = 39.3% | 34.8% - 44.0% | 243/435 = 55.9% | 51.2% - 60.5% |

Exact McNemar test (paired, same test images): exact-name p = 3.5e-12 (115 images fixed, 32 broken); family p = 8.4e-12.

## Per-colour exact accuracy on the test split (colours with >= 10 test images)

| Colour | Test images | Before | After |
|---|---|---|---|
| black | 110 | 25% | 49% |
| blue | 84 | 2% | 25% |
| white | 35 | 29% | 60% |
| grey | 31 | 26% | 23% |
| red | 27 | 41% | 59% |
| navy | 26 | 4% | 46% |
| brown | 22 | 18% | 27% |
| pink | 22 | 41% | 32% |
| green | 12 | 17% | 25% |
| purple | 12 | 8% | 33% |

## Learned reference colours (CIELAB)

| Colour | Dev images | Ideal swatch L, a, b | Calibrated L, a, b |
|---|---|---|---|
| black | 110 | [6.3, 0.0, -0.0] | [25.41, -0.1, 0.09] |
| blue | 83 | [41.1, 23.8, -61.9] | [48.25, 0.02, -12.61] |
| white | 36 | [96.5, 0.0, -0.0] | [83.5, 1.36, 2.46] |
| grey | 32 | [53.6, 0.0, -0.0] | [49.31, -0.76, -0.04] |
| red | 27 | [43.3, 63.3, 40.0] | [44.97, 43.31, 18.16] |
| navy | 26 | [15.4, 12.3, -29.0] | [24.57, 2.21, -4.58] |
| pink | 22 | [72.0, 37.6, -1.3] | [67.66, 28.42, 5.94] |
| brown | 21 | [35.7, 13.6, 24.6] | [35.88, 5.55, 9.38] |
| green | 13 | [51.3, -45.9, 34.2] | [72.28, -13.63, 12.52] |
| purple | 11 | [32.9, 42.2, -39.1] | [52.62, 14.12, -10.14] |
| cream | 8 | [93.2, -0.7, 13.3] | [84.81, 3.73, 13.85] |
| olive | 8 | [39.3, -7.3, 25.6] | [36.84, -0.67, 10.12] |
| beige | 7 | [81.4, 1.6, 20.1] | [75.92, 1.87, 13.27] |
| yellow | 6 | [84.4, -5.2, 78.8] | [83.09, -1.15, 35.2] |
| charcoal | 5 | [23.2, 1.2, -3.1] | [30.7, -2.88, -0.12] |
| khaki | 5 | [71.5, -2.7, 30.2] | [72.44, 1.33, 15.94] |
| orange | 5 | [65.4, 36.3, 66.1] | [55.52, 30.89, 32.24] |
| maroon | 3 | [23.5, 39.3, 15.4] | [41.72, 14.37, 0.45] |
| tan | 3 | [63.4, 10.8, 33.0] | [55.77, 5.51, 21.18] |

## Transfer to the user's phone photos (domain-shift check)

The same calibration applied to the 10 labelled phone photos in `labels.csv` (clothes photographed at home, not product shots):

| Naming reference | Exact name |
|---|---|
| ideal swatches (before) | 4/10 |
| calibrated on product photos (after) | 2/10 |

Predictions that changed:

- jacket.jpeg (label grey): charcoal -> brown
- shoes1.jpeg (label beige): brown -> tan
- top1.jpeg (label maroon): maroon -> brown
- trousers.jpeg (label light grey): light grey -> white

**Decision:** the calibration does not transfer to phone photos (domain shift: studio lighting vs room lighting), so the app keeps the ideal swatches. A calibration for the app should be learned from labelled phone photos with its own dev / test split.
Only 10 phone photos, so this check is indicative.
