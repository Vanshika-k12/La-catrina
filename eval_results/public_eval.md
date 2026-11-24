# Public-dataset evaluation (Fashion Product Images, small)

Sample: 900 images, random (seed 0), up to 100 per category. Images are 60x80 px product photos on white backgrounds. No training: CLIP is used zero-shot.

Average processing time: 0.34 s per image on CPU (CLIP + colours).

## 1. Category accuracy (CLIP zero-shot)

| Prompt design | Accuracy | 95% CI |
|---|---|---|
| generic: one prompt 'a photo of a {category}' | 796/900 = 88.4% | 86.2% - 90.4% |
| our ensemble, before tank/camisole | 788/900 = 87.6% | 85.2% - 89.6% |
| our ensemble, current (used by the app) | 785/900 = 87.2% | 84.9% - 89.2% |

### Per-class results (current prompts)

```
              precision    recall  f1-score   support

       shirt       0.81      1.00      0.89       100
     T-shirt       0.91      0.83      0.87       100
       jeans       0.90      0.63      0.74       100
    trousers       0.71      0.98      0.82       100
       skirt       0.93      0.71      0.81       100
       dress       0.83      0.90      0.87       100
      jacket       1.00      0.82      0.90       100
    footwear       1.00      0.99      0.99       100
   accessory       0.88      0.99      0.93       100

    accuracy                           0.87       900
   macro avg       0.89      0.87      0.87       900
weighted avg       0.89      0.87      0.87       900

```

### Confusion matrix (rows = dataset label, columns = prediction)

```
                pred:shirt  pred:T-shirt  pred:jeans  pred:trousers  pred:skirt  pred:dress  pred:jacket  pred:footwear  pred:accessory
true:shirt             100             0           0              0           0           0            0              0               0
true:T-shirt            14            83           0              0           0           3            0              0               0
true:jeans               1             0          63             35           0           0            0              0               1
true:trousers            0             0           1             98           0           1            0              0               0
true:skirt               0             3           5              5          71          11            0              0               5
true:dress               0             3           0              0           5          90            0              0               2
true:jacket              9             2           0              0           0           3           82              0               4
true:footwear            0             0           0              0           0           0            0             99               1
true:accessory           0             0           1              0           0           0            0              0              99
```

## 2. Main colour (K-Means + palette naming)

869 of 900 images have a dataset colour with an exact palette equivalent (31 excluded, e.g. Multi, Silver, Gold, Peach).

| Metric | Result | 95% CI |
|---|---|---|
| Exact palette name | 190/869 = 21.9% | 19.2% - 24.7% |
| Same colour family | 318/869 = 36.6% | 33.5% - 39.8% |

Most common colour confusions (dataset -> predicted):

- black -> charcoal: 108
- blue -> charcoal: 45
- white -> light grey: 38
- blue -> grey: 35
- black -> grey: 33
- navy -> charcoal: 31
- blue -> denim: 30
- blue -> light grey: 22

## Caveats

- Product photos (clean white background, centred) differ from phone photos of clothes on a bed; colour is easier here, category may be harder at 60x80 px.
- Dataset labels were mapped to our 9 categories; types with no clear match (kurtas, sarees, shorts, innerwear, cosmetics...) were excluded.
- The dataset's colour label describes the product overall; multi-coloured items can legitimately have a different 'main' colour.
