# Evaluation report

- Backbone: CLIP ViT-B/32 (frozen), classifier: logistic regression, C = 100.0
- Cross-validation on train (5-fold): 94.7% ± 3.4%
- **Test accuracy: 44 / 45 correct (97.8%)**

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Food & Drinks | 0.938 | 1.000 | 0.968 | 15 |
| People | 1.000 | 1.000 | 1.000 | 15 |
| Interior | 1.000 | 0.933 | 0.966 | 15 |

## Confusion matrix

Rows are the true class, columns the predicted class.

| true \ predicted | Food & Drinks | People | Interior |
|---|---|---|---|
| Food & Drinks | 15 | 0 | 0 |
| People | 0 | 15 | 0 |
| Interior | 1 | 0 | 14 |

![Confusion matrix](confusion_matrix.png)

## Misclassified images

| File | True | Predicted | Probabilities |
|---|---|---|---|
| `data/test/interior/337d124eeda7482b9aba84aae4ef9528.jpg` | interior | food_drinks | food_drinks 0.92, people 0.00, interior 0.08 |

## Lowest probability on the true class

| File | True | Predicted | Probabilities |
|---|---|---|---|
| `data/test/interior/337d124eeda7482b9aba84aae4ef9528.jpg` | interior | food_drinks | food_drinks 0.92, people 0.00, interior 0.08 |
| `data/test/people/4eda754771ad4d60ba25939c7b1efa9f.jpg` | people | people | food_drinks 0.01, people 0.69, interior 0.30 |
| `data/test/food_drinks/ef69ad42af5f4eedb1a9387f6797d0d2.jpg` | food_drinks | food_drinks | food_drinks 0.77, people 0.01, interior 0.22 |
| `data/test/food_drinks/cceb51f7e99641cba8a0bd8a96195148.jpg` | food_drinks | food_drinks | food_drinks 0.87, people 0.04, interior 0.09 |
| `data/test/people/ef9b37c3b3084d9bba978a5e44aafbe9.jpg` | people | people | food_drinks 0.00, people 0.89, interior 0.11 |

## Most difficult example

`data/test/interior/337d124eeda7482b9aba84aae4ef9528.jpg`: true **interior**, predicted **food_drinks** (food_drinks 0.92, people 0.00, interior 0.08)

![Difficult example](difficult_example.jpg)
