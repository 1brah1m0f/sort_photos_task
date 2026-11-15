# Evaluation report

- Backbone: CLIP ViT-B/32 (frozen), classifier: logistic regression, C = 1.0, class_weight = balanced
- Model selection: 5-fold CV (grouped by venue/source, 233 groups) on train, balanced_accuracy 92.1% ± 3.1%
- **Test accuracy: 44 / 45 correct (97.8%)**

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Food & Drinks | 1.000 | 1.000 | 1.000 | 17 |
| People | 0.947 | 1.000 | 0.973 | 18 |
| Interior | 1.000 | 0.900 | 0.947 | 10 |

## Confusion matrix

Rows are the true class, columns the predicted class.

| true \ predicted | Food & Drinks | People | Interior |
|---|---|---|---|
| Food & Drinks | 17 | 0 | 0 |
| People | 0 | 18 | 0 |
| Interior | 0 | 1 | 9 |

![Confusion matrix](confusion_matrix.png)

## Misclassified images

| File | True | Predicted | Probabilities |
|---|---|---|---|
| `data/test/interior/0ae7783d3e9d4de3b6246e9b9d2fce0a.jpg` | interior | people | food_drinks 0.12, people 0.58, interior 0.29 |

## Lowest probability on the true class

| File | True | Predicted | Probabilities |
|---|---|---|---|
| `data/test/interior/0ae7783d3e9d4de3b6246e9b9d2fce0a.jpg` | interior | people | food_drinks 0.12, people 0.58, interior 0.29 |
| `data/test/interior/bcfaa5e59734468c9f69462475a2d9af.jpg` | interior | interior | food_drinks 0.18, people 0.34, interior 0.48 |
| `data/test/interior/5432e6aea9e543e5a805bf90d9887feb.jpg` | interior | interior | food_drinks 0.12, people 0.35, interior 0.52 |
| `data/test/interior/cc6a9af6cee042e49a2dd716ab143711.jpg` | interior | interior | food_drinks 0.21, people 0.25, interior 0.53 |
| `data/test/people/2fb066309ba54573a5d8f03d4738f692.jpg` | people | people | food_drinks 0.10, people 0.55, interior 0.35 |

## Most difficult example

`data/test/interior/0ae7783d3e9d4de3b6246e9b9d2fce0a.jpg`: true **interior**, predicted **people** (food_drinks 0.12, people 0.58, interior 0.29)

![Difficult example](difficult_example.jpg)
