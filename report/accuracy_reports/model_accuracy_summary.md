# Smart Grinding Model Accuracy Report

## MAE Performance (Test Set)

| Model Input Type | Mean MAE | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| all | 6.0821 | 2.2279 | 2.1035 | 11.0894 |
| ae_spec+vib_spec | 6.5291 | 2.2890 | 2.2537 | 12.3091 |
| ae_spec+ae_features+vib_spec+vib_features | 6.6734 | 2.5525 | 2.4615 | 14.0621 |
| ae_spec+ae_features | 8.6988 | 2.8339 | 2.7581 | 14.3162 |
| vib_spec+vib_features | 8.9272 | 2.6270 | 2.0235 | 13.5875 |
| ae_spec | 9.0676 | 2.3597 | 3.9071 | 14.1558 |
| vib_spec | 9.1044 | 2.3068 | 3.8106 | 13.0523 |
| vib_features+pp | 11.5940 | 1.1074 | 9.2006 | 14.2606 |
| ae_features+pp | 11.6038 | 1.1032 | 9.4662 | 14.1747 |
| ae_features+vib_features+pp | 11.6039 | 1.1042 | 9.4086 | 14.2959 |
| ae_features+vib_features | 11.9669 | 1.1003 | 9.7081 | 14.6056 |
| vib_features | 11.9706 | 1.1008 | 9.7103 | 14.5986 |
| ae_features | 11.9716 | 1.1017 | 9.7016 | 14.6074 |

**Best Performing Model (Lowest MAE):** `all` with MAE = 6.0821

## MSE Performance (Test Set)

| Model Input Type | Mean MSE | Std Dev |
| :--- | :--- | :--- |
| all | 900.4121 | 424.9893 |
| ae_spec+vib_spec | 961.4703 | 439.8245 |
| ae_spec+ae_features+vib_spec+vib_features | 993.4607 | 488.0358 |
| ae_spec+ae_features | 1328.7397 | 609.2637 |
| vib_spec+vib_features | 1355.4154 | 588.3655 |
| ae_spec | 1369.8383 | 566.2430 |
| vib_spec | 1372.7374 | 521.8657 |
| vib_features+pp | 1846.4936 | 455.4925 |
| ae_features+pp | 1848.5718 | 454.5762 |
| ae_features+vib_features+pp | 1848.6708 | 455.1211 |
| ae_features+vib_features | 1934.2188 | 462.3699 |
| vib_features | 1935.1280 | 462.5477 |
| ae_features | 1935.3791 | 462.7491 |
