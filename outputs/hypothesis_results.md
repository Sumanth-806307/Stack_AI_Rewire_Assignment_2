# Hypothesis Test Results

**Significant at α=0.05:** 5/9

| Driver | Test | Statistic | p-value | Significant? | Effect Size | Direction |
|--------|------|-----------|---------|--------------|-------------|-----------|
| Total spend (monetary, skewed) | Mann-Whitney U | 61626.5 | 8.8e-05 | Yes | 0.3627 | lower |
| Transaction count (discrete, skewed) | Mann-Whitney U | 62236.5 | 0.000153 | Yes | 0.3591 | lower |
| Distinct categories (bounded ordinal) | Mann-Whitney U | 64002.0 | 0.000582 | Yes | 0.3394 | lower |
| Avg transaction value (ratio, skewed) | Mann-Whitney U | 72797.5 | 0.183725 | No | 0.0053 | lower |
| Return count (zero-inflated) | Mann-Whitney U | 73377.0 | 0.147038 | No | 0.1368 | lower |
| Return rate per customer (zero-inflated proportion) | Mann-Whitney U | 110279.0 | 9e-06 | Yes | 0.1767 | lower |
| Acquisition channel (categorical, 4 levels) | Chi-square | 5.8387 | 0.119725 | No | 0.0624 | highest_rate: organic |
| Region (categorical, 4 levels) | Chi-square | 3.7571 | 0.288915 | No | 0.05 | highest_rate: east |
| Signup year (categorical, 3 cohorts) | Chi-square | 70.7192 | 0.0 | Yes | 0.2171 | highest_rate: cohort 2026 |

### Non-significant drivers
- avg_txn_value
- return_count
- acquisition_channel
- region
