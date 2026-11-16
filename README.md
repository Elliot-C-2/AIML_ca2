# Customer Segmentation & Utility Demand Forecasting

Individual coursework project for the AI & Machine Learning module (ST1511), Diploma in Applied AI & Analytics, Singapore Polytechnic (2026).

The project has two parts:

- **Part A:** customer segmentation with unsupervised learning (K-Means and Agglomerative Clustering).
- **Part B:** forecasting monthly gas, electricity and water demand with time-series models (Holt-Winters and SARIMA).

Both parts follow the same workflow: **EDA → preprocessing → baseline → tuning → evaluation → insights.**

---

## Part A: Customer Segmentation

**Goal:** Group 200 customers by their gender, age, income and spending, then work out what each group is like and which groups the business should focus on.

### Approach

- **Feature selection:**
  - EDA showed early signs of clusters in income against spending.
  - Spending was similar across genders, so gender was dropped.
  - Adding age lowered the silhouette score from 0.555 to 0.409, so the final features are **income and spending**, scaled with `StandardScaler`.
- **Choosing k:**
  - The elbow method and the silhouette score both pointed to **k = 5**.
- **Two algorithms:**
  - K-Means (centroid-based) and Agglomerative Clustering (hierarchical, Ward linkage) produced almost identical clusters.
  - Their silhouette scores were also almost identical: **0.555** and **0.554**.
  - Two different methods agreeing suggests the clusters are real patterns, not something one method invented.

### The five customer segments

| Cluster | Age | Income | Spending | Customers | Profile |
|---|---|---|---|---|---|
| 0 | ~43 | ~55k | ~50 | 81 | Average customers with balanced income and spending |
| 1 | ~33 | ~87k | ~82 | 39 | Young professionals: high earners who enjoy spending |
| 2 | ~25 | ~26k | ~79 | 22 | Young spenders: low income but high spending |
| 3 | ~41 | ~88k | ~17 | 35 | Wealthy savers: high income but very little spending |
| 4 | ~45 | ~26k | ~21 | 23 | Budget-conscious: low income, careful spending |

### Key finding

**Life stage affects spending more than income does.**

- Younger customers (25–33) spend a lot regardless of income.
- Older customers (41–45) spend little, even when their income is high.

### Recommendations

- **Wealthy savers (Cluster 3):** they have the most untapped potential. Target them with quality and value messaging, and with premium, long-lasting products rather than impulse deals.
- **Young spenders (Cluster 2):** they're valuable but at risk. Keep them with loyalty programmes, instalment plans and social media marketing. As their income grows, they could become Cluster 1 customers.

---

## Part B: Utility Demand Forecasting

**Goal:** Forecast monthly gas, electricity and water consumption.
**Data:** 408 months from 1990 to 2023, with no missing values.

### EDA

- **Electricity** has a clear upward trend that slows after 2015. It's the most predictable of the three.
- **Gas** is cyclical with no clear direction, and has an anomalous spike around 2014.
- **Water** is the most volatile, with sharp drops around 2004 and 2014.
- All three have a **12-month seasonal pattern**, which is why seasonal models (Holt-Winters and SARIMA) were used.

### Approach

1. **Chronological split:** 70% train, 15% validation, 15% test. A random split would leak future data into training.
2. **Baselines:**
   - Holt-Winters with additive trend and seasonality.
   - SARIMA(1,1,1)(1,1,1,12).
3. **Tuning on the validation set:**
   - Holt-Winters: every combination of trend (add, mul, none) and seasonality (add, mul).
   - SARIMA: a manual grid search over p, q, P and Q, plus `auto_arima`.
4. **Final evaluation:** the best configurations were retrained on train + validation, then tested on data the models had never seen.

### Results

**Holt-Winters beat SARIMA for all three variables.** The simpler model captured the straightforward seasonal patterns better, and SARIMA's extra parameters didn't add accuracy.

| Variable | Trend | Seasonal | Test MAPE |
|---|---|---|---|
| Electricity | none | mul | **3.20%** |
| Gas | mul | add | 11.50% |
| Water | none | mul | 17.55% |

- **Why no trend for electricity and water:** their trends have levelled off recently, so a trend term would project growth that no longer exists.
- **Why multiplicative seasonality:** as electricity rose from about 600 to over 1,000 MWh, its seasonal swings grew in proportion.
- **Manual grid search vs auto_arima:** the manual search won for electricity and water. It optimises MAPE (forecast accuracy), whereas `auto_arima` optimises AIC (fit to the training data).

### 60-month forecast and recommendations

The final models were retrained on 100% of the data to forecast 5 years ahead, with 95% confidence intervals.

- **Electricity:** the most reliable forecast (3.20% MAPE). It's suitable for budget planning and allocating resources.
- **Gas:** moderately reliable (11.50%). Use it for planning, but leave a buffer.
- **Water:** the least reliable (17.55%). Its confidence interval goes negative towards the end of the forecast, which is physically impossible and shows the limits of forecasting volatile data.

**Limitations:**
- The forecasts assume historical patterns will continue.
- They don't account for outside factors like policy, the economy or climate.
- They can't predict one-off anomalies like the 2014 gas spike.

---

## Repository structure

```
p2518750_CA2_PartA_ElliotChang.ipynb   Customer segmentation
p2518750_CA2_PartB_ElliotChang.ipynb   Utility demand forecasting
CA2_Dataset/
    CA2-Customer-Data.csv
    CA2-Energy-Consumption-Data.csv
```

## Running it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels pmdarima
```

Open either notebook in Jupyter and run the cells in order.

## Tools

Python · pandas · scikit-learn (K-Means, Agglomerative Clustering) · SciPy · statsmodels (Holt-Winters, SARIMA) · pmdarima · matplotlib · seaborn
