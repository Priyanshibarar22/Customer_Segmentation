#  Customer Segmentation — Mall Customers Dataset

### Unsupervised Machine Learning | K-Means & Hierarchical Clustering


##  Project Overview

Customer segmentation is the process of dividing customers into meaningful groups based on shared characteristics. Unlike classification, this is an **unsupervised learning** problem — there are no pre-defined labels. The algorithm must discover hidden structure in the data on its own.

This project applies two clustering algorithms — **K-Means** and **Agglomerative Hierarchical Clustering** — to segment 200 mall customers based on their **Annual Income** and **Spending Score**. The goal is to identify distinct customer personas that a marketing team can act on with targeted strategies.

---

##  Business Problem

**Question:** Can we group mall customers into meaningful segments based on their income and spending behaviour — without any labelled data?

**Why it matters:**
- Generic marketing campaigns waste budget reaching uninterested customers
- Different customer segments respond to completely different offers and messaging
- Understanding who your customers are (not just in aggregate) drives smarter business decisions
- Segmentation enables personalised experiences at scale

**Type of problem:** Unsupervised Learning — Clustering

---

##  Dataset

| Property | Details |
|---|---|
| **File** | `Mall_Customers.csv` |
| **Rows** | 200 customers |
| **Columns** | 5 features |
| **Missing Values** | None |
| **Duplicates** | None |
| **Memory Usage** | 7.9+ KB |

### Feature Description

| Column | Type | Description |
|---|---|---|
| `CustomerID` | int64 | Unique customer identifier (dropped before modelling) |
| `Genre` | object | Customer gender — Male / Female |
| `Age` | int64 | Customer age in years (18–70) |
| `Annual Income (k$)` | int64 | Annual income in thousands of dollars ($15k–$137k) |
| `Spending Score (1-100)` | int64 | Mall-assigned spending score from 1 (lowest) to 100 (highest) |

### Descriptive Statistics

| Feature | Mean | Std | Min | Max |
|---|---|---|---|---|
| Age | 38.85 | 13.97 | 18 | 70 |
| Annual Income (k$) | 60.56 | 26.26 | 15 | 137 |
| Spending Score | 50.20 | 25.82 | 1 | 99 |

---

##  Tech Stack

| Category | Library / Tool |
|---|---|
| Language | Python 3.8+ |
| Environment | Google Colab |
| Data Manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| K-Means Clustering | `sklearn.cluster.KMeans` |
| Hierarchical Clustering | `sklearn.cluster.AgglomerativeClustering` |
| Dendrogram | `scipy.cluster.hierarchy` |
| Cluster Validation | `sklearn.metrics.silhouette_score` |
| Scaling | `sklearn.preprocessing.StandardScaler` |

---

##  Project Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│                      PROJECT PIPELINE                          │
│                                                                │
│  1. Data Loading & Inspection   →  print, head, tail, info     │
│  2. EDA                         →  Age dist, Income vs Score   │
│  3. Feature Selection           →  Annual Income + Spending    │
│  4. Standard Scaling            →  StandardScaler              │
│  5. Optimal K — Elbow Method    →  WCSS for k=1 to 10          │
│  6. Optimal K — Silhouette      →  Scores for k=2 to 10        │
│  7. K-Means Clustering          →  k=5, k-means++              │
│  8. Hierarchical Clustering     →  Dendrogram → Ward, k=5      │
│  9. Cluster Profiling           →  groupby().mean()            │
│ 10. Comparison                  →  Silhouette score both       │
└────────────────────────────────────────────────────────────────┘
```

---

## Exploratory Data Analysis

### Age Distribution
The age histogram shows a **right-skewed distribution** with the majority of customers between 18–40 years old, peaking in the late 20s to early 30s. The mall primarily serves a young adult demographic. Older customers (50+) are present but less frequent.

### Income vs Spending Score (Scatter Plot)
The scatter plot of `Annual Income` vs `Spending Score` reveals **five visually distinct clusters** even before any algorithm is applied:
- Low income, high spenders (bottom-left, high score)
- High income, high spenders (top-right quadrant)
- High income, low spenders (bottom-right quadrant)
- Low income, low spenders (bottom-left, low score)
- Average income, average spenders (dense centre cluster)

This visual confirms that clustering on these two features is well-motivated.

---

## Preprocessing

### Feature Selection

`CustomerID` is excluded — it is a row label with no analytical value. `Genre` and `Age` are also excluded for this analysis, keeping the focus on income and spending behaviour — the two most business-relevant dimensions for mall marketing.

### Standard Scaling

`Annual Income` ranges from $15k–$137k while `Spending Score` ranges from 1–100. Without scaling, income would dominate the distance calculations in both K-Means and Hierarchical clustering purely because its numeric range is larger. StandardScaler brings both features to mean=0, std=1.

---

##  Finding Optimal Clusters

### Elbow Method (WCSS)
The Within-Cluster Sum of Squares (WCSS) is computed for k=1 to 10. The "elbow" — the point where WCSS stops dropping steeply — indicates the optimal number of clusters. The elbow is clearly visible at **k=5**, where the curve bends and the rate of improvement slows significantly.

### Silhouette Score Analysis

| Clusters (k) | Silhouette Score |
|---|---|
| 2 | 0.397 |
| 3 | 0.467 |
| 4 | 0.494 |
| **5** | **0.555**  |
| 6 | 0.514 |
| 7 | 0.502 |
| 8 | 0.455 |
| 9 | 0.457 |
| 10 | 0.445 |

**k=5 produces the highest silhouette score (0.555)** — confirming the Elbow Method finding. Both methods independently agree: **5 is the optimal number of clusters**.

> A silhouette score ranges from -1 to 1. Scores above 0.5 indicate well-separated, cohesive clusters. A score of 0.555 is a strong result for real-world customer data.

---

##  K-Means Clustering



### Cluster Size Distribution

| Cluster | Count | % of Customers |
|---|---|---|
| 0 | 81 | 40.5% |
| 1 | 39 | 19.5% |
| 2 | 22 | 11.0% |
| 3 | 35 | 17.5% |
| 4 | 23 | 11.5% |

### Cluster Profiles (K-Means)

| Cluster | Avg Age | Avg Income (k$) | Avg Spending Score | Persona |
|---|---|---|---|---|
| **0** | 42.7 | 55.3 | 49.5 |  Average People |
| **1** | 32.7 | 86.5 | 82.1 |  High Value Targets |
| **2** | 25.3 | 25.7 | 79.4 |  Young Free Spenders |
| **3** | 41.1 | 88.2 | 17.1 |  Careful High Earners |
| **4** | 45.2 | 26.3 | 20.9 |  Budget Conscious |

**`init='k-means++'`** is used instead of random initialisation — this smart seeding strategy places initial centroids far apart, leading to faster convergence and more stable, reproducible results.

---

##  Hierarchical Clustering

### Dendrogram

The dendrogram uses **Ward linkage** — which minimises the total within-cluster variance at each merge step. By examining where the longest vertical lines (largest Euclidean distances) appear, the dendrogram confirms **5 natural groupings** in the data — consistent with K-Means and the silhouette analysis.


### HC Cluster Size Distribution

| HC Cluster | Count |
|---|---|
| 2 | 85 |
| 1 | 39 |
| 0 | 32 |
| 4 | 23 |
| 3 | 21 |

### HC Cluster Profiles

| HC Cluster | Avg Age | Avg Income (k$) | Avg Spending Score | Persona |
|---|---|---|---|---|
| **0** | 41.0 | 89.4 | 15.6 | Careful High Earners |
| **1** | 32.7 | 86.5 | 82.1 | High Value Targets |
| **2** | 42.5 | 55.8 | 49.1 | Average People |
| **3** | 25.3 | 25.1 | 80.0 | Young Free Spenders |
| **4** | 45.2 | 26.3 | 20.9 | Budget Conscious |

The HC cluster profiles are **nearly identical to K-Means** — both algorithms independently discovered the same five customer personas, which strongly validates the segmentation.

---

##  Cluster Profiles & Business Insights

### The 5 Customer Personas

####  Cluster 1 — "High Value Targets" (39 customers)
- **Profile:** Young (avg 33), high income (~$87k), very high spending score (~82)
- **Who they are:** Young professionals with disposable income who love to spend
- **Strategy:** VIP loyalty programmes, early access to new products, premium brand partnerships, exclusive events

####  Cluster 2/3 — "Young Free Spenders" (35 customers)
- **Profile:** Very young (avg 25), low income (~$26k), high spending score (~80)
- **Who they are:** Young shoppers who spend beyond their income — possibly students or young earners who prioritise experiences
- **Strategy:** Instalment payment options, trendy/affordable fashion, social media campaigns, BOGO offers

####  Cluster 3/0 — "Careful High Earners" (23–32 customers)
- **Profile:** Middle-aged (avg 41), high income (~$88k), very low spending score (~17)
- **Who they are:** Affluent customers who are conservative spenders — possibly saving or investing
- **Strategy:** Premium quality messaging, value-for-money positioning, exclusive high-end products, personalised outreach to shift perception of mall value

####  Cluster 4 — "Budget Conscious" (23 customers)
- **Profile:** Older (avg 45), low income (~$26k), low spending score (~21)
- **Who they are:** Price-sensitive, infrequent shoppers with limited discretionary spend
- **Strategy:** Discount events, clearance sales, loyalty point multipliers, essential goods focus

####  Cluster 0/2 — "Average People" (81–85 customers — largest group)
- **Profile:** Middle-aged (avg 43), average income (~$55k), average spending (~49)
- **Who they are:** The mall's core, steady customer base — moderate in every dimension
- **Strategy:** General promotions, seasonal campaigns, family-oriented offers, weekend deals

---

##  Model Comparison

| Metric | K-Means | Hierarchical Clustering |
|---|---|---|
| Algorithm Type | Partitioning (centroid-based) | Agglomerative (bottom-up) |
| Optimal k | 5 | 5 |
| **Silhouette Score** | **0.5547**  | 0.5538 |
| Cluster Personas | 5 distinct groups | 5 distinct groups (same) |
| Scalability | High (works on large datasets) | Low (memory-intensive for large n) |
| Requires k upfront | Yes | Yes (after dendrogram) |
| Interpretability | Centroids easy to interpret | Dendrogram gives full merge history |

> **K-Means marginally outperforms Hierarchical Clustering** (0.5547 vs 0.5538) — both excellent. The fact that both algorithms independently arrive at the **same 5 customer personas** is strong validation that these segments genuinely exist in the data.

---

##  Future Implementation

### 1.  Multi-Dimensional Clustering
- Include `Age` and `Genre` as additional features for richer, more nuanced segmentation
- Apply **PCA (Principal Component Analysis)** to reduce dimensionality before clustering when working with more features
- Explore **t-SNE or UMAP** for 2D visualisation of high-dimensional clusters

### 2.  Advanced Clustering Algorithms
- **DBSCAN** — density-based clustering that can detect arbitrarily shaped clusters and automatically identify outliers/noise points
- **Gaussian Mixture Models (GMM)** — soft clustering that assigns probability scores of belonging to each cluster rather than hard assignments
- **Mean Shift** — parameter-free clustering that finds cluster centres as density peaks

### 3. Cluster Validation & Stability
- **Davies-Bouldin Index** — additional internal validation metric (lower = better)
- **Calinski-Harabasz Score** — ratio of between-cluster to within-cluster dispersion
- **Bootstrap stability analysis** — test whether clusters remain stable across random subsamples of the data

### 4.  Deployment & Integration
- **Streamlit Dashboard** — interactive tool where marketing teams can explore cluster profiles, filter by segment, and export customer lists
- **Real-time scoring API** — Flask/FastAPI endpoint that assigns a new customer to a cluster based on their income and spending data
- **CRM Integration** — push cluster labels directly into tools like Salesforce or HubSpot to enable automated segment-targeted campaigns

### 5.  Business Intelligence
- **RFM Analysis** — extend segmentation with Recency, Frequency, Monetary value data for e-commerce/retail contexts
- **Time-series clustering** — track how customers move between segments over time (customer lifecycle analysis)
- **A/B Testing** — measure the real-world revenue impact of segment-specific marketing strategies vs. generic campaigns
- **CLV Integration** — weight clusters by Customer Lifetime Value to prioritise retention investment

### 6.  Continuous Learning
- **Automated retraining** — re-cluster quarterly as new customer data arrives; monitor for segment drift
- **Anomaly detection** — flag customers whose behaviour suddenly shifts across segments
- **Cohort analysis** — track newly acquired customers to see which segments they naturally fall into

---

