# 🐉 Wide & Deep Recommender System

**Predict user ratings and generate personalized anime recommendations**
Based on Cheng et al.’s “Wide & Deep Learning for Recommender Systems” (Google, 2016) using the MyAnimeList 2023 dataset.

---

## 📂 Repository Structure

* **README.md**
* **Wide\_and\_Deep\_Recommendation.ipynb** — main notebook
* **data/**

  * anime-dataset-2023.csv
  * users-details-2023.csv
  * users-score-2023.csv
* **models/**

  * wide\_and\_deep\_model.pth — saved PyTorch weights (optional)

---

## 1. Project Overview

* **Goal**
  Build a hybrid recommender that *memorizes* popular patterns (wide) and *generalizes* via embeddings (deep) to predict 1–10 user ratings and deliver top-N anime suggestions.
* **Key Paper**
  Cheng et al., *Wide & Deep Learning for Recommender Systems* (2016)

  > Combines a linear “wide” model (memorization) with a neural “deep” model (generalization) in one architecture.
* **Dataset**

  * *anime-dataset-2023.csv*: Anime metadata (ID, title, score, genres, rank, popularity, …)
  * *users-details-2023.csv*: User profiles (MAL ID, days watched, mean score, completed, episodes watched, …)
  * *users-score-2023.csv*: Explicit ratings given by users (1–10 scale)

---

## 2. Data Preparation & Pre-processing

1. **Load & Rename**
   Standardize join columns: `user_id`, `anime_id`, `rating`, `Anime_Title`.
2. **Rank Imputation**
   Fit a linear regression (Popularity → Rank) to predict missing Rank values.
3. **Merge Tables**
   Join ratings ⨝ anime ⨝ users → one master DataFrame of interactions plus side-info.
4. **Sampling**
   Randomly sample **100 000** rows (seed=42) for RAM/GPU efficiency.
5. **Wide Features** *(memorization)*

   * Numeric: Score, Popularity, Rank → replace unknowns, mean-impute, MinMax scale
   * Categorical: Genres → fill missing with “Unknown”, one-hot encode
6. **Deep Features** *(generalization)*

   * IDs: user\_id, anime\_id → LabelEncode → Embedding (32 dims each)
7. **Target**
   Ensure rating is numeric (NaN → 0), float32.
8. **Train/Test Split**
   80 % train / 20 % test (random\_state=42), preserving alignment with wide features.

---

## 3. Model Architecture

* **Wide component**: Linear layer on numeric + one-hot features to *memorize* high-signal patterns.
* **Deep component**:

  * Embeds each user and each anime into 32-dim vectors.
  * Concatenates them (64 dims) and feeds through two ReLU-activated hidden layers (128 → 64).
  * Outputs a single score.
* **Joint output**: Sum of wide output + deep output.
* **Loss**: Mean Squared Error between predicted vs. true rating.
* **Optimizer**: Adam (learning rate 1 e-3).
* **Batch size**: 64, **Epochs**: 20.

---

## 4. Training & Evaluation

* **Training loop**

  1. Forward pass → wide + deep branch → prediction
  2. Compute MSE loss
  3. Backpropagate & update both sets of weights
* **Validation**
  Compute RMSE on the held-out test set
* **Typical Results**

  * Train RMSE ≈ 1.50
  * Test  RMSE ≈ 1.70

---

## 5. Generating Top-N Recommendations

* To recommend for a given user:

  1. Score **all** anime for that user in one batch (wide features reused, user ID repeated).
  2. Use `torch.topk` to pick the highest-scoring N titles.
* Personalization comes from the user embedding vector interacting with each anime embedding.

---

## 6. How to Use

1. **Clone** this repo and place the three CSVs in `/data`.
2. **Install** dependencies (Python 3.10+):

   ```bash
   pip install pandas numpy scikit-learn torch
   ```
3. **Open** `Wide_and_Deep_Recommendation.ipynb` and run all cells in order.
4. **Inspect** training logs for per-epoch loss and final test RMSE.
5. **Call** the recommendation function with your desired user ID:

   ```python
   recommend_top_n(model, user_id=42, n=10)
   ```

---

## 7. Extensions & Future Work

* Add manual cross features (e.g. `user_country × genre`) to the wide part
* Introduce BatchNorm / Dropout in the deep MLP for regularization
* Swap MSE loss for pairwise ranking (BPR) on implicit feedback
* Deploy as a REST API; cache top-K per user for sub-20 ms inference
* Incorporate user demographic embeddings to handle cold-start users

---

## 8. References & Citation

Cheng, Heng-Tze, et al. “Wide & Deep Learning for Recommender Systems.” *arXiv* (2016).
arXiv:1606.07792

BibTeX:

```bibtex
@article{Cheng2016WideDeep,
  title   = {Wide \& Deep Learning for Recommender Systems},
  author  = {Cheng, Heng-Tze and Koc, Levent and Harmsen, et al.},
  journal = {arXiv preprint arXiv:1606.07792},
  year    = {2016}
}
```

---

## 9. License

Released under the **MIT License** – feel free to fork, modify, and share. 🎉
