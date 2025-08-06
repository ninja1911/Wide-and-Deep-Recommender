# 🐉 Wide & Deep Recommender System  
**Predict user ratings and generate personalized anime recommendations**  
Based on Cheng et al.’s “Wide & Deep Learning for Recommender Systems” (Google, 2016) using the MyAnimeList 2023 dataset.

---

## 📂 Repository Structure  
- **README.md**  
- **Wide_and_Deep_Recommendation.ipynb** — main notebook  
- **data/**  
  - `anime-dataset-2023.csv`  
  - `users-details-2023.csv`  
  - `users-score-2023.csv`  

---

## 1. Project Overview  
- **Goal**  
  Build a hybrid recommender that combines **memorization** (wide) of obvious popularity and ranking signals with **generalization** (deep) via learned embeddings, in order to predict user ratings on a 1–10 scale and deliver top-N anime suggestions.  
- **Key Paper**  
  Cheng et al., *Wide & Deep Learning for Recommender Systems* (2016)  
- **Why this matters**  
  Classical recommenders either “remember” (collaborative filtering, popularity baselines) or “generalize” (deep neural CF), but rarely both. Wide & Deep unifies these strengths in a single model.

---

## 2. Data Description  
- **anime-dataset-2023.csv**  
  Contains metadata on ~17 000 anime titles (ID, title, community score, genres, global rank, popularity, etc.).  
- **users-details-2023.csv**  
  Contains profiles for ~900 000 MyAnimeList users (user ID, days watched, mean score, episodes watched, …).  
- **users-score-2023.csv**  
  Contains explicit ratings (1–10) that users have assigned to anime titles, yielding ~12 million user-anime interactions.

  The MyAnimeList 2023 dataset can be found on Kaggle here:

[Kaggle – MyAnimeList Dataset 2023](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset) ([kaggle.com][1])

[1]: https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset?utm_source=chatgpt.com "Anime Dataset 2023"


---

## 3. Pre-processing Highlights  
1. **Rank imputation**: Missing global ranks are predicted from popularity via linear regression.  
2. **Merge**: Ratings ⨝ anime ⨝ user tables into one master DataFrame.  
3. **Sampling**: Random 100 000 rows for prototyping speed.  
4. **Wide features** (memorization):  
   - Scaled numeric columns (Score, Popularity, Rank)  
   - One-hot encoded Genres  
5. **Deep features** (generalization):  
   - Embeddings for high-cardinality IDs (user_id, anime_id)  
6. **Train/Test split**: 80 % train, 20 % test, preserve alignment with wide-feature matrix.

---

## 4. Model Architecture  
- **Wide branch**  
  A single linear layer over the scaled numeric + one-hot genre features to “memorize” high-signal rules (e.g. a top-ranked anime should get a boost).  
- **Deep branch**  
  Embedding layers for user and anime IDs (32-dim each), concatenated and fed through two hidden ReLU layers (128 → 64 → 1), to “generalize” and capture latent taste similarities.  
- **Joint output**  
  Sum of wide and deep outputs, trained end-to-end with MSE loss and Adam optimizer.

---

## 5. Training & Evaluation  
- **Loss function**: Mean Squared Error on explicit ratings.  
- **Optimizer**: Adam, learning rate 1×10⁻³.  
- **Batch size**: 64, **Epochs**: 20.  
- **Metric**: Root-Mean-Squared-Error (RMSE) on held-out 20 % test set.  
- **Typical results**: Train RMSE ≈ 1.50, Test RMSE ≈ 1.70 (on 1–10 scale).

---

## 6. Generating Recommendations  
For a given user:  
1. Repeat that user’s embedding across all anime IDs.  
2. Score every anime in one forward pass (wide features + deep embeddings).  
3. Select the top-N highest predicted ratings.

---

## 7. Why Wide & Deep Beats Traditional Recommenders  

| Aspect                | Traditional CF / MF       | Wide & Deep                      |
|-----------------------|---------------------------|----------------------------------|
| **Memorization**      | Popularity baselines only | Wide branch directly models “if-then” rules (e.g. high rank → high rating) |
| **Generalization**    | CF struggles with cold-start; MF limited to latent factors | Deep branch infers unseen user-item patterns via embeddings |
| **Side information**  | Hard to incorporate       | Both branches accept numeric & categorical side features |
| **Feature crosses**   | Manual in LR or FM only   | Wide can take manual cross-features; Deep learns arbitrary high-order combos |
| **Latency**           | Simple dot-products fast  | One linear + small MLP is still sub-millisecond on GPU |
| **Interpretability**  | MF latent factors obscure | Wide weights are directly interpretable |

---

## 8. Importance of Wide vs. Deep  

- **Wide**  
  - Captures **explicit** associations you know are strong (e.g. “if an anime is in the top 50 by popularity, boost its score”).  
  - Ensures rare but critical exceptions (cold-start hits) are learned via sparse weights.

- **Deep**  
  - Learns **latent** user tastes and item characteristics from data, filling gaps where memorization fails.  
  - Smoothly generalizes to new user-anime pairs never seen in training.

Together, they **cover each other’s blind spots**: memorization handles sharp rules; generalization captures nuanced similarities.

---

## 9. How to Run  
1. Clone the repo, place the three CSVs in `data/`.  
2. Create a Python 3.10 environment, install dependencies (pandas, numpy, scikit-learn, torch).  
3. Open and run **Wide_and_Deep_Recommendation.ipynb** cells in order.  
4. Inspect training logs for per-epoch loss and final test RMSE.  
5. Call the recommendation API function for any user ID to view top-N suggestions.

---

## 10. References & Citation  
Cheng, Heng-Tze, et al. “Wide & Deep Learning for Recommender Systems.” arXiv preprint arXiv:1606.07792 (2016).

```bibtex
@article{Cheng2016WideDeep,
  title   = {Wide & Deep Learning for Recommender Systems},
  author  = {Cheng, Heng-Tze and Koc, Levent and Harmsen, et al.},
  journal = {arXiv preprint arXiv:1606.07792},
  year    = {2016}
}
