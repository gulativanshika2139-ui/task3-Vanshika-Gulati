# AI Recommendation Logic DecodeLabs Project 3

A content-based Tech Stack Recommender built using Python and scikit-learn as part of the **DecodeLabs AI Industrial Training (Batch 2026)**.

---

## How It Works

- User enters 3+ skills as input
- Duplicate skills are filtered automatically
- Skills are vectorized using TF-IDF weighting
- Cosine Similarity is calculated against **15 job role profiles**
- Top-N most relevant career paths are returned, ranked by match score
- Skill gaps are shown for each recommended role

---

## Pipeline

```
Input: 3+ user skills (comma separated)
        │
        ▼
Deduplicate + normalize (lowercase, strip whitespace)
        │
        ▼
TF-IDF vectorization across all role profiles + user input
        │
        ▼
Cosine Similarity scored against each role
        │
        ▼
Top-N roles ranked by score
        │
        ▼
Skill gap computed per recommended role
        │
        ▼
Output: Ranked roles with match % + bar + gap skills
```

---

## Example

```
Your skills: python, machine learning, sql, pandas
How many recommendations? (default 3, max 5): 3

  #1  Data Scientist
       Match  : ████████████████░░░░  79.4%
       Gaps   : scikit-learn, statistics, tensorflow

  #2  AI Engineer
       Match  : █████████████░░░░░░░  65.1%
       Gaps   : computer vision, nlp, pytorch, transformers

  #3  Data Analyst
       Match  : ████████████░░░░░░░░  61.8%
       Gaps   : dashboards, powerbi, reporting, tableau
```

---

## Results

| Metric | Detail |
|---|---|
| Role profiles | 15 |
| Vectorization | TF-IDF (sklearn) |
| Similarity method | Cosine Similarity |
| Output | Ranked roles + match % + skill gaps |
| Min input | 3 skills |
| Max recommendations | 5 |

---
## How to Run

```
python recommender.py
```

Requires Python 3.6+.

---

## Dependencies

```
pip install scikit-learn
```

No other external libraries required.
