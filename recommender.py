# Project 3: AI Recommendation Logic - Tech Stack Recommender
# DecodeLabs Industrial Training - Batch 2026

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- DATASET: Job roles and their required skills ---
job_roles = {
    "Data Scientist":         "python machine learning sql data analysis statistics numpy pandas tensorflow scikit-learn",
    "Web Developer":          "html css javascript react nodejs frontend backend api responsive design",
    "DevOps Engineer":        "aws docker kubernetes linux ci cd automation cloud git jenkins pipelines",
    "Android Developer":      "java kotlin android mobile firebase api rest xml gradle",
    "Cybersecurity Analyst":  "networking linux ethical hacking encryption security firewall penetration testing",
    "Backend Developer":      "python java nodejs sql rest api databases server django flask",
    "AI Engineer":            "python deep learning tensorflow pytorch neural networks nlp computer vision transformers",
    "Cloud Architect":        "aws azure google cloud infrastructure terraform networking automation iac",
    "Data Analyst":           "excel sql python data visualization tableau powerbi reporting dashboards",
    "Full Stack Developer":   "html css javascript react nodejs python sql mongodb rest api fullstack",
    "ML Engineer":            "python mlops model deployment docker kubernetes tensorflow pytorch feature engineering",
    "Database Administrator": "sql mysql postgresql oracle database tuning backup recovery indexing",
    "UI/UX Designer":         "figma sketch wireframing prototyping user research css design systems accessibility",
    "Blockchain Developer":   "solidity ethereum smart contracts web3 cryptocurrency cryptography defi",
    "Game Developer":         "unity c# opengl game physics 3d modeling animation rendering"
}


def get_skill_gap(user_skill_set, role_doc):
    """Return skills in the role profile that the user didn't mention."""
    role_terms = set(role_doc.lower().split())
    missing = role_terms - user_skill_set
    # Filter out common stop-like tokens that aren't real skills
    noise = {"and", "or", "the", "a", "with", "for", "in", "of", "to"}
    return sorted(missing - noise)


def get_recommendations(user_skills, top_n=3):
    role_names = list(job_roles.keys())
    role_docs  = list(job_roles.values())

    vectorizer  = TfidfVectorizer()
    all_docs    = role_docs + [user_skills]
    tfidf_matrix = vectorizer.fit_transform(all_docs)

    user_vector  = tfidf_matrix[-1]
    role_vectors = tfidf_matrix[:-1]

    scores = cosine_similarity(user_vector, role_vectors)[0]
    ranked = sorted(zip(role_names, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def render_bar(score, width=20):
    filled = int(round(score * width))
    return "█" * filled + "░" * (width - filled)


def main():
    sep = "=" * 55
    print(sep)
    print("  PROJECT 3: AI TECH STACK RECOMMENDER")
    print("  DecodeLabs Industrial Training — Batch 2026")
    print(sep)
    print("\n  Enter your skills (comma separated, minimum 3)")
    print("  Example: python, machine learning, sql, pandas\n")

    try:
        raw_input_text = input("Your skills: ")
    except (KeyboardInterrupt, EOFError):
        print("\n  Session cancelled.")
        return

    user_skills_raw  = raw_input_text.lower().strip()
    skill_list       = [s.strip() for s in user_skills_raw.split(",") if s.strip()]

    if len(skill_list) < 3:
        print("\n  [ERROR] Please enter at least 3 skills.")
        return

    # Deduplicate while preserving order
    seen = set()
    unique_skills = []
    for s in skill_list:
        if s not in seen:
            unique_skills.append(s)
            seen.add(s)

    user_query    = " ".join(unique_skills)
    user_skill_set = set(unique_skills)

    print(f"\n  Skills recognised : {len(unique_skills)}")
    print(f"  Input             : {', '.join(unique_skills)}")

    print("\n  [PROCESS] Vectorizing skills using TF-IDF...")
    print("  [PROCESS] Calculating Cosine Similarity against 15 role profiles...")

    # Let user pick how many results they want
    try:
        top_n_input = input("\n  How many recommendations? (default 3, max 5): ").strip()
        top_n = int(top_n_input) if top_n_input.isdigit() else 3
        top_n = max(1, min(top_n, 5))
    except (ValueError, KeyboardInterrupt):
        top_n = 3

    recommendations = get_recommendations(user_query, top_n=top_n)

    print(f"\n{sep}")
    print(f"  TOP {top_n} RECOMMENDED CAREER PATH{'S' if top_n > 1 else ''}")
    print(sep)

    for rank, (role, score) in enumerate(recommendations, 1):
        pct  = round(score * 100, 2)
        bar  = render_bar(score)
        gap  = get_skill_gap(user_skill_set, job_roles[role])
        gap_display = ", ".join(gap[:5]) if gap else "None — strong match"

        print(f"\n  #{rank}  {role}")
        print(f"       Match  : {bar}  {pct}%")
        print(f"       Gaps   : {gap_display}")

    print(f"\n{sep}")
    print("  Tip: Add gap skills to your profile and re-run for a higher match score.")
    print(sep)


if __name__ == "__main__":
    main()
