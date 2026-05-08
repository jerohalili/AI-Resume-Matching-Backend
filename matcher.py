import fitz
import easyocr
import torch
import re
import os
import numpy as np
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz, process

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POPPLER_PATH = os.path.join(BASE_DIR, "poppler-25.12.0", "Library", "bin")

# --- INITIALIZATION ---
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./models/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_PATH, device=device)
reader = easyocr.Reader(['en'], gpu=True)

# Skills to ignore globally
SKILL_STOP_WORDS = {"lead", "data", "management", "reporting", "remote", "team", "vision"}

def _generate_ocr_tolerant_pattern(skill_name: str) -> str:
    pattern = re.escape(skill_name.lower().strip())
    pattern = pattern.replace(r'\/', r'[\/\\\|I1l]')
    pattern = pattern.replace(r'\.', r'[\.\,\-\s]?')
    pattern = pattern.replace(r'\ ', r'\s+')
    pattern = pattern.replace('cl', '(cl|ll)')
    return rf'(?i)(?:^|(?<=[^a-z0-9])){pattern}(?=[^a-z0-9]|$)'

def extract_text(file_path):
    text = ""
    if file_path.lower().endswith('.pdf'):
        doc = fitz.open(file_path)
        text = " ".join([page.get_text("text") for page in doc])
        doc.close()

    if len(text.strip()) < 50:
        images = convert_from_path(file_path, poppler_path=POPPLER_PATH)
        raw_pages = [" ".join(reader.readtext(np.array(img), detail=0)) for img in images]
        text = " ".join(raw_pages)

    clean_text = re.sub(r'\s+', ' ', text).strip()
    meaningful_chars = re.sub(r'[\s\.\,\-\|\/•]', '', clean_text)
    alnum_chars = [c for c in meaningful_chars if c.isalnum()]
    
    if not meaningful_chars:
        return clean_text, 0.0
    
    quality_score = (len(alnum_chars) / len(meaningful_chars)) * 100
    return clean_text, min(round(quality_score, 2), 100.0)

def discover_skills_dynamically(text, jobs_corpus):
    unique_skills = {}
    for job in jobs_corpus:
        for skill in job.get('required_skills', []):
            skill_clean = skill.lower().strip()
            if skill_clean in SKILL_STOP_WORDS or len(skill_clean) < 2:
                continue
            unique_skills[skill_clean] = skill.strip()

    searchable = f" {text.lower()} "
    found_skills = []
    
    # Sort by length DESCENDING to handle multi-word skills first
    sorted_skills = sorted(unique_skills.items(), key=lambda x: len(x[0]), reverse=True)

    for skill_lc, original_name in sorted_skills:
        pattern = _generate_ocr_tolerant_pattern(skill_lc)
        
        # 1. Regex Match
        if re.search(pattern, searchable):
            found_skills.append(original_name)
            searchable = re.sub(pattern, ' [MASKED_SKILL] ', searchable)
        else:
            # 2. Fuzzy Window Search for "Dirty" OCR
            words = searchable.split()
            for i in range(len(words)):
                # Check 1-word and 2-word windows (expand to 3 if skills are long)
                for n in range(1, 3): 
                    window = " ".join(words[i : i + n])
                    if not window or "[MASKED_SKILL]" in window:
                        continue
                        
                    if fuzz.ratio(skill_lc, window) > 90:
                        found_skills.append(original_name)
                        # Mask the fuzzy hit to prevent redundant matches
                        searchable = searchable.replace(window, " [MASKED_SKILL] ", 1)
                        break 
    
    return list(dict.fromkeys(found_skills))

def fuzzy_skill_verify(discovered_list, ground_truth_list, threshold=85):
    verified = []
    found_in_discovered = set()
    disc_lower = {d.lower().strip(): d for d in discovered_list}

    for truth in ground_truth_list:
        t_clean = truth.lower().strip()
        if t_clean in disc_lower:
            verified.append(truth)
            found_in_discovered.add(disc_lower[t_clean])
            continue

        if disc_lower:
            best = process.extractOne(
                t_clean, 
                list(disc_lower.keys()), 
                scorer=fuzz.token_set_ratio, 
                score_cutoff=threshold
            )
            if best:
                verified.append(truth)
                found_in_discovered.add(disc_lower[best[0]])
                
    return verified, list(found_in_discovered)

def calculate_cosine_similarity(resume_text, resume_skills, jobs_corpus):
    skill_str = ' '.join(resume_skills) if resume_skills else ''
    resume_profile = f"Skills: {skill_str}. Resume: {resume_text[:2500]}"
    resume_vec = model.encode(resume_profile, convert_to_tensor=True)

    processed_job_texts = [
        f"Job: {j['title']}. Core: {' '.join(j['required_skills'] * 2)}. {j.get('description', '')}"
        for j in jobs_corpus
    ]

    job_vecs = model.encode(processed_job_texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(resume_vec, job_vecs)[0]
    resume_set = {s.lower().strip() for s in resume_skills}

    results = []
    for i, score in enumerate(cosine_scores):
        job_skills = jobs_corpus[i]['required_skills']
        matched = [s for s in job_skills if s.lower().strip() in resume_set]
        missing = [s for s in job_skills if s.lower().strip() not in resume_set]

        match_ratio = len(matched) / len(job_skills) if job_skills else 0
        final_score = (float(score) * 0.6) + (match_ratio * 0.4)

        results.append({
            "job_id": jobs_corpus[i]['id'],
            "title": jobs_corpus[i]['title'],
            "score": f"{round(final_score * 100, 2)}%",
            "matched_skills": matched,
            "missing_skills": missing
        })

    return sorted(results, key=lambda x: float(x['score'].replace('%', '')), reverse=True)[:5]