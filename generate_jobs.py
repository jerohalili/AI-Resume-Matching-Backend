import json
import random
from openai import OpenAI

# --- CONFIGURATION ---
# Local Client setup for AnythingLLM / Local LLM
client = OpenAI(
    base_url="http://127.0.0.1:3001/api/v1", 
    api_key="local api key"
)

def generate_jobs_from_labels(input_file, output_file):
    """
    Synthesizes a realistic jobs dataset from a labels.json library.
    Implements core/noise sampling and professional persona enforcement.
    """
    try:
        with open(input_file, 'r') as f:
            labels_data = json.load(f)
    except Exception as e:
        print(f"Error loading labels: {e}")
        return

    # 1. ANALYZE LABELS: Group all skills by their unique job titles
    role_library = {}
    for entry in labels_data.values():
        title = entry.get('expected_job_title', 'Specialist')
        # Guard against missing or null skills; convert to list to ensure indexability
        skills = list(set(entry.get('expected_skills', []) or []))
        
        if title not in role_library:
            role_library[title] = []
        role_library[title].append(skills)

    # 2. SEED DATA for Realism
    companies = [
        "Microcosmos", "TechFlow", "Nexus Systems", "CloudScale", 
        "Vertex Solutions", "InnovateIQ", "Cyberdyne", "Aether Tech",
        "Global Logistics", "Starlight Data", "Ember Systems", "Axiom Corp"
    ]
    
    jobs_dataset = []
    job_id = 1

    print(f"Starting synthesis of {len(role_library) * 2} professional job postings...")

    # 3. GENERATION LOOP
    for title, lists_of_skills in role_library.items():
        # Create a broader "pool" of all skills associated with this specific title
        all_possible_skills = set().union(*[set(s) for s in lists_of_skills])
        
        # Create 2 variations per title to provide dataset variety
        for _ in range(2):
            company = random.choice(companies)
            
            # Pick a core set from one of the actual resumes
            core_set = random.choice(lists_of_skills)
            
            # --- STOCHASTIC SAMPLING LOGIC (Safety Hardened) ---
            if not core_set:
                # Fallback: pull from the broader title pool if the specific resume was empty
                pool_list = list(all_possible_skills)
                if pool_list:
                    sample_count = min(len(pool_list), random.randint(3, 5))
                    job_skills = random.sample(pool_list, sample_count)
                else:
                    job_skills = ["Technical Excellence"]
            else:
                # Ensure we are sampling from unique elements
                unique_core = list(set(core_set))
                pop_size = len(unique_core)
                
                # Calculate sample size (70-90% of unique skills)
                sample_size = int(pop_size * random.uniform(0.7, 0.9))
                
                # FINAL SAFETY GUARD: random.sample(population, k) requires k <= len(population)
                k = max(1, min(sample_size, pop_size))
                job_skills = random.sample(unique_core, k)
            
            # Add 1-3 "Aspirational" (Noise) skills that weren't in the sampled list
            extra_pool = list(all_possible_skills - set(job_skills))
            if extra_pool:
                noise_limit = min(len(extra_pool), random.randint(1, 3))
                job_skills.extend(random.sample(extra_pool, noise_limit))

            random.shuffle(job_skills)
            skills_context = ", ".join(job_skills) if job_skills else "General Industry Standards"

            # 4. LLM PROMPT: Engineering for High-Energy Professionalism
            prompt = (
                f"Role: {title} at {company}\n"
                f"Requirements: {skills_context}\n"
                f"Task: Write a high-impact, professional 2-sentence job description. "
                f"Sentence 1: The overarching mission or impact of the role. "
                f"Sentence 2: Naturally integrate at least 2 or 3 of the technical requirements. "
                f"CRITICAL: Do NOT use 'Seeking a', 'We are looking for', or 'Join our team'. "
                f"Tone: Aggressive professional engineering. No fluff."
            )

            try:
                response = client.chat.completions.create(
                    model="Meta-Llama-3.1-8B-Instruct-Q8_0", 
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8
                )
                description = response.choices[0].message.content.strip().replace('"', '')
            except Exception:
                # Robust fallback for API timeouts
                description = (f"Leading critical infrastructure initiatives as a {title} for {company}. "
                               f"Expertise in {', '.join(job_skills[:2])} is essential for system optimization.")

            # 5. ASSEMBLE OBJECT
            jobs_dataset.append({
                "id": job_id,
                "title": f"{title} at {company}",
                "description": description,
                "required_skills": job_skills
            })
            
            print(f"[{job_id}] Synthesized: {title} @ {company}")
            job_id += 1

    # 6. FINAL SHUFFLE
    random.shuffle(jobs_dataset)

    # 7. SAVE OUTPUT
    with open(output_file, 'w') as f:
        json.dump(jobs_dataset, f, indent=4)

    print(f"\nDone! Updated {output_file} with {len(jobs_dataset)} realistic entries.")

if __name__ == "__main__":
    generate_jobs_from_labels('labels.json', 'jobs_data.json')