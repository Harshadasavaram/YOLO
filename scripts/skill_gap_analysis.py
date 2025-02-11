# import json

# def extract_skills(text, skills_database):
#     """Extract matching skills from the given text based on a skills database."""
#     text_lower = text.lower()
#     matched_skills = [skill for skill in skills_database if skill in text_lower]
#     return matched_skills

# def analyze_skill_gaps():
#     """Perform skill gap analysis between the resume and job requirements."""
#     # Load the skills database
#     with open("data/skills_database.json", "r") as file:
#         skills_data = json.load(file)
#         skills_database = skills_data["skills"]

#     # Load parsed resume
#     with open("data/resume_parsed.txt", "r") as file:
#         resume_content = file.read()
    
#     # Extract skills from the resume
#     resume_skills = extract_skills(resume_content, skills_database)

#     # Load job listings
#     with open("data/job_listings.json", "r") as file:
#         job_listings = json.load(file)
    
#     # Analyze skill gaps for each job
#     skill_gap_results = []
#     for job in job_listings:
#         job_skills = extract_skills(job["description"], skills_database)
#         missing_skills = [skill for skill in job_skills if skill not in resume_skills]
#         missing_skills_with_resources = [
#             {"skill": skill, "resource": skills_data["resources"].get(skill, "No resource available")}
#             for skill in missing_skills
#         ]
#         skill_gap_results.append({
#             "title": job["title"],
#             "company": job["company"],
#             "location": job["location"],
#             "missing_skills": missing_skills_with_resources
#         })
    
#     # Save skill gap results to a file
#     with open("data/skill_gaps.json", "w") as file:
#         json.dump(skill_gap_results, file, indent=4)
    
#     print("Skill gap analysis completed. Results saved to 'data/skill_gaps.json'.")

# if __name__ == "__main__":
#     analyze_skill_gaps()

# import json

# def extract_skills(text, skills_database):
#     """Extract matching skills from the given text based on a skills database."""
#     text_lower = text.lower()
#     matched_skills = [skill for skill in skills_database if skill in text_lower]
#     return matched_skills

# def analyze_skill_gaps(user_skills, job_listings):
#     """
#     Perform skill gap analysis between the resume skills and job requirements.

#     Args:
#         user_skills (list): List of skills extracted from the resume.
#         job_listings (list): List of job dictionaries containing 'title' and 'description'.

#     Returns:
#         list: A list of dictionaries containing missing skills for each job.
#     """
#     # Load the skills database
#     with open("data/skills_database.json", "r") as file:
#         skills_data = json.load(file)
#         skills_database = skills_data["skills"]

#     # Analyze skill gaps for each job
#     skill_gap_results = []
#     for job in job_listings:
#         job_title = job.get("title", "Unknown Job Title")
#         job_description = job.get("description", "").lower()

#         # Extract required skills from job description
#         job_skills = extract_skills(job_description, skills_database)

#         # Identify missing skills
#         missing_skills = [skill for skill in job_skills if skill not in user_skills]

#         # Attach learning resources for missing skills
#         missing_skills_with_resources = [
#             {"skill": skill, "resource": skills_data["resources"].get(skill, "No resource available")}
#             for skill in missing_skills
#         ]

#         skill_gap_results.append({
#             "title": job_title,
#             "company": job.get("company", "Unknown Company"),
#             "location": job.get("location", "Unknown Location"),
#             "missing_skills": missing_skills_with_resources
#         })

#     return skill_gap_results

# import json
# import os

# def analyze_skill_gaps():
#     """Perform skill gap analysis and append results instead of overwriting."""
#     # Load the skills database
#     with open("data/skills_database.json", "r") as file:
#         skills_data = json.load(file)
#         skills_database = skills_data["skills"]

#     # Load parsed resume
#     with open("data/resume_parsed.txt", "r") as file:
#         resume_content = file.read()
    
#     # Extract skills from the resume
#     resume_skills = extract_skills(resume_content, skills_database)

#     # Load job listings
#     with open("data/job_listings.json", "r") as file:
#         job_listings = json.load(file)
    
#     # Load existing skill gap data if available
#     skill_gap_results = []
#     if os.path.exists("data/skill_gaps.json"):
#         with open("data/skill_gaps.json", "r") as file:
#             skill_gap_results = json.load(file)

#     # Analyze skill gaps for each job
#     for job in job_listings:
#         job_skills = extract_skills(job["description"], skills_database)
#         missing_skills = [skill for skill in job_skills if skill not in resume_skills]
#         missing_skills_with_resources = [
#             {"skill": skill, "resource": skills_data["resources"].get(skill, "No resource available")}
#             for skill in missing_skills
#         ]
#         skill_gap_results.append({
#             "title": job["title"],
#             "company": job["company"],
#             "location": job["location"],
#             "missing_skills": missing_skills_with_resources
#         })
    
#     # Save skill gap results to a file (Append Mode)
#     with open("data/skill_gaps.json", "w") as file:
#         json.dump(skill_gap_results, file, indent=4)
    
#     print("Skill gap analysis completed. Results saved to 'data/skill_gaps.json'.")

import os
import json
import spacy
from pymongo import MongoClient
from pdfminer.high_level import extract_text

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Connect to MongoDB
client = MongoClient("mongodb://127.0.0.1:27017/")
db = client["career_advisor"]
skills_collection = db["skills"]

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    """Extracts text from a PDF resume."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Error: Resume file not found at {file_path}")

    text = extract_text(file_path)
    if not text.strip():
        raise ValueError("❌ Resume text extraction failed. Check if the PDF contains selectable text.")
    
    return text

# Function to fetch predefined skills from MongoDB
def fetch_all_skills():
    """Fetches all predefined skills from MongoDB."""
    skill_docs = skills_collection.find({}, {"_id": 0, "skill": 1})
    skills = {doc["skill"].lower().strip() for doc in skill_docs if "skill" in doc}
    
    if not skills:
        raise ValueError("❌ No skills found in MongoDB! Insert skills first.")
    
    return skills

# Function to extract skills from resume text
def extract_skills(text):
    """Extracts skills from resume text and matches them with predefined skills."""
    doc = nlp(text)
    extracted_skills = set()
    
    # Get predefined skills from MongoDB
    predefined_skills = fetch_all_skills()

    # Match words with predefined skills
    for token in doc:
        word = token.text.lower().strip()
        if word in predefined_skills:
            extracted_skills.add(word)

    return list(extracted_skills)

# Analyze the skill gap
def analyze_skill_gap(resume_path):
    """
    Analyzes the skill gap between the resume and job market demands.
    
    Returns:
        List of missing skills.
    """

    # Validate file existence
    if not isinstance(resume_path, str):
        raise TypeError("Invalid file path: Expected a string.")

    # Extract text from resume
    resume_text = extract_text_from_pdf(resume_path)

    # Extract skills from resume
    extracted_skills = extract_skills(resume_text)

    # Fetch required skills from MongoDB
    all_required_skills = fetch_all_skills()

    # Find missing skills
    missing_skills = list(all_required_skills - set(extracted_skills))

    # Display results
    print("\n✅ Skill Gap Analysis Completed!")
    print(f"🟢 Extracted Skills: {extracted_skills}" if extracted_skills else "⚠️ No skills extracted!")
    print(f"🔴 Missing Skills: {missing_skills}" if missing_skills else "🎉 No missing skills!")

    return missing_skills  # 🔥 Fixed return statement

# Run script independently for testing
if __name__ == "__main__":
    resume_path = "data/sample_resume.pdf"  # Change this to an actual resume path
    try:
        missing_skills = analyze_skill_gap(resume_path)
        if missing_skills:
            print("\n⚠️ Top 5 Missing Skills:")
            for skill in missing_skills[:5]:
                print(f"🔹 {skill}")
        else:
            print("\n🎉 No missing skills detected!")
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(e)

