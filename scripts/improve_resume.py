
# import openai
# import json
# import logging
# import os

# # Configure logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # Load OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# def generate_suggestions(job_title, resume_text):
#     """
#     Generate suggestions to improve a resume based on a given job title using OpenAI's ChatCompletion API.
#     """
#     try:
#         # Prepare the prompt
#         prompt = (
#             f"Job Title: {job_title}\n"
#             f"Candidate's Resume:\n{resume_text}\n\n"
#             f"Provide detailed suggestions to improve the resume so it aligns better with the job title."
#         )
        
#         # Make the API call using ChatCompletion
#         response = openai.ChatCompletion.create(
#             model="gpt-4",  # Updated to use the latest model
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant who provides detailed suggestions for improving resumes."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.7,
#             max_tokens=500
#         )
        
#         # Extract suggestions from the response
#         suggestions = response['choices'][0]['message']['content']
#         return suggestions

#     except Exception as e:
#         logging.error(f"Error generating suggestions: {e}")
#         return None

# def load_file(file_path, file_type="text"):
#     """
#     Load content from a file.
#     """
#     try:
#         if file_type == "text":
#             with open(file_path, "r") as file:
#                 return file.read()
#         elif file_type == "json":
#             with open(file_path, "r") as file:
#                 return json.load(file)
#     except FileNotFoundError:
#         logging.error(f"File not found: {file_path}")
#         return None
#     except Exception as e:
#         logging.error(f"Error loading file {file_path}: {e}")
#         return None
#     except Exception as e:
#         logging.error(f"Error generating suggestions: {e}")
#         if hasattr(e, 'response') and e.response is not None:
#             logging.error(f"Response details: {e.response}")


# def save_json(data, file_path):
#     """
#     Save data to a JSON file.
#     """
#     try:
#         with open(file_path, "w") as file:
#             json.dump(data, file, indent=4)
#         logging.info(f"Data successfully saved to {file_path}.")
#     except Exception as e:
#         logging.error(f"Error saving data to {file_path}: {e}")

# def main():
#     """
#     Main function to generate and save resume improvement suggestions.
#     """
#     logging.info("Starting resume improvement process...")
    
#     # Define paths
#     data_folder = "data"
#     resume_path = os.path.join(data_folder, "resume_parsed.txt")
#     job_listings_path = os.path.join(data_folder, "job_listings.json")
#     output_path = os.path.join(data_folder, "resume_improvements.json")

#     # Load parsed resume
#     resume_text = load_file(resume_path, file_type="text")
#     if not resume_text:
#         return

#     # Load job listings
#     job_listings = load_file(job_listings_path, file_type="json")
#     if not job_listings:
#         return

#     # Generate suggestions for each job listing
#     suggestions_dict = {}
#     for job in job_listings:
#         job_title = job.get("title", "Unknown Job Title")
#         logging.info(f"Generating suggestions for job: {job_title}")
#         suggestions = generate_suggestions(job_title, resume_text)
#         if suggestions:
#             suggestions_dict[job_title] = suggestions

#     # Save suggestions to a JSON file
#     save_json(suggestions_dict, output_path)

# if __name__ == "__main__":
#     main()

#9th feb
# import json
# import os
# import spacy
# import chardet
# from sentence_transformers import SentenceTransformer, util

# # Load pre-trained NLP models
# nlp = spacy.load("en_core_web_sm")
# model = SentenceTransformer("all-MiniLM-L6-v2")

# def detect_encoding(file_path):
#     """Detect file encoding to handle decoding issues."""
#     with open(file_path, "rb") as f:
#         raw_data = f.read()
#         result = chardet.detect(raw_data)
#         return result["encoding"]

# def read_resume_text(file_path):
#     """Read resume text with detected encoding."""
#     encoding = detect_encoding(file_path)
#     with open(file_path, "r", encoding=encoding, errors="replace") as file:
#         return file.read()

# def extract_entities(text):
#     """Extract key skills, education, and experience from the resume."""
#     doc = nlp(text)
#     skills, education, experience = set(), set(), set()

#     for ent in doc.ents:
#         if ent.label_ in ["ORG", "EDUCATION"]:
#             education.add(ent.text)
#         elif ent.label_ in ["WORK_OF_ART", "PRODUCT", "FAC"]:
#             skills.add(ent.text)
#         elif ent.label_ in ["DATE", "TIME"]:
#             experience.add(ent.text)

#     return list(skills), list(education), list(experience)

# def load_json_data(file_path):
#     """Load JSON data from a given file."""
#     with open(file_path, "r", encoding="utf-8") as file:
#         return json.load(file)

# def match_jobs(resume_text, job_listings):
#     """Match jobs based on skill similarity."""
#     resume_embedding = model.encode(resume_text, convert_to_tensor=True)
#     matched_jobs = []

#     for job in job_listings:
#         job_text = job["title"] + " " + job.get("description", "")
#         job_embedding = model.encode(job_text, convert_to_tensor=True)
#         similarity = util.pytorch_cos_sim(resume_embedding, job_embedding).item()

#         matched_jobs.append({
#             "title": job["title"],
#             "company": job["company"],
#             "location": job["location"],
#             "url": job["url"],
#             "similarity_score": round(similarity, 2)
#         })

#     matched_jobs.sort(key=lambda x: x["similarity_score"], reverse=True)
#     return matched_jobs[:5]  # Return top 5 matches

# def generate_resume_improvements(skills_found, missing_skills):
#     """Generate resume improvement suggestions."""
#     suggestions = []

#     if len(skills_found) < 5:
#         suggestions.append("Consider adding more skills to highlight your expertise.")

#     if len(missing_skills) > 0:
#         suggestions.append(f"Improve your resume by learning: {', '.join(missing_skills)}.")

#     if "Python" not in skills_found:
#         suggestions.append("Python is highly valued in tech jobs. Consider including it.")

#     return suggestions

# def main():
#     # Read parsed resume text
#     resume_text = read_resume_text("data/resume_parsed.txt")

#     # Extract skills, education, and experience
#     skills_found, education, experience = extract_entities(resume_text)

#     # Load job listings
#     job_listings = load_json_data("data/job_listings.json")

#     # Match jobs based on similarity
#     matched_jobs = match_jobs(resume_text, job_listings)

#     # Load skill gap data
#     skill_gaps = load_json_data("data/skill_gaps.json")

#     # Extract missing skills
#     missing_skills = {skill["skill"] for job in skill_gaps for skill in job["missing_skills"]}

#     # Generate resume improvement suggestions
#     resume_suggestions = generate_resume_improvements(skills_found, missing_skills)

#     # Save resume improvements
#     improvements_data = {
#         "skills_found": skills_found,
#         "missing_skills": list(missing_skills),
#         "resume_suggestions": resume_suggestions,
#         "matched_jobs": matched_jobs
#     }

#     with open("data/resume_improvements.json", "w", encoding="utf-8") as file:
#         json.dump(improvements_data, file, indent=4)

#     print("✅ Resume improvement suggestions and job matches saved to 'data/resume_improvements.json'.")

# if __name__ == "__main__":
#     main()


#10th feb 12:05 am


# import json
# import os
# import spacy
# import chardet
# import numpy as np
# from sentence_transformers import SentenceTransformer, util
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Load pre-trained NLP models
# nlp = spacy.load("en_core_web_sm")
# model = SentenceTransformer("all-MiniLM-L6-v2")

# def detect_encoding(file_path):
#     """Detect file encoding to handle decoding issues."""
#     with open(file_path, "rb") as f:
#         raw_data = f.read()
#         result = chardet.detect(raw_data)
#         return result["encoding"]

# def read_resume_text(file_path):
#     """Read resume text with detected encoding."""
#     encoding = detect_encoding(file_path)
#     with open(file_path, "r", encoding=encoding, errors="replace") as file:
#         return file.read()

# def extract_entities(text):
#     """Extract skills, education, and experience from the resume."""
#     doc = nlp(text)
#     skills, education, experience = set(), set(), set()
    
#     for ent in doc.ents:
#         if ent.label_ in ["ORG", "EDUCATION"]:
#             education.add(ent.text)
#         elif ent.label_ in ["WORK_OF_ART", "PRODUCT", "FAC"]:
#             skills.add(ent.text)
#         elif ent.label_ in ["DATE", "TIME"]:
#             experience.add(ent.text)
    
#     return list(skills), list(education), list(experience)

# def load_json_data(file_path):
#     """Load JSON data from a given file."""
#     with open(file_path, "r", encoding="utf-8") as file:
#         return json.load(file)

# def compute_tfidf_similarity(resume_text, job_listings):
#     """Compute TF-IDF similarity between resume and job descriptions."""
#     job_texts = [job["title"] + " " + job.get("description", "") for job in job_listings]
    
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform([resume_text] + job_texts)
#     cosine_similarities = (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1:]
    
#     return cosine_similarities

# def match_jobs(resume_text, job_listings):
#     """Match jobs using a hybrid approach (TF-IDF + semantic similarity)."""
#     resume_embedding = model.encode(resume_text, convert_to_tensor=True)
#     tfidf_similarities = compute_tfidf_similarity(resume_text, job_listings)
    
#     matched_jobs = []
    
#     for idx, job in enumerate(job_listings):
#         job_text = job["title"] + " " + job.get("description", "")
#         job_embedding = model.encode(job_text, convert_to_tensor=True)
#         semantic_similarity = util.pytorch_cos_sim(resume_embedding, job_embedding).item()
        
#         # Weighted score: 70% semantic similarity + 30% TF-IDF
#         combined_score = (0.7 * semantic_similarity) + (0.3 * tfidf_similarities[idx])
        
#         matched_jobs.append({
#             "title": job["title"],
#             "company": job["company"],
#             "location": job["location"],
#             "url": job["url"],
#             "similarity_score": round(combined_score, 2)
#         })
    
#     matched_jobs.sort(key=lambda x: x["similarity_score"], reverse=True)
#     return matched_jobs[:20]  # Return top 20 matches

# def generate_resume_improvements(skills_found, missing_skills, education, experience):
#     """Generate detailed resume improvement suggestions."""
#     suggestions = []
    
#     if len(skills_found) < 5:
#         suggestions.append("Your resume has fewer skills listed. Consider adding more to showcase your expertise.")
    
#     if missing_skills:
#         suggestions.append(f"You are missing key skills required for your desired roles: {', '.join(missing_skills)}. Consider learning these through online courses, certifications, or projects.")
    
#     if "Python" not in skills_found and "python" not in missing_skills:
#         suggestions.append("Python is a highly sought-after skill. If you know it, explicitly mention it in your resume.")
    
#     if education:
#         suggestions.append(f"Your resume includes education details: {', '.join(education)}. Ensure they are formatted properly.")
#     else:
#         suggestions.append("Consider adding your educational background. Recruiters often look for degrees and certifications relevant to the job role.")
    
#     if experience:
#         suggestions.append("Your resume includes work experience. Ensure it's structured with bullet points highlighting key achievements.")
#     else:
#         suggestions.append("Adding relevant experience, even in the form of internships or projects, can make your resume stronger.")
    
#     return suggestions

# def main():
#     # Read parsed resume text
#     resume_text = read_resume_text("data/resume_parsed.txt")
    
#     # Extract skills, education, and experience
#     skills_found, education, experience = extract_entities(resume_text)
    
#     # Load job listings
#     job_listings = load_json_data("data/job_listings.json")
    
#     # Match jobs using hybrid approach
#     matched_jobs = match_jobs(resume_text, job_listings)
    
#     # Load skill gap data
#     skill_gaps = load_json_data("data/skill_gaps.json")
    
#     # Extract missing skills
#     missing_skills = {skill["skill"] for job in skill_gaps for skill in job["missing_skills"]}
    
#     # Generate resume improvement suggestions
#     resume_suggestions = generate_resume_improvements(skills_found, missing_skills, education, experience)
    
#     # Save resume improvements
#     improvements_data = {
#         "skills_found": skills_found,
#         "missing_skills": list(missing_skills),
#         "resume_suggestions": resume_suggestions,
#         "matched_jobs": matched_jobs
#     }
    
#     with open("data/resume_improvements.json", "w", encoding="utf-8") as file:
#         json.dump(improvements_data, file, indent=4)
    
#     print(" Resume improvement suggestions and job matches saved to 'data/resume_improvements.json'.")

# if __name__ == "__main__":
#     main()

# import json
# import os
# import spacy
# import chardet
# import numpy as np
# from sentence_transformers import SentenceTransformer, util
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Load pre-trained NLP models
# nlp = spacy.load("en_core_web_sm")
# model = SentenceTransformer("all-MiniLM-L6-v2")

# def detect_encoding(file_path):
#     """Detect file encoding to handle decoding issues."""
#     with open(file_path, "rb") as f:
#         raw_data = f.read()
#         result = chardet.detect(raw_data)
#         return result["encoding"]

# def read_resume_text(file_path):
#     """Read resume text with detected encoding."""
#     encoding = detect_encoding(file_path)
#     with open(file_path, "r", encoding=encoding, errors="replace") as file:
#         return file.read()

# def extract_entities(text):
#     """Extract skills, education, and experience from the resume."""
#     doc = nlp(text)
#     skills, education, experience = set(), set(), set()
    
#     for ent in doc.ents:
#         if ent.label_ in ["ORG", "EDUCATION"]:
#             education.add(ent.text)
#         elif ent.label_ in ["WORK_OF_ART", "PRODUCT", "FAC"]:
#             skills.add(ent.text)
#         elif ent.label_ in ["DATE", "TIME"]:
#             experience.add(ent.text)
    
#     return list(skills), list(education), list(experience)

# def load_json_data(file_path):
#     """Load JSON data from a given file."""
#     with open(file_path, "r", encoding="utf-8") as file:
#         return json.load(file)

# def generate_resume_improvements(skills_found, missing_skills, education, experience):
#     """Generate detailed resume improvement suggestions."""
#     suggestions = []
    
#     if len(skills_found) < 5:
#         suggestions.append("Your resume has fewer skills listed. Consider adding more to showcase your expertise.")
    
#     if missing_skills:
#         suggestions.append(f"You are missing key skills required for your desired roles: {', '.join(missing_skills)}. Consider learning these through online courses, certifications, or projects.")
    
#     if "Python" not in skills_found and "python" not in missing_skills:
#         suggestions.append("Python is a highly sought-after skill. If you know it, explicitly mention it in your resume.")
    
#     if education:
#         suggestions.append(f"Your resume includes education details: {', '.join(education)}. Ensure they are formatted properly.")
#     else:
#         suggestions.append("Consider adding your educational background. Recruiters often look for degrees and certifications relevant to the job role.")
    
#     if experience:
#         suggestions.append("Your resume includes work experience. Ensure it's structured with bullet points highlighting key achievements.")
#     else:
#         suggestions.append("Adding relevant experience, even in the form of internships or projects, can make your resume stronger.")
    
#     return suggestions

# # def improve_resume(resume_text, job_listings, skill_gaps):
# def improve_resume(resume_text, extracted_data, job_listings, skill_gaps):

#     """
#     Main function that extracts resume details, matches jobs, and provides improvement suggestions.
#     """
#     # Extract skills, education, and experience
#     skills_found, education, experience = extract_entities(resume_text)

#     # Extract missing skills
#     missing_skills = {skill["skill"] for job in skill_gaps for skill in job["missing_skills"]}

#     # Generate resume improvement suggestions
#     resume_suggestions = generate_resume_improvements(skills_found, missing_skills, education, experience)

#     # Save resume improvements
#     improvements_data = {
#         "skills_found": skills_found,
#         "missing_skills": list(missing_skills),
#         "resume_suggestions": resume_suggestions
#     }

#     return improvements_data

# import json
# import spacy
# import numpy as np
# from pymongo import MongoClient
# from sentence_transformers import SentenceTransformer
# from scripts.parse_resume import parse_resume

# # Load pre-trained NLP models
# nlp = spacy.load("en_core_web_sm")
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Connect to MongoDB
# client = MongoClient("mongodb://127.0.0.1:27017/")
# db = client["career_advisor"]
# skill_gaps_collection = db["skill_gaps"]
# resume_improvements_collection = db["resume_improvements"]

# def extract_entities(text):
#     """Extract skills, education, and experience from the resume."""
#     doc = nlp(text)
#     skills, education, experience = set(), set(), set()
    
#     for ent in doc.ents:
#         if ent.label_ in ["ORG", "EDUCATION"]:
#             education.add(ent.text)
#         elif ent.label_ in ["WORK_OF_ART", "PRODUCT", "FAC"]:
#             skills.add(ent.text)
#         elif ent.label_ in ["DATE", "TIME"]:
#             experience.add(ent.text)
    
#     return list(skills), list(education), list(experience)

# def fetch_missing_skills():
#     """Fetch missing skills from MongoDB skill_gaps collection."""
#     skill_gap_entry = skill_gaps_collection.find_one({}, {"_id": 0, "missing_skills": 1})
#     return skill_gap_entry["missing_skills"] if skill_gap_entry else []

# def generate_resume_improvements(skills_found, missing_skills, education, experience):
#     """Generate detailed resume improvement suggestions."""
#     suggestions = []

#     if len(skills_found) < 5:
#         suggestions.append("Your resume has fewer skills listed. Consider adding more to showcase your expertise.")

#     if missing_skills:
#         suggestions.append(f"You are missing key skills required for your desired roles: {', '.join(missing_skills)}. Consider learning these through online courses, certifications, or projects.")

#     if "Python" not in skills_found and "python" not in missing_skills:
#         suggestions.append("Python is a highly sought-after skill. If you know it, explicitly mention it in your resume.")

#     if education:
#         suggestions.append(f"Your resume includes education details: {', '.join(education)}. Ensure they are formatted properly.")
#     else:
#         suggestions.append("Consider adding your educational background. Recruiters often look for degrees and certifications relevant to the job role.")

#     if experience:
#         suggestions.append("Your resume includes work experience. Ensure it's structured with bullet points highlighting key achievements.")
#     else:
#         suggestions.append("Adding relevant experience, even in the form of internships or projects, can make your resume stronger.")

#     return suggestions

# def improve_resume(resume_path):
#     """
#     Extracts resume details, fetches missing skills, and provides improvement suggestions.
#     """
#     # Parse resume using parse_resume.py
#     resume_id = parse_resume(resume_path)
    
#     if not resume_id:
#         print(" Error processing the resume.")
#         return None

#     # Fetch the resume from MongoDB
#     resume_entry = db.resumes.find_one({"_id": resume_id})
#     resume_text = resume_entry.get("resume_text", "")

#     # Extract skills, education, and experience
#     skills_found, education, experience = extract_entities(resume_text)

#     # Fetch missing skills from MongoDB
#     missing_skills = fetch_missing_skills()

#     # Generate resume improvement suggestions
#     resume_suggestions = generate_resume_improvements(skills_found, missing_skills, education, experience)

#     # Save resume improvements to MongoDB
#     improvements_data = {
#         "resume_id": resume_id,
#         "skills_found": skills_found,
#         "missing_skills": missing_skills,
#         "resume_suggestions": resume_suggestions
#     }

#     inserted_id = resume_improvements_collection.insert_one(improvements_data).inserted_id
#     print(f" Resume improvements saved in MongoDB with ID: {inserted_id}")

#     return improvements_data

# if __name__ == "__main__":
#     resume_path = "data/sample_resume.pdf"  # Replace with actual path
#     improve_resume(resume_path)

# import json
# import spacy
# import numpy as np
# from pymongo import MongoClient
# from sentence_transformers import SentenceTransformer
# from scripts.parse_resume import parse_resume

# # Load pre-trained NLP models
# nlp = spacy.load("en_core_web_sm")
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Connect to MongoDB
# client = MongoClient("mongodb://127.0.0.1:27017/")
# db = client["career_advisor"]
# skill_gaps_collection = db["skill_gaps"]
# resume_improvements_collection = db["resume_improvements"]

# # Mapping of skills to learning resources
# LEARNING_RESOURCES = {
#     "machine learning": ["https://www.coursera.org/learn/machine-learning", "https://www.udacity.com/course/intro-to-machine-learning--ud120"],
#     "deep learning": ["https://www.deeplearning.ai/courses/", "https://www.udacity.com/course/deep-learning-nanodegree--nd101"],
#     "mongodb": ["https://university.mongodb.com/", "https://www.udemy.com/course/mongodb-the-complete-developers-guide/"],
#     "pytorch": ["https://pytorch.org/tutorials/", "https://www.udacity.com/course/deep-learning-pytorch--ud188"],
#     "data visualization": ["https://www.tableau.com/learn/training", "https://www.udacity.com/course/data-visualization-nanodegree--nd197"],
#     "tensorflow": ["https://www.tensorflow.org/tutorials", "https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187"],
#     "data analysis": ["https://www.coursera.org/specializations/data-science-python", "https://www.kaggle.com/learn/data-analysis"],
#     "scikit-learn": ["https://scikit-learn.org/stable/tutorial/index.html", "https://www.datacamp.com/courses/supervised-learning-with-scikit-learn"],
#     "numpy": ["https://numpy.org/doc/stable/user/quickstart.html", "https://www.datacamp.com/courses/intro-to-python-for-data-science"],
#     "power bi": ["https://learn.microsoft.com/en-us/power-bi/", "https://www.udemy.com/course/microsoft-power-bi-up-running-with-power-bi-desktop/"],
#     "computer vision": ["https://www.coursera.org/specializations/computer-vision", "https://www.udacity.com/course/computer-vision-nanodegree--nd891"]
# }

# def extract_entities(text):
#     """Extract skills, education, and experience from the resume."""
#     doc = nlp(text)
#     skills, education, experience = set(), set(), set()
    
#     for ent in doc.ents:
#         if ent.label_ in ["ORG", "EDUCATION"]:
#             education.add(ent.text)
#         elif ent.label_ in ["WORK_OF_ART", "PRODUCT", "FAC"]:
#             skills.add(ent.text)
#         elif ent.label_ in ["DATE", "TIME"]:
#             experience.add(ent.text)
    
#     return list(skills), list(education), list(experience)

# def fetch_missing_skills():
#     """Fetch missing skills from MongoDB skill_gaps collection."""
#     skill_gap_entry = skill_gaps_collection.find_one({}, {"_id": 0, "missing_skills": 1})
#     return skill_gap_entry["missing_skills"] if skill_gap_entry else []

# def generate_resume_improvements(skills_found, missing_skills, education, experience, job_listings):
#     """Generate detailed resume improvement suggestions based on missing skills and job listings."""
#     suggestions = []

#     if len(skills_found) < 5:
#         suggestions.append("Your resume has fewer skills listed. Consider adding more to showcase your expertise.")

#     if missing_skills:
#         suggestions.append(f"You are missing key skills required for your desired roles: {', '.join(missing_skills)}. Consider learning these through online courses, certifications, or projects.")

#     if "Python" not in skills_found and "python" not in missing_skills:
#         suggestions.append("Python is a highly sought-after skill. If you know it, explicitly mention it in your resume.")

#     if education:
#         suggestions.append(f"Your resume includes education details: {', '.join(education)}. Ensure they are formatted properly.")
#     else:
#         suggestions.append("Consider adding your educational background. Recruiters often look for degrees and certifications relevant to the job role.")

#     if experience:
#         suggestions.append("Your resume includes work experience. Ensure it's structured with bullet points highlighting key achievements.")
#     else:
#         suggestions.append("Adding relevant experience, even in the form of internships or projects, can make your resume stronger.")
    
#     if job_listings:
#         top_skills = set()
#         for job in job_listings[:5]:  # Analyze top 5 job listings
#             top_skills.update(job.get("skills_required", []))
#         missing_from_jobs = top_skills.difference(set(skills_found))
#         if missing_from_jobs:
#             suggestions.append(f"Consider adding these in-demand skills based on job listings: {', '.join(missing_from_jobs)}.")
    
#     return suggestions

# def improve_resume(resume_text, extracted_data, job_listings, skill_gaps):
#     """
#     Improves the resume based on extracted data, job listings, and skill gaps.
#     """
#     skills_found = extracted_data.get("skills", [])
#     education = extracted_data.get("education", [])
#     experience = extracted_data.get("experience", [])
    
#     # Generate resume improvement suggestions
#     resume_suggestions = generate_resume_improvements(skills_found, skill_gaps, education, experience, job_listings)
    
#     # Include learning resources for missing skills
#     skill_gaps_with_resources = [{"skill": skill, "resources": LEARNING_RESOURCES.get(skill, [])} for skill in skill_gaps]
    
#     # Save resume improvements to MongoDB
#     improvements_data = {
#         "skills_found": skills_found,
#         "missing_skills": skill_gaps_with_resources,
#         "resume_suggestions": resume_suggestions
#     }
#     inserted_id = resume_improvements_collection.insert_one(improvements_data).inserted_id
#     print(f" Resume improvements saved in MongoDB with ID: {inserted_id}")
    
#     return resume_suggestions, skill_gaps_with_resources

# if __name__ == "__main__":
#     sample_text = "John Doe has experience in Python, SQL, and Machine Learning. He graduated from MIT with a degree in Computer Science."
#     extracted_sample = {
#         "skills": ["Python", "SQL", "Machine Learning"],
#         "education": ["MIT - Computer Science"],
#         "experience": ["Software Engineer at Google"]
#     }
#     job_listings_sample = [{"skills_required": ["Python", "Django", "SQL", "AWS"]}]
#     skill_gaps_sample = ["Django", "AWS"]
    
#     improve_resume(sample_text, extracted_sample, job_listings_sample, skill_gaps_sample)


import json
import spacy
import numpy as np
import requests
import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from scripts.parse_resume import parse_resume

# Load pre-trained NLP models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to MongoDB
client = MongoClient("mongodb://127.0.0.1:27017/")
db = client["career_advisor"]
skill_gaps_collection = db["skill_gaps"]
resume_improvements_collection = db["resume_improvements"]

# EdX and Kaggle API configurations
EDX_API_URL = "https://api.edx.org/catalog/v1/courses"
KAGGLE_API_URL = "https://www.kaggle.com/learn/courses"
EDX_API_KEY = os.getenv("EDX_API_KEY")  # Retrieve from environment variables

def fetch_edx_courses(skill):
    """Fetch online courses from EdX for a given skill."""
    if not EDX_API_KEY:
        print("EDX API key is missing. Set the EDX_API_KEY environment variable.")
        return []
    
    headers = {"Authorization": f"Bearer {EDX_API_KEY}"}
    params = {"search": skill}
    response = requests.get(EDX_API_URL, params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        courses = [course["marketing_url"] for course in data.get("results", [])[:3]]
        return courses
    return []

def fetch_kaggle_courses(skill):
    """Fetch learning resources from Kaggle Learn for a given skill."""
    kaggle_courses = {
        "machine learning": "https://www.kaggle.com/learn/intro-to-machine-learning",
        "deep learning": "https://www.kaggle.com/learn/deep-learning",
        "data visualization": "https://www.kaggle.com/learn/data-visualization",
        "data analysis": "https://www.kaggle.com/learn/pandas",
        "power bi": "https://www.kaggle.com/learn/courses",
        "scikit-learn": "https://www.kaggle.com/learn/courses",
        "mongodb": "https://www.kaggle.com/learn/courses",
        "computer vision": "https://www.kaggle.com/learn/courses",
        "numpy": "https://www.kaggle.com/learn/courses",
        "pytorch": "https://www.kaggle.com/learn/courses",
        "tensorflow": "https://www.kaggle.com/learn/courses"
    }
    return [kaggle_courses.get(skill.lower(), KAGGLE_API_URL)]

def get_learning_resources(skill):
    """Retrieve learning resources from predefined links or external APIs."""
    predefined_resources = fetch_edx_courses(skill) + fetch_kaggle_courses(skill)
    return predefined_resources

def generate_resume_improvements(skills_found, missing_skills, education, experience, job_listings):
    """Generate detailed resume improvement suggestions."""
    suggestions = []
    if len(skills_found) < 5:
        suggestions.append("Your resume has fewer skills listed. Consider adding more to showcase your expertise.")
    if missing_skills:
        suggestions.append(f"You are missing key skills required for your desired roles: {', '.join(missing_skills)}. Consider learning these through online courses, certifications, or projects.")
    if "Python" not in skills_found and "python" not in missing_skills:
        suggestions.append("Python is a highly sought-after skill. If you know it, explicitly mention it in your resume.")
    if education:
        suggestions.append(f"Your resume includes education details: {', '.join(education)}. Ensure they are formatted properly.")
    else:
        suggestions.append("Consider adding your educational background. Recruiters often look for degrees and certifications relevant to the job role.")
    if experience:
        suggestions.append("Your resume includes work experience. Ensure it's structured with bullet points highlighting key achievements.")
    else:
        suggestions.append("Adding relevant experience, even in the form of internships or projects, can make your resume stronger.")
    return suggestions

def improve_resume(resume_text, extracted_data, job_listings, skill_gaps):
    """
    Improves the resume based on extracted data, job listings, and skill gaps.
    """
    skills_found = extracted_data.get("skills", [])
    education = extracted_data.get("education", [])
    experience = extracted_data.get("experience", [])
    
    # Generate resume improvement suggestions
    resume_suggestions = generate_resume_improvements(skills_found, skill_gaps, education, experience, job_listings)
    
    # Include learning resources for missing skills with labeled output
    skill_gaps_with_resources = [{"skill": skill, "label": "Missing skill found in your resume", "resources": get_learning_resources(skill)} for skill in skill_gaps]
    
    # Save resume improvements to MongoDB
    improvements_data = {
        "skills_found": skills_found,
        "missing_skills": skill_gaps_with_resources,
        "resume_suggestions": resume_suggestions
    }
    inserted_id = resume_improvements_collection.insert_one(improvements_data).inserted_id
    print(f" Resume improvements saved in MongoDB with ID: {inserted_id}")
    
    return resume_suggestions, skill_gaps_with_resources

if __name__ == "__main__":
    sample_text = "John Doe has experience in Python, SQL, and Machine Learning. He graduated from MIT with a degree in Computer Science."
    extracted_sample = {
        "skills": ["Python", "SQL", "Machine Learning"],
        "education": ["MIT - Computer Science"],
        "experience": ["Software Engineer at Google"]
    }
    job_listings_sample = [{"skills_required": ["Python", "Django", "SQL", "AWS"]}]
    skill_gaps_sample = ["Django", "AWS"]
    
    improve_resume(sample_text, extracted_sample, job_listings_sample, skill_gaps_sample)
