# from pdfminer.high_level import extract_text
# import os

# def parse_resume(file_path):
#     """Parse a resume PDF and extract its text."""
#     if os.path.exists(file_path):
#         try:
#             return extract_text(file_path)
#         except Exception as e:
#             print(f"Error parsing resume: {e}")
#             return None
#     else:
#         print("File not found.")
#         return None

# if __name__ == "__main__":
#     resume_path = "data/sample_resume.pdf"
#   # Replace with your sample file path
#     parsed_text = parse_resume(resume_path)
#     if parsed_text:
#         with open("data/resume_parsed.txt", "w") as file:
#             file.write(parsed_text)
#         print("Parsed resume saved to 'data/resume_parsed.txt'")


# from pdfminer.high_level import extract_text
# import os

# def parse_resume(file_path):
#     """Parse a resume PDF and extract its text and structured data."""
#     if not os.path.exists(file_path):
#         print("Error: File not found.")
#         return None, None

#     try:
#         resume_text = extract_text(file_path)
        
#         # Placeholder for extracted structured data
#         extracted_data = {
#             "skills": ["Python", "Machine Learning"],  # Dummy skills (Enhance later with NLP)
#             "experience": "2 years",  # Dummy experience
#         }

#         return resume_text, extracted_data  # Returning both resume text and structured data

#     except Exception as e:
#         print(f"Error parsing resume: {e}")
#         return None, None  # Ensures it always returns two values

# if __name__ == "__main__":
#     resume_path = "data/sample_resume.pdf"  # Replace with your actual resume file path
#     resume_text, extracted_data = parse_resume(resume_path)

#     if resume_text:
#         with open("data/resume_parsed.txt", "w") as file:
#             file.write(resume_text)
#         print("Parsed resume saved to 'data/resume_parsed.txt'.")

# from pdfminer.high_level import extract_text
# from pymongo import MongoClient
# import spacy
# import os
# import re

# # Load SpaCy NLP model
# nlp = spacy.load("en_core_web_sm")

# # Connect to MongoDB
# client = MongoClient("mongodb://127.0.0.1:27017/")
# db = client["career_advisor"]
# resumes_collection = db["resumes"]

# # Sample skill keywords (expand this list for better results)
# SKILL_KEYWORDS = {"python", "machine learning", "sql", "tensorflow", "keras", "pandas", "numpy", "data analysis"}

# def extract_email(text):
#     """Extracts email from text using regex."""
#     match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
#     return match.group() if match else "Unknown"

# def extract_skills(text):
#     """Extracts skills using NLP and predefined skill keywords."""
#     doc = nlp(text.lower())
#     skills = {token.text for token in doc if token.text in SKILL_KEYWORDS}
#     return list(skills)

# def parse_resume(file_path):
#     """Parse a resume PDF, extract text, and store structured data in MongoDB."""
#     if not os.path.exists(file_path):
#         print("Error: File not found.")
#         return None

#     try:
#         resume_text = extract_text(file_path)

#         # Extract structured data
#         extracted_data = {
#             "name": "Unknown",  # Can be improved with Named Entity Recognition
#             "email": extract_email(resume_text),
#             "skills": extract_skills(resume_text),
#             "experience": "Unknown"  # Can be enhanced with regex parsing
#         }

#         # Store in MongoDB
#         resume_data = {
#             "resume_text": resume_text,
#             "structured_data": extracted_data
#         }
#         resume_id = resumes_collection.insert_one(resume_data).inserted_id

#         print(f"Resume successfully stored in MongoDB with ID: {resume_id}")
#         return resume_id

#     except Exception as e:
#         print(f"Error parsing resume: {e}")
#         return None

# if __name__ == "__main__":
#     resume_path = "data/sample_resume.pdf"  # Replace with actual path
#     resume_id = parse_resume(resume_path)

from pdfminer.high_level import extract_text
from pymongo import MongoClient
import spacy
import os
import re

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Connect to MongoDB
client = MongoClient("mongodb://127.0.0.1:27017/")
db = client["career_advisor"]
resumes_collection = db["resumes"]

# Sample skill keywords (expand this list for better results)
SKILL_KEYWORDS = {
    "python", "machine learning", "sql", "tensorflow", "keras", "pandas", "numpy", 
    "data analysis", "java", "deep learning", "cloud computing", "javascript", "react", 
    "docker", "kubernetes", "git", "linux", "aws", "c++", "data structures", "algorithms", 
    "natural language processing", "computer vision", "cybersecurity", "devops", "pytorch",
    "big data", "hadoop", "spark", "tableau", "power bi", "flutter", "android development", 
    "ios development", "blockchain", "ethical hacking", "networking", "microservices", 
    "REST APIs", "ui/ux design", "web design", "video editing"
}

def extract_email(text):
    """Extracts email from text using regex."""
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group() if match else "Unknown"

def extract_skills(text):
    """Extracts skills using NLP and predefined skill keywords."""
    doc = nlp(text.lower())
    skills = {token.text for token in doc if token.text in SKILL_KEYWORDS}
    return list(skills)

def parse_resume(file_path, store_in_db=True):
    """Parse a resume PDF, extract structured data, and optionally store it in MongoDB."""
    if not os.path.exists(file_path):
        print("Error: File not found.")
        return None

    try:
        resume_text = extract_text(file_path)

        # Extract structured data
        extracted_data = {
            "name": "Unknown",  # Can be improved with Named Entity Recognition
            "email": extract_email(resume_text),
            "skills": extract_skills(resume_text),
            "experience": "Unknown"  # Can be enhanced with regex parsing
        }

        resume_entry = {
            "resume_text": resume_text,
            "structured_data": extracted_data
        }

        if store_in_db:
            # Store in MongoDB
            resume_id = resumes_collection.insert_one(resume_entry).inserted_id
            print(f"Resume successfully stored in MongoDB with ID: {resume_id}")

        return resume_entry  # Ensure it returns a dictionary, not a tuple

    except Exception as e:
        print(f"Error parsing resume: {e}")
        return None

if __name__ == "__main__":
    resume_path = "data/sample_resume.pdf"  # Replace with actual path
    extracted_data = parse_resume(resume_path)

    if extracted_data:
        print("\nExtracted Data:", extracted_data)

