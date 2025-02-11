# from scripts.fetch_jobs import fetch_jobs_from_adzuna
# from scripts.parse_resume import parse_resume
# import json
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# def match_jobs_to_resume():
#     """
#     Match jobs fetched from the Adzuna API with skills from the parsed resume.
#     Saves matched jobs to 'data/matched_jobs.json'.
#     """
#     try:
#         # Step 1: Fetch job listings
#         logging.info("Fetching job listings from Adzuna API...")
#         fetch_jobs_from_adzuna()

#         # Step 2: Parse the resume
#         logging.info("Parsing the resume...")
#         resume_path = "data/sample_resume.pdf"
#         parse_resume(resume_path)

#         # Step 3: Load job listings
#         with open("data/job_listings.json", "r") as file:
#             job_listings = json.load(file)

#         # Step 4: Load parsed resume skills
#         with open("data/resume_parsed.txt", "r") as file:
#             resume_content = file.read()

#         # Step 5: Match jobs to skills in resume
#         logging.info("Matching jobs to resume...")
#         resume_skills = set(resume_content.lower().split())  # Tokenize and lowercase resume content
#         matched_jobs = [
#             job for job in job_listings if any(
#                 skill in job["description"].lower() for skill in resume_skills
#             )
#         ]

#         # Step 6: Save matched jobs
#         with open("data/matched_jobs.json", "w") as file:
#             json.dump(matched_jobs, file, indent=4)
        
#         logging.info(f"{len(matched_jobs)} jobs matched and saved to 'data/matched_jobs.json'.")

#     except FileNotFoundError as e:
#         logging.error(f"File not found: {e}")
#     except json.JSONDecodeError as e:
#         logging.error(f"Error decoding JSON: {e}")
#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {e}")

# if __name__ == "__main__":
#     match_jobs_to_resume()

# import os
# import json
# import spacy
# from dotenv import load_dotenv
# from scripts.parse_resume import parse_resume
# import scripts.match_jobs as job_matcher
# from scripts.improve_resume import improve_resume
# from scripts.skill_gap_analysis import analyze_skill_gaps
# from scripts.fetch_jobs import fetch_jobs
# from scripts.chatbot import chat_with_ai  # Conversational AI chatbot

# # Load environment variables
# load_dotenv()

# def main():
#     print("\nWelcome to AI-Driven Career Advisor")

#     while True:
#         print("\nMenu:")
#         print("1. Upload Resume & Get Job Matches")
#         print("2. Chat with AI Advisor")
#         print("3. Exit")

#         choice = input("\nEnter your choice (1/2/3): ")

#         if choice == "1":
#             # Step 1: Upload & Parse Resume
#             resume_path = input("\nEnter the path of your resume file (PDF or TXT): ").strip()
            
#             if not os.path.exists(resume_path):
#                 print("\nError: File not found! Please check the file path.")
#                 continue  # Restart loop to let user try again

#             resume_text, extracted_data = parse_resume(resume_path)

#             if resume_text is None or extracted_data is None:
#                 print("\nFailed to parse resume. Please check the file format and try again.")
#                 continue

#             print("\nExtracted Resume Data:", extracted_data)

#             # Step 2: Fetch Jobs
#             job_listings = fetch_jobs()  # Fetch jobs dynamically
#             if not job_listings:
#                 print("\nNo job listings found. Please try again later.")
#                 continue

#             print(f"\n{len(job_listings)} jobs fetched successfully.")

#             # Step 3: Match Resume with Jobs
#             job_matches = job_matcher.match_jobs(resume_text, job_listings)
#             print("\nTop Job Matches:")
#             print(job_matches.head(10))  # Show top 10 matches

#             # Step 4: Skill Gap Analysis
#             skill_gaps = analyze_skill_gaps(extracted_data["skills"], job_listings)
#             print("\nSkill Gap Analysis Suggestions:")
#             print(skill_gaps)

#             # Step 5: Resume Improvement Suggestions
#             resume_suggestions = improve_resume(extracted_data)
#             print("\nResume Improvement Suggestions:")
#             print(resume_suggestions)

#         elif choice == "2":
#             print("\nAI Career Advisor Chatbot")
#             while True:
#                 user_input = input("\nYou: ")
#                 if user_input.lower() in ["exit", "quit", "bye"]:
#                     print("Exiting chat. Have a great day!")
#                     break
#                 response = chat_with_ai(user_input)
#                 print("\nAI Advisor:", response)

#         elif choice == "3":
#             print("Exiting. Have a great day!")
#             break

#         else:
#             print("Invalid choice! Please try again.")

# if __name__ == "__main__":
#     main()

# import os
# import json
# from dotenv import load_dotenv
# from scripts.parse_resume import parse_resume
# import scripts.match_jobs as job_matcher
# from scripts.improve_resume import improve_resume
# from scripts.skill_gap_analysis import analyze_skill_gaps
# from scripts.fetch_jobs import fetch_jobs
# from scripts.chatbot import chat_with_ai  

# # Load environment variables
# load_dotenv()

# def save_json(filename, data):
#     """Utility function to append or update JSON data."""
#     if os.path.exists(filename):
#         with open(filename, "r") as file:
#             existing_data = json.load(file)
#     else:
#         existing_data = []

#     existing_data.append(data)  # Append new entry
#     with open(filename, "w") as file:
#         json.dump(existing_data, file, indent=4)

# def load_json(filename):
#     """Utility function to load JSON data."""
#     if os.path.exists(filename):
#         with open(filename, "r") as file:
#             return json.load(file)
#     return None

# def main():
#     """Main function for AI-Driven Career Advisor."""
#     print("\nWelcome to AI-Driven Career Advisor")

#     while True:
#         print("\nMenu:")
#         print("1. Upload Resume & Get Job Matches")
#         print("2. Chat with AI Advisor")
#         print("3. View Previous Results")
#         print("4. Exit")

#         choice = input("\nEnter your choice (1/2/3/4): ").strip()

#         if choice == "1":
#             # Step 1: Upload & Parse Resume
#             resume_path = input("\nEnter the path of your resume file (PDF or TXT): ").strip()
            
#             if not os.path.exists(resume_path):
#                 print("\nError: File not found! Please enter a valid file path.")
#                 continue  

#             resume_text, extracted_data = parse_resume(resume_path)

#             if not resume_text or not extracted_data:
#                 print("\nFailed to parse resume. Please check the file format and try again.")
#                 continue

#             print("\nExtracted Resume Data:", json.dumps(extracted_data, indent=4))

#             # Step 2: Fetch Job Listings
#             job_listings = fetch_jobs()  
#             if not job_listings:
#                 print("\nNo job listings found. Please try again later.")
#                 continue

#             print(f"\n{len(job_listings)} job listings fetched successfully.")

#             # Step 3: Match Resume with Jobs
#             job_matches = job_matcher.match_jobs(resume_text, job_listings)

#             print("\nTop Job Matches:")
#             print(job_matches.head(10))  # Show top 10 matches

#             # Step 4: Skill Gap Analysis
#             skill_gaps = analyze_skill_gaps(extracted_data["skills"], job_listings)

#             print("\nSkill Gap Analysis:")
#             for gap in skill_gaps[:5]:  # Show top 5 skill gaps
#                 print(f"- {gap['title']} (Missing Skills: {', '.join([s['skill'] for s in gap['missing_skills']])})")

#             # Step 5: Resume Improvement Suggestions
#             resume_suggestions = improve_resume(resume_text, extracted_data, job_listings, skill_gaps)

#             print("\nResume Improvement Suggestions:")
#             for suggestion in resume_suggestions:
#                 print(f"- {suggestion}")

#             # Save results for future access
#             result_data = {
#                 "resume_data": extracted_data,
#                 "job_matches": job_matches.head(10).to_dict(orient="records"),
#                 "skill_gaps": skill_gaps,
#                 "resume_suggestions": resume_suggestions
#             }
#             save_json("data/results.json", result_data)

#         elif choice == "2":
#             print("\nAI Career Advisor Chatbot (Type 'exit' to quit chat)")

#             while True:
#                 user_input = input("\nYou: ").strip()
#                 if user_input.lower() in ["exit", "quit", "bye"]:
#                     print("\nExiting chat. Have a great day!")
#                     break

#                 response = chat_with_ai(user_input)
#                 if response:
#                     print("\nAI Advisor:", response)
#                 else:
#                     print("\nAI Advisor is unable to generate a response. Try asking differently!")

#         elif choice == "3":
#             print("\nViewing Previous Results:")
#             past_results = load_json("data/results.json")
#             if past_results:
#                 print(json.dumps(past_results, indent=4))
#             else:
#                 print("\nNo previous results found.")

#         elif choice == "4":
#             print("\nExiting AI-Driven Career Advisor. Have a great day!")
#             break

#         else:
#             print("\nInvalid choice! Please enter 1, 2, 3, or 4.")

# if __name__ == "__main__":
#     main()



# import os
# import json
# from dotenv import load_dotenv
# from pymongo import MongoClient
# from scripts.parse_resume import parse_resume
# from scripts.match_jobs import match_jobs
# from scripts.improve_resume import improve_resume
# from scripts.skill_gap_analysis import analyze_skill_gap
# from scripts.fetch_jobs import fetch_jobs
# from scripts.chatbot import chat_with_ai

# # Load environment variables
# load_dotenv()

# # MongoDB Connection
# client = MongoClient("mongodb://127.0.0.1:27017/")
# db = client["career_advisor"]
# skills_collection = db["skills"]

# def save_json(filename, data):
#     """Utility function to save JSON data."""
#     os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure directory exists
#     if os.path.exists(filename):
#         with open(filename, "r") as file:
#             existing_data = json.load(file)
#     else:
#         existing_data = []

#     existing_data.append(data)  # Append new entry
#     with open(filename, "w") as file:
#         json.dump(existing_data, file, indent=4)

# def load_json(filename):
#     """Utility function to load JSON data."""
#     if os.path.exists(filename):
#         with open(filename, "r") as file:
#             return json.load(file)
#     return None

# def upload_resume_and_match_jobs():
#     """Handles resume upload, parsing, job fetching, and matching."""
#     resume_path = input("\nEnter the path of your resume file (PDF or TXT): ").strip()

#     if not os.path.exists(resume_path):
#         print("\nError: File not found! Please enter a valid file path.")
#         return  

#     # Parse Resume
#     parsed_data = parse_resume(resume_path)

#     if not parsed_data:
#         print("\nFailed to parse resume. Please check the file format and try again.")
#         return

#     resume_text = parsed_data.get("resume_text", "")
#     extracted_data = parsed_data.get("structured_data", {})

#     print("\nExtracted Resume Data:")
#     print(json.dumps(extracted_data, indent=4))

#     # Fetch Job Listings
#     job_listings = fetch_jobs()
#     if not job_listings:
#         print("\nNo job listings found. Please try again later.")
#         return

#     print(f"\n{len(job_listings)} job listings fetched successfully.")

#     # Match Resume with Jobs
#     job_matches = match_jobs(resume_text, job_listings)

#     print("\nTop Job Matches:")
#     print(job_matches.head(10))  # Show top 10 matches

#     # Skill Gap Analysis
#     extracted_skills = extracted_data.get("skills", [])
#     if not extracted_skills:
#         print("\nNo skills extracted from resume.")
#         skill_gaps = []
#     else:
#         # skill_gaps = analyze_skill_gap(extracted_skills)
#         skill_gaps = analyze_skill_gap(resume_path)  # Pass resume_path instead of extracted_skills

#         print("\nSkill Gap Analysis:")
#         for gap in skill_gaps[:5]:  # Show top 5 skill gaps
#             print(f"- {gap}")

#     # Resume Improvement Suggestions
#     resume_suggestions = improve_resume(resume_text, extracted_data, job_listings, skill_gaps)

#     print("\nResume Improvement Suggestions:")
#     for suggestion in resume_suggestions:
#         print(f"- {suggestion}")

#     # Save results for future access
#     result_data = {
#         "resume_data": extracted_data,
#         "job_matches": job_matches.head(10).to_dict(orient="records"),
#         "skill_gaps": skill_gaps,
#         "resume_suggestions": resume_suggestions
#     }
#     save_json("data/results.json", result_data)

# def chat_with_advisor():
#     """Handles AI-based chatbot interactions."""
#     print("\nAI Career Advisor Chatbot (Type 'exit' to quit chat)")

#     while True:
#         user_input = input("\nYou: ").strip()
#         if user_input.lower() in ["exit", "quit", "bye"]:
#             print("\nExiting chat. Have a great day!")
#             break

#         response = chat_with_ai(user_input)
#         if response:
#             print("\nAI Advisor:", response)
#         else:
#             print("\nAI Advisor is unable to generate a response. Try asking differently!")

# def view_previous_results():
#     """Displays previously saved results."""
#     print("\nViewing Previous Results:")
#     past_results = load_json("data/results.json")
#     if past_results:
#         print(json.dumps(past_results, indent=4))
#     else:
#         print("\nNo previous results found.")

# def main():
#     """Main function for AI-Driven Career Advisor."""
#     print("\nWelcome to AI-Driven Career Advisor")

#     while True:
#         print("\nMenu:")
#         print("1. Upload Resume & Get Job Matches")
#         print("2. Chat with AI Advisor")
#         print("3. View Previous Results")
#         print("4. Exit")

#         choice = input("\nEnter your choice (1/2/3/4): ").strip()

#         if choice == "1":
#             upload_resume_and_match_jobs()
#         elif choice == "2":
#             chat_with_advisor()
#         elif choice == "3":
#             view_previous_results()
#         elif choice == "4":
#             print("\nExiting AI-Driven Career Advisor. Have a great day!")
#             break
#         else:
#             print("\nInvalid choice! Please enter 1, 2, 3, or 4.")

# if __name__ == "__main__":
#     main()

import os
import json
from dotenv import load_dotenv
from pymongo import MongoClient
from scripts.parse_resume import parse_resume
from scripts.match_jobs import match_jobs
from scripts.improve_resume import improve_resume
from scripts.skill_gap_analysis import analyze_skill_gap
from scripts.fetch_jobs import fetch_jobs
from scripts.chatbot import chat_with_ai

# Load environment variables
load_dotenv()

# MongoDB Connection
client = MongoClient("mongodb://127.0.0.1:27017/")
db = client["career_advisor"]
skills_collection = db["skills"]

def save_json(filename, data):
    """Utility function to save JSON data."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure directory exists
    if os.path.exists(filename):
        with open(filename, "r") as file:
            existing_data = json.load(file)
    else:
        existing_data = []

    existing_data.append(data)  # Append new entry
    with open(filename, "w") as file:
        json.dump(existing_data, file, indent=4)

def load_json(filename):
    """Utility function to load JSON data."""
    if os.path.exists(filename):
        with open(filename, "r") as file:
            return json.load(file)
    return None

def upload_resume_and_match_jobs():
    """Handles resume upload, parsing, job fetching, and matching."""
    resume_path = input("\nEnter the path of your resume file (PDF or TXT): ").strip()

    if not os.path.exists(resume_path):
        print("\nError: File not found! Please enter a valid file path.")
        return  

    # Parse Resume
    parsed_data = parse_resume(resume_path)

    if not parsed_data:
        print("\nFailed to parse resume. Please check the file format and try again.")
        return

    resume_text = parsed_data.get("resume_text", "")
    extracted_data = parsed_data.get("structured_data", {})

    print("\nExtracted Resume Data:")
    print(json.dumps(extracted_data, indent=4))

    # Fetch Job Listings
    job_listings = fetch_jobs()
    if not job_listings:
        print("\nNo job listings found. Please try again later.")
        return

    print(f"\n{len(job_listings)} job listings fetched successfully.")

    # Match Resume with Jobs
    job_matches = match_jobs(resume_text, job_listings)

    print("\nTop Job Matches:")
    print(job_matches.head(10))  # Show top 10 matches

    # Skill Gap Analysis
    extracted_skills = extracted_data.get("skills", [])
    if not extracted_skills:
        print("\nNo skills extracted from resume.")
        skill_gaps = []
    else:
        skill_gaps = analyze_skill_gap(resume_path)  # Pass resume_path instead of extracted_skills

        print("\nSkill Gap Analysis:")
        for gap in skill_gaps[:5]:  # Show top 5 skill gaps
            print(f"- {gap}")

    # Resume Improvement Suggestions
    resume_suggestions = improve_resume(resume_text, extracted_data, job_listings, skill_gaps)

    print("\nResume Improvement Suggestions:")
    for suggestion in resume_suggestions:
        print(f"- {suggestion}")

    # Save results for future access
    result_data = {
        "resume_data": extracted_data,
        "job_matches": job_matches.head(10).to_dict(orient="records"),
        "skill_gaps": skill_gaps,
        "resume_suggestions": resume_suggestions
    }
    save_json("data/results.json", result_data)

def chat_with_advisor():
    """Handles AI-based chatbot interactions."""
    print("\nAI Career Advisor Chatbot (Type 'exit' to quit chat)")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nExiting chat. Have a great day!")
            break

        response = chat_with_ai(user_input)
        if response:
            print("\nAI Advisor:", response)
        else:
            print("\nAI Advisor is unable to generate a response. Try asking differently!")

def view_previous_results():
    """Displays previously saved results."""
    print("\nViewing Previous Results:")
    past_results = load_json("data/results.json")
    if past_results:
        print(json.dumps(past_results, indent=4))
    else:
        print("\nNo previous results found.")

def main():
    """Main function for AI-Driven Career Advisor."""
    print("\nWelcome to AI-Driven Career Advisor")

    while True:
        print("\nMenu:")
        print("1. Upload Resume & Get Job Matches")
        print("2. Chat with AI Advisor")
        print("3. View Previous Results")
        print("4. Exit")

        choice = input("\nEnter your choice (1/2/3/4): ").strip()

        if choice == "1":
            upload_resume_and_match_jobs()
        elif choice == "2":
            chat_with_advisor()
        elif choice == "3":
            view_previous_results()
        elif choice == "4":
            print("\nExiting AI-Driven Career Advisor. Have a great day!")
            break
        else:
            print("\nInvalid choice! Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
