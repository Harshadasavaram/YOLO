# import os
# import requests
# import json
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Get API credentials from environment variables
# API_URL = "https://api.adzuna.com/v1/api/jobs/in/search/1"  # India-specific Adzuna API URL
# APP_ID = os.getenv("APP_ID")
# APP_KEY = os.getenv("APP_KEY")

# def fetch_jobs_from_adzuna():
#     """Fetch job listings from the Adzuna API and save them to a JSON file."""
#     # Parameters for the API request
#     params = {
#         "app_id": APP_ID,
#         "app_key": APP_KEY,
#         "results_per_page": 100,#number of listings per page
#         "what": "software engineer",  # Search keywords
#         "where": "India",             # Location
#     } 
    
#     try:
#         # Make the API request
#         response = requests.get(API_URL, params=params)
#         response.raise_for_status()  # Raise an error for bad HTTP responses
        
#         # Print the raw response for debugging (optional)
#         #print(response.json())
        
#         # Parse the JSON response
#         jobs = response.json()
#         job_listings = [
#             {
#                 "title": job["title"],
#                 "company": job["company"]["display_name"],
#                 "location": job["location"]["display_name"],
#                 "description": job["description"],
#                 "url": job["redirect_url"]
#             }
#             for job in jobs.get("results", [])
#         ]

#         # Ensure the 'data' folder exists
#         os.makedirs("data", exist_ok=True)

#         # Save job listings to a JSON file
#         with open("data/job_listings.json", "w") as file:
#             json.dump(job_listings, file, indent=4)
        
#         print(f"Fetched {len(job_listings)} job listings and saved to 'data/job_listings.json'.")

#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching job listings: {e}")

# if __name__ == "__main__":
#     fetch_jobs_from_adzuna()
# import os
# import requests
# import json
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Get API credentials from .env file
# API_URL = "https://api.adzuna.com/v1/api/jobs/in/search/1"
# APP_ID = os.getenv("APP_ID")
# APP_KEY = o
# s.getenv("APP_KEY")

# def fetch_jobs():
#     """Fetch job listings from the Adzuna API and return them as a list of dictionaries."""
#     params = {
#         "app_id": APP_ID,
#         "app_key": APP_KEY,
#         "results_per_page": 100,  # Adjust as needed
#         "what": "software engineer",
#         "where": "India",
#     }

#     try:
#         response = requests.get(API_URL, params=params)
#         response.raise_for_status()  # Check for HTTP errors

#         jobs = response.json()
#         job_listings = [
#             {
#                 "title": job["title"],
#                 "company": job["company"]["display_name"],
#                 "location": job["location"]["display_name"],
#                 "description": job["description"],
#                 "url": job["redirect_url"]
#             }
#             for job in jobs.get("results", [])
#         ]

#         return job_listings  # Return job listings instead of writing to a file

#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching job listings: {e}")
#         return []

# if __name__ == "__main__":
#     jobs = fetch_jobs()
#     print(f"Fetched {len(jobs)} job listings.")
   
import os
import requests
import json
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Get API credentials from .env file
API_URL = "https://api.adzuna.com/v1/api/jobs/in/search/1"
APP_ID = os.getenv("APP_ID")
APP_KEY = os.getenv("APP_KEY")

# Connect to MongoDB
client = MongoClient("mongodb://127.0.0.1:27017/")
db = client["career_advisor"]
jobs_collection = db["jobs"]

def fetch_jobs():
    """Fetch job listings from the Adzuna API and store them in MongoDB."""
    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": 100,  # Adjust as needed
        "what": "software engineer",
        "where": "India",
    }

    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()  # Check for HTTP errors

        jobs = response.json().get("results", [])
        job_listings = []

        for job in jobs:
            job_entry = {
                "title": job["title"],
                "company": job["company"]["display_name"],
                "location": job["location"]["display_name"],
                "description": job["description"],
                "url": job["redirect_url"],
            }

            # Check if job already exists (to prevent duplicates)
            if not jobs_collection.find_one({"url": job_entry["url"]}):
                jobs_collection.insert_one(job_entry)
                job_listings.append(job_entry)

        print(f"Stored {len(job_listings)} new job listings in MongoDB.")
        return job_listings

    except requests.exceptions.RequestException as e:
        print(f"Error fetching job listings: {e}")
        return []

if __name__ == "__main__":
    jobs = fetch_jobs()
    print(f"Fetched and stored {len(jobs)} new job listings.")
