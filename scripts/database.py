# from pymongo import MongoClient

# # Connect to MongoDB
# client = MongoClient("mongodb://127.0.0.1:27017/")
# db = client["career_advisor"]  

# # Define collections
# users_collection = db["users"]
# jobs_collection = db["jobs"]
# resumes_collection = db["resumes"]

# # Sample Resume Data
# sample_resume = {
#     "name": "John Doe",
#     "email": "johndoe@example.com",
#     "skills": ["Python", "Machine Learning", "SQL"],
#     "experience": 3,
#     "education": "Bachelor's in Computer Science"
# }

# # Insert into MongoDB
# resume_id = resumes_collection.insert_one(sample_resume).inserted_id
# print(f"Resume inserted with ID: {resume_id}")

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# MongoDB connection settings
MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME = "career_advisor"

def get_db():
    """Establish a connection to the MongoDB database and return the database object."""
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        print("✅ Connected to MongoDB successfully!")
        return db
    except ConnectionFailure as e:
        print(f"⚠️ MongoDB connection failed: {e}")
        return None

# Initialize database
db = get_db()

# Define collections (ensure db is not None)
if db:
    users_collection = db["users"]
    jobs_collection = db["jobs"]
    resumes_collection = db["resumes"]

# Helper functions
def insert_resume(resume_data):
    """Insert a resume document into the 'resumes' collection."""
    if db:
        resume_id = resumes_collection.insert_one(resume_data).inserted_id
        print(f"✅ Resume inserted with ID: {resume_id}")
        return resume_id
    return None

def get_all_resumes():
    """Retrieve all resumes from the database."""
    if db:
        return list(resumes_collection.find())
    return []

def insert_job(job_data):
    """Insert a job document into the 'jobs' collection."""
    if db:
        job_id = jobs_collection.insert_one(job_data).inserted_id
        print(f"✅ Job inserted with ID: {job_id}")
        return job_id
    return None

def get_all_jobs():
    """Retrieve all job listings from the database."""
    if db:
        return list(jobs_collection.find())
    return []

# Sample usage
if __name__ == "__main__":
    # Sample Resume Data
    sample_resume = {
        "name": "John Doe",
        "email": "johndoe@example.com",
        "skills": ["Python", "Machine Learning", "SQL"],
        "experience": 3,
        "education": "Bachelor's in Computer Science"
    }
    
    # Insert sample resume
    insert_resume(sample_resume)

    # Fetch all resumes
    resumes = get_all_resumes()
    print("📂 Resumes in DB:", resumes)
