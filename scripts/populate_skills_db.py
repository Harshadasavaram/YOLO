import json
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["career_advisor"]  # Database name
collection = db["skills"]       # Collection name

# Load skills data from JSON
with open("data/skills_database.json", "r") as file:
    skills_data = json.load(file)

# Insert data into MongoDB
collection.delete_many({})  # Clear existing data
collection.insert_one(skills_data)

print("Skills database has been populated successfully!")

