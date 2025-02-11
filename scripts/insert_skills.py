from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://127.0.0.1:27017/")
db = client["career_advisor"]
skills_collection = db["skills"]

# Predefined skills list
skills_list = [
    "python", "machine learning", "data analysis", "deep learning", "nlp",
    "sql", "mongodb", "tensorflow", "pandas", "numpy", "scikit-learn",
    "pytorch", "computer vision", "excel", "tableau", "power bi", "data visualization"
]

# Insert skills into MongoDB
for skill in skills_list:
    skills_collection.insert_one({"skill": skill.lower()})  # Store in lowercase for consistency

print("✅ Skills inserted successfully!")
