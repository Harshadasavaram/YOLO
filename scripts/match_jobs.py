# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import json
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # Load pre-trained SentenceTransformer model
# model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')


# def match_jobs():
#     """
#     Match resume content with job descriptions based on semantic similarity using SentenceTransformer.
#     """
#     try:
#         # Step 1: Load job listings
#         logging.info("Loading job listings...")
#         with open("data/job_listings.json", "r") as file:
#             job_listings = json.load(file)

#         # Step 2: Load parsed resume
#         logging.info("Loading parsed resume...")
#         with open("data/resume_parsed.txt", "r") as file:
#             resume_content = file.read()

#         # Step 3: Generate embeddings
#         logging.info("Generating embeddings...")
#         resume_embedding = model.encode(resume_content)  # Resume embedding
#         job_embeddings = [model.encode(job["description"]) for job in job_listings]  # Job embeddings

#         # Step 4: Compute cosine similarity
#         logging.info("Computing cosine similarities...")
#         similarities = cosine_similarity([resume_embedding], job_embeddings)

       
#         # Step 5: Filter matched jobs based on threshold
#         threshold = 0.7
#         matched_jobs = [
#             {
#                 "title": job["title"],
#                 "company": job.get("company", "Unknown"),
#                 "location": job.get("location", "Unknown"),
#                 "url": job.get("url", "N/A"),
#                 "similarity_score": float(similarities[0][idx])  # Convert to float
#             }
#             for idx, job in enumerate(job_listings)
#             if similarities[0][idx] > threshold
#         ]


#         # Step 6: Save matched jobs to a new file
#         with open("data/matched_jobs.json", "w") as file:
#             json.dump(matched_jobs, file, indent=4)

#         logging.info(f"Found {len(matched_jobs)} matching jobs. Results saved to 'data/matched_jobs.json'.")

#     except FileNotFoundError as e:
#         logging.error(f"File not found: {e}")
#     except json.JSONDecodeError as e:
#         logging.error(f"Error decoding JSON: {e}")
#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {e}")

# if __name__ == "__main__":
#     match_jobs()

#second version

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sentence_transformers import SentenceTransformer, util

# # Load the pre-trained model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Sample resume and job descriptions
# resume_text = "Experienced software engineer skilled in Python, machine learning, and web development."

# job_descriptions = {
#     "Software Engineer": "Looking for a software engineer proficient in Python and machine learning.",
#     "Software Engineer - Frontend": "Frontend engineer with React, JavaScript, and UI/UX experience.",
#     "Software Engineer - Java": "Seeking a Java software engineer with Spring Boot experience.",
#     "Software Engineer II": "Software engineer with expertise in backend development and databases.",
#     "Lead Software Engineer": "Lead engineer role requiring expertise in team management and architecture.",
#     "Senior Software Engineer": "Senior software engineer experienced in cloud computing and AI.",
#     "Software Engineer - DevOps": "DevOps engineer with experience in CI/CD and cloud deployment.",
#     "Software Engineer - Javascript": "Full-stack engineer with a strong JavaScript background."
# }

# # Compute similarity scores
# job_titles = list(job_descriptions.keys())
# job_texts = list(job_descriptions.values())

# resume_embedding = model.encode(resume_text, convert_to_tensor=True)
# job_embeddings = model.encode(job_texts, convert_to_tensor=True)

# # Compute cosine similarity
# cosine_similarities = util.pytorch_cos_sim(resume_embedding, job_embeddings)[0].cpu().numpy()

# # Create DataFrame with results
# job_df = pd.DataFrame({
#     "Job Title": job_titles,
#     "Cosine Similarity": cosine_similarities
# })

# # Sort by similarity in descending order
# job_df = job_df.sort_values(by="Cosine Similarity", ascending=False).reset_index(drop=True)

# # Set up the plot
# fig, ax = plt.subplots(figsize=(10, 6))
# bar_width = 0.5  # Adjust bar width for spacing
# y_positions = np.arange(len(job_df))  # Get positions for bars

# # Plot bars
# bars = ax.barh(y_positions, job_df["Cosine Similarity"], color='skyblue', edgecolor='black', height=bar_width)

# # Add similarity scores as text
# for bar, score in zip(bars, job_df["Cosine Similarity"]):
#     ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{score:.2f}', va='center', fontsize=10, fontweight='bold')

# # Formatting the plot
# ax.set_yticks(y_positions)
# ax.set_yticklabels(job_df["Job Title"], fontsize=10)
# ax.set_xlabel("Similarity Score", fontsize=12)
# ax.set_title("Comparison of Similarity Scores for Matched Jobs", fontsize=14, fontweight="bold")

# # Invert y-axis so highest similarity is at the top
# ax.invert_yaxis()
# ax.grid(axis='x', linestyle="--", alpha=0.6)
# ax.set_xlim(0, 1)  # Set limit for similarity scores

# # Prevent job titles from getting cut off
# plt.subplots_adjust(left=0.3)  # Adjust left margin
# plt.tight_layout()  # Automatically adjust layout

# # Show the plot
# plt.show()

# # Print the sorted job matches
# print(job_df)

#3rd version

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sentence_transformers import SentenceTransformer, util
# import seaborn as sns

# # Load the pre-trained model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# def match_jobs(resume_text, job_descriptions):
#     """
#     Matches the given resume text with job descriptions using cosine similarity.

#     Args:
#         resume_text (str): The text content of the resume.
#         job_descriptions (dict): Dictionary containing job titles and descriptions.

#     Returns:
#         pd.DataFrame: Sorted DataFrame with job titles and similarity scores.
#     """

#     # Compute similarity scores
#     job_titles = list(job_descriptions.keys())
#     job_texts = list(job_descriptions.values())

#     resume_embedding = model.encode(resume_text, convert_to_tensor=True)
#     job_embeddings = model.encode(job_texts, convert_to_tensor=True)

#     # Compute cosine similarity
#     cosine_similarities = util.pytorch_cos_sim(resume_embedding, job_embeddings)[0].cpu().numpy()

#     # Create DataFrame with results
#     job_df = pd.DataFrame({
#         "Job Title": job_titles,
#         "Cosine Similarity": cosine_similarities
#     })

#     # Sort by similarity in descending order
#     job_df = job_df.sort_values(by="Cosine Similarity", ascending=False).reset_index(drop=True)

#     return job_df  # Return DataFrame instead of printing

# # Ensure this part runs only when executing directly, not when imported
# if __name__ == "__main__":
#     # Sample data for testing
#     resume_text = "Experienced software engineer skilled in Python, machine learning, and web development."
#     job_descriptions = {
#         "Software Engineer": "Looking for a software engineer proficient in Python and machine learning.",
#         "Software Engineer - Frontend": "Frontend engineer with React, JavaScript, and UI/UX experience.",
#         "Software Engineer - Java": "Seeking a Java software engineer with Spring Boot experience.",
#         "Software Engineer II": "Software engineer with expertise in backend development and databases.",
#         "Lead Software Engineer": "Lead engineer role requiring expertise in team management and architecture.",
#         "Senior Software Engineer": "Senior software engineer experienced in cloud computing and AI.",
#         "Software Engineer - DevOps": "DevOps engineer with experience in CI/CD and cloud deployment.",
#         "Software Engineer - Javascript": "Full-stack engineer with a strong JavaScript background."
#     }

#     job_df = match_jobs(resume_text, job_descriptions)

#     # Plot results
#     fig, ax = plt.subplots(figsize=(12, 6))
#     y_positions = np.arange(len(job_df))
#     colors = sns.color_palette("coolwarm", len(job_df))

#     bars = ax.barh(y_positions, job_df["Cosine Similarity"], color=colors, edgecolor='black')

#     for bar, score in zip(bars, job_df["Cosine Similarity"]):
#         ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f'{score:.2f}', 
#                 va='center', fontsize=10, fontweight='bold', color='black')

#     ax.set_yticks(y_positions)
#     ax.set_yticklabels(job_df["Job Title"], fontsize=12, fontweight='bold')
#     ax.set_xlabel("Similarity Score", fontsize=12, fontweight='bold')
#     ax.set_title("🔥 Job Match Similarity Scores 🔥", fontsize=14, fontweight="bold", color="darkred")
#     ax.invert_yaxis()
#     ax.grid(axis='x', linestyle="--", alpha=0.6)
#     ax.set_xlim(0, 1)

#     plt.subplots_adjust(left=0.3)
#     plt.tight_layout()
#     plt.show()

#     print(job_df)


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sentence_transformers import SentenceTransformer, util
# import seaborn as sns

# # Load the pre-trained model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# def match_jobs(resume_text, job_listings):
#     """
#     Matches the given resume text with job descriptions using cosine similarity.

#     Args:
#         resume_text (str): The text content of the resume.
#         job_listings (list): List of dictionaries, each containing job "title" and "description".

#     Returns:
#         pd.DataFrame: Sorted DataFrame with job titles and similarity scores.
#     """

#     if not job_listings:
#         print("No job listings available for matching.")
#         return pd.DataFrame(columns=["Job Title", "Cosine Similarity"])

#     # Extract job titles and descriptions from the list of dictionaries
#     job_titles = [job["title"] for job in job_listings]
#     job_texts = [job["description"] for job in job_listings]

#     # Compute embeddings
#     resume_embedding = model.encode(resume_text, convert_to_tensor=True)
#     job_embeddings = model.encode(job_texts, convert_to_tensor=True)

#     # Compute cosine similarity
#     cosine_similarities = util.pytorch_cos_sim(resume_embedding, job_embeddings)[0].cpu().numpy()

#     # Create DataFrame with results
#     job_df = pd.DataFrame({
#         "Job Title": job_titles,
#         "Cosine Similarity": cosine_similarities
#     })

#     # Sort by similarity in descending order
#     job_df = job_df.sort_values(by="Cosine Similarity", ascending=False).reset_index(drop=True)

#     return job_df  # Return DataFrame instead of printing

# # Ensure this part runs only when executing directly, not when imported
# if __name__ == "__main__":
#     # Sample resume text
#     resume_text = "Experienced software engineer skilled in Python, machine learning, and web development."

#     # Sample job listings (list of dictionaries)
#     job_listings = [
#         {"title": "Software Engineer", "description": "Looking for a software engineer proficient in Python and machine learning."},
#         {"title": "Software Engineer - Frontend", "description": "Frontend engineer with React, JavaScript, and UI/UX experience."},
#         {"title": "Software Engineer - Java", "description": "Seeking a Java software engineer with Spring Boot experience."},
#         {"title": "Software Engineer II", "description": "Software engineer with expertise in backend development and databases."},
#         {"title": "Lead Software Engineer", "description": "Lead engineer role requiring expertise in team management and architecture."},
#         {"title": "Senior Software Engineer", "description": "Senior software engineer experienced in cloud computing and AI."},
#         {"title": "Software Engineer - DevOps", "description": "DevOps engineer with experience in CI/CD and cloud deployment."},
#         {"title": "Software Engineer - Javascript", "description": "Full-stack engineer with a strong JavaScript background."}
#     ]

#     # Match jobs
#     job_df = match_jobs(resume_text, job_listings)

#     # Plot results
#     fig, ax = plt.subplots(figsize=(12, 6))
#     y_positions = np.arange(len(job_df))
#     colors = sns.color_palette("coolwarm", len(job_df))

#     bars = ax.barh(y_positions, job_df["Cosine Similarity"], color=colors, edgecolor='black')

#     for bar, score in zip(bars, job_df["Cosine Similarity"]):
#         ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f'{score:.2f}', 
#                 va='center', fontsize=10, fontweight='bold', color='black')

#     ax.set_yticks(y_positions)
#     ax.set_yticklabels(job_df["Job Title"], fontsize=12, fontweight='bold')
#     ax.set_xlabel("Similarity Score", fontsize=12, fontweight='bold')
#     ax.set_title("Job Match Similarity Scores", fontsize=14, fontweight="bold", color="darkred")
#     ax.invert_yaxis()
#     ax.grid(axis='x', linestyle="--", alpha=0.6)
#     ax.set_xlim(0, 1)

#     plt.subplots_adjust(left=0.3)
#     plt.tight_layout()
#     plt.show()

#     print(job_df)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

def match_jobs(resume_text, job_listings):
    """
    Matches the given resume text with job descriptions using cosine similarity.

    Args:
        resume_text (str): The text content of the resume.
        job_listings (list): List of job dictionaries with "title" and "description".

    Returns:
        pd.DataFrame: Sorted DataFrame with job titles and similarity scores.
    """
    if not job_listings:
        print("No job listings available for matching.")
        return pd.DataFrame(columns=["Job Title", "Cosine Similarity"])

    # Extract job titles and descriptions
    job_titles = [job["title"] for job in job_listings]
    job_texts = [job["description"] for job in job_listings]

    # Compute embeddings
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_embeddings = model.encode(job_texts, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_similarities = util.pytorch_cos_sim(resume_embedding, job_embeddings)[0].cpu().numpy()

    # Create DataFrame with results
    job_df = pd.DataFrame({
        "Job Title": job_titles,
        "Cosine Similarity": cosine_similarities
    })

    # Sort by similarity in descending order
    job_df = job_df.sort_values(by="Cosine Similarity", ascending=False).reset_index(drop=True)

    return job_df

# Ensure this part runs only when executing directly, not when imported
if __name__ == "__main__":
    from pymongo import MongoClient

    # Connect to MongoDB and fetch jobs
    client = MongoClient("mongodb://127.0.0.1:27017/")
    db = client["career_advisor"]
    jobs_collection = db["jobs"]
    
    job_listings = list(jobs_collection.find({}, {"_id": 0, "title": 1, "description": 1}))

    # Sample resume text
    resume_text = "Experienced software engineer skilled in Python, machine learning, and web development."

    # Match jobs
    job_df = match_jobs(resume_text, job_listings)

    # Plot results
    fig, ax = plt.subplots(figsize=(12, 6))
    y_positions = np.arange(len(job_df))
    colors = sns.color_palette("coolwarm", len(job_df))

    bars = ax.barh(y_positions, job_df["Cosine Similarity"], color=colors, edgecolor='black')

    for bar, score in zip(bars, job_df["Cosine Similarity"]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f'{score:.2f}', 
                va='center', fontsize=10, fontweight='bold', color='black')

    ax.set_yticks(y_positions)
    ax.set_yticklabels(job_df["Job Title"], fontsize=12, fontweight='bold')
    ax.set_xlabel("Similarity Score", fontsize=12, fontweight='bold')
    ax.set_title("Job Match Similarity Scores", fontsize=14, fontweight="bold", color="darkred")
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle="--", alpha=0.6)
    ax.set_xlim(0, 1)

    plt.subplots_adjust(left=0.3)
    plt.tight_layout()
    plt.show()

    print(job_df)
