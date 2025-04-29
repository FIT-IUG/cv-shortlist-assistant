# CV Shortlist Assistant
Welcome to Our CV Shortlist Assistant repository!  This tool is part of our graduation project and leverages AI to streamline the resume screening process, automating candidate selection based on job descriptions. It utilizes Large Language Models (LLMs), embedding-based similarity analysis, and automated ranking mechanisms to evaluate resumes efficiently and transparently.

## Problem Addressed
Traditional hiring processes are time-consuming, biased, and inefficient. Our tool aims to address these issues by offering a solution that can quickly and fairly rank candidates based on their fit to the provided job descriptions.

## Key Features
1. Resume Parsing: Automatically extracts data from various resume formats (PDF, DOCX).
2. Similarity Scoring: Uses cosine similarity to match resumes with job descriptions.
3. Ranking: Generates a shortlist of the top 5 candidates based on their similarity to the job description.
4. Strengths & Weaknesses: Provides detailed analysis of each candidate's qualifications, identifying their strengths and weaknesses.

## System Design and Architecture
The system operates in several stages:
1. Data Collection: Collects resumes and job descriptions
2. Preprocessing: Processes and extracts key features from resumes and job descriptions.
3. Embedding Generation: Converts text to numerical representations using pre-trained LLMs.
4. Prompt Engineering: Designs specific prompts to instruct the LLM on evaluating CVs based on the job description and user preferences.
5. Similarity Scoring: Calculates cosine similarity scores between the job description embedding and each CV embedding.
6. Thresholding: Determines a threshold score to filter out CVs that do not meet the minimum similarity requirement.
7. Review and Refinement: Optionally, human reviewers assess shortlisted candidates.
8. Output Generation: Compiles the final shortlist and presents it in a user-friendly format.
9. Iterate and Improve: Continuously gathers feedback and improves the pipeline based on new data and outcomes.

You can view the full architecture diagram and pipeline flowcharts through the following links:
- System Architecture Diagram: https://app.eraser.io/workspace/ZgdxLoDESthLwnW3Pv1d?origin=share
- LLM Pipeline Flowchart: https://app.eraser.io/workspace/DTtzx6UA6Z8QD1eiGKI8?origin=share

## Try Our Tool
Experience the CV Shortlist Assistant Tool in action by visiting the following link:
https://cv-shortlist-assistant.streamlit.app/
## Live Demo Video
https://drive.google.com/file/d/1aHxlvb0HqGkvgGBL8IdQbOleraQPYFtM/view?usp=sharing

## Contributing
We welcome contributions to improve this project. Feel free to open an issue or submit a pull request with any improvements or bug fixes.

## License
This project is open-source and available for free. You can freely use, modify, and distribute it.

