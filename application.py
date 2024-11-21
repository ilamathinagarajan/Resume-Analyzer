import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import pytesseract
from PIL import Image
import requests
from requests.exceptions import InvalidURL,ConnectionError
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
from io import BytesIO
import base64
import re
st.set_page_config(
    page_title="Resume Analyzer",
    page_icon="📂", 
    layout="centered"
)
def add_background_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/png;base64,{encoded_image});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_background_local("C:\\Users\\Aswathy\\OneDrive\\Desktop\\Github\\Resume Analyser RNN LSTM\\bg.jpg")

import streamlit as st

st.markdown(
    """
    <style>
    h1 {
        color: RebeccaPurple;
        font-size: 3rem;
        margin-bottom: 30px;
        text-align: center;
        font-family: Cursive, Lucida-Handwriting;  
    }

    small {
        display: block;
        font-size: 0.8rem;
        color: #666;
        text-align: center;
    }
    .stFileUploader, .stTextInput {
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 20px;
        color: #333;
    }

    .stFileUploader label ,.stTextInput label {
        font-weight: bold;
        color: black;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    p{
        font-family: Cursive, Lucida-Handwriting; 
    }
    .stTextInput input {
        background-color: #e8f5e9;
        border-radius: 5px;
    }
    h3{
        text-align:center;
        font-size: 1.8rem;
        font-family: Cursive, Lucida-Handwriting;
    }
    .msg{
        font-size: 1.1rem;
        text-align:center;
        color:LightSlateGrey;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div>', unsafe_allow_html=True)

st.markdown("<h1>Resume Analyser</h1>", unsafe_allow_html=True)

st.markdown('<h3>Is your resume good enough?</h3>', unsafe_allow_html=True)

st.markdown('<p class="msg">A free and fast AI resume analyser, that checks how well your resume align with job descriptions and gives you suggestions to improve your resume, highlight key skills and tailor it to specific job requirements, thus increases your chance of passing the resume screening process!</p>',unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your resume:", type=["pdf", "docx", "txt", "jpg", "jpeg", "png"])
st.markdown('<small>Privacy guaranteed</small>', unsafe_allow_html=True)
job_url = st.text_input("Enter the job description URL:")

st.markdown('</div>', unsafe_allow_html=True)


nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

try:
    stopwords.words('english')
    word_tokenize('test')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

skills_list = ['Agile', 'Azure', 'DevOps', 'Machine Learning', 'Data Science', 'Deep Learning', 'Data Analysis', 'Data Visualization',
               'Cloud Computing', 'Big Data', 'Web Development', 'Software Development', 'Mobile App Development', 'Frontend Development',
               'Backend Development', 'Full Stack Development', 'REST API', 'GraphQL', 'Database Management', 'Microservices', 'Docker', 
               'Kubernetes', 'CI/CD', 'Scrum', 'Version Control (Git)', 'GitHub', 'GitLab', 'Automation', 'Test Automation', 'UI/UX Design', 
               'Cybersecurity', 'Blockchain', 'Artificial Intelligence', 'Natural Language Processing (NLP)', 'TensorFlow', 'Scikit-learn', 
               'AWS', 'Azure', 'Google Cloud', 'Azure DevOps', 'Postman', 'Jenkins', 'CI/CD Pipeline', 'Cloud Security', 'Database Design', 
               'NoSQL', 'Relational Databases', 'MongoDB', 'MySQL', 'PostgreSQL', 'Oracle', 'SQLAlchemy', 'Web Scraping', 'SEO', 'Selenium', 
               'Tableau', 'Power BI', 'Business Intelligence', 'Game Development', 'API Development', 'Project Management', 'Agile Methodology', 
               'Kanban', 'Jira', 'Lean', 'Software Testing', 'Unit Testing', 'JUnit', 'Mockito', 'JUnit Testing', 'Test-Driven Development (TDD)', 
               'Microservices Architecture', 'CI/CD Pipelines', 'Cloud Infrastructure', 'Serverless Architecture', 'Containerization', 
               'Infrastructure as Code', 'ElasticSearch', 'Nginx', 'Apache', 'JSP', 'JUnit', 'Maven', 'Redux', 'React', 'Vue.js', 'Angular', 
               'Flutter', 'Xamarin', 'Laravel', 'Spring Framework', 'Spring Boot', 'FastAPI', 'Flask', 'Express.js', 'Django', 'Node.js', 
               'Ruby on Rails', 'SaaS', 'Game Engines (Unity, Unreal Engine)', 'Docker Compose', 'Terraform', 'Chef', 'Puppet', 'Vagrant', 
               'NGINX', 'Apache Kafka', 'Service-Oriented Architecture (SOA)', 'AWS Lambda', 'WebSockets', 'Firebase', 'Vagrant', 'MongoDB', 
               'JPA', 'Hibernate', 'Figma', 'Photoshop', 'Illustrator', 'UI/UX Design', 'Wireframing', 'Sketch', 'Zeplin', 'Canva', 'PowerShell', 
               'Terraform', 'Solidity', 'Automated Deployment', 'Jenkins CI/CD', 'Ansible', 'Serverless Framework', 'Python', 'Java', 'C#', 'C++', 
               'JavaScript', 'TypeScript', 'Ruby', 'PHP', 'Swift', 'Kotlin', 'Go', 'Rust', 'Perl', 'Objective-C', 'SQL', 'R', 'HTML', 'CSS', 'Sass', 
               'VHDL', 'MATLAB', 'Lua', 'Shell Scripting', 'Dart', 'ActionScript', 'Scala', 'Haskell', 'Elixir', 'F#', 'Visual Basic', 'SQL Server', 
               'PL/SQL', 'T-SQL']

def preprocess_text(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return filtered_words

def extract_skills_from_text(text, skill_list):
    text_lower = text.lower()
    extracted_skills = [skill for skill in skill_list if skill.lower() in text_lower]
    return extracted_skills

def extract_text_from_docx(file):
    doc = Document(BytesIO(file))
    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_pdf(file):
    reader = PdfReader(BytesIO(file))
    return '\n'.join(page.extract_text() for page in reader.pages)

def extract_text_from_txt(file):
    return file.decode('utf-8')

def extract_text_from_image(file):
    image = Image.open(BytesIO(file))
    return pytesseract.image_to_string(image)

def extract_skills(file_bytes):
    file_extension = file_bytes[:4] 
    
    if file_extension.startswith(b'%PDF'):
        text = extract_text_from_pdf(file_bytes)
    elif file_extension.startswith(b'PK'):
        text = extract_text_from_docx(file_bytes)
    else:
        text = extract_text_from_txt(file_bytes)
        
    skills = extract_skills_from_text(text, skills_list)
    return skills

def preprocess_skills(skills):
    return ' '.join(eval(skills))

def extract_skills_from_job_description(url):
    response = requests.get(url)
    
    if response.status_code != 200:
        st.error("Failed to retrieve the web page.")
        return []
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    job_text = soup.get_text()
    
    extracted_skills = extract_skills_from_text(job_text, skills_list)
    
    return extracted_skills

# Loading model and tokenizer
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = tf.keras.models.load_model('C:\\Users\\Aswathy\\OneDrive\\Desktop\\Github\\Resume Analyser RNN LSTM\\rnn_lstm_model.keras')

# Load the tokenizer
with open('C:\\Users\\Aswathy\\OneDrive\\Desktop\\Github\\Resume Analyser RNN LSTM\\tokenizer_rnn_lstm.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

recommendations_dict = {
    "Agile": "Explore Agile methodologies like Scrum and Kanban through practical application in team projects.",
    "Azure": "Master Azure cloud solutions by exploring services like Azure Virtual Machines, App Services, and Azure DevOps.",
    "DevOps": "Learn DevOps tools like Jenkins, Docker, and Kubernetes to enhance continuous integration and delivery.",
    "Machine Learning": "Work on machine learning models using libraries like TensorFlow and scikit-learn.",
    "Data Science": "Enhance your data science expertise by building projects involving data preprocessing, analysis, and visualization.",
    "Deep Learning": "Experiment with deep learning architectures like CNNs and RNNs using TensorFlow or PyTorch.",
    "Data Analysis": "Leverage tools like Pandas and NumPy to analyze datasets and draw actionable insights.",
    "Data Visualization": "Use tools like Tableau, Power BI, or Matplotlib to present data effectively.",
    "Cloud Computing": "Gain expertise in cloud platforms like AWS, Azure, or Google Cloud through certification programs.",
    "Big Data": "Work on Big Data projects using frameworks like Apache Hadoop and Spark.",
    "Web Development": "Build interactive websites using HTML, CSS, JavaScript, and frameworks like React or Angular.",
    "Software Development": "Enhance software development skills by building real-world applications in languages like Java, Python, or C#.",
    "Mobile App Development": "Develop mobile apps using frameworks like Flutter, React Native, or Xamarin.",
    "Frontend Development": "Work on responsive user interfaces using frameworks like React, Angular, or Vue.js.",
    "Backend Development": "Strengthen backend skills with frameworks like Django, Spring Boot, or Express.js.",
    "Full Stack Development": "Build full-stack projects to integrate frontend and backend skills using popular stacks like MERN or MEAN.",
    "REST API": "Practice building and consuming REST APIs using tools like Postman and frameworks like Flask or Express.js.",
    "GraphQL": "Learn GraphQL by creating APIs and integrating them into frontend applications.",
    "Database Management": "Enhance your database management skills with tools like MySQL, PostgreSQL, or MongoDB.",
    "Microservices": "Explore microservices architecture with frameworks like Spring Boot and tools like Docker and Kubernetes.",
    "Docker": "Familiarize yourself with Docker for containerization and deployment of applications.",
    "Kubernetes": "Learn Kubernetes basics to manage and scale containerized applications efficiently.",
    "CI/CD": "Implement CI/CD pipelines using Jenkins, GitHub Actions, or Azure DevOps.",
    "Scrum": "Participate in Scrum meetings to understand and apply the framework in projects.",
    "Version Control (Git)": "Improve your version control skills by working on projects with Git and GitHub.",
    "GitHub": "Explore GitHub features like pull requests and workflows for collaborative development.",
    "GitLab": "Use GitLab to manage CI/CD pipelines and collaborate on projects.",
    "Automation": "Automate tasks using scripting languages like Python or tools like Selenium.",
    "Test Automation": "Develop automated test scripts using Selenium or Cypress.",
    "UI/UX Design": "Practice creating user interfaces and experiences using tools like Figma and Adobe XD.",
    "Cybersecurity": "Strengthen your cybersecurity knowledge by exploring penetration testing and vulnerability assessments.",
    "Blockchain": "Dive into blockchain development with tools like Solidity and frameworks like Ethereum.",
    "Artificial Intelligence": "Work on AI projects focusing on natural language processing, image recognition, or chatbots.",
    "Natural Language Processing (NLP)": "Explore NLP techniques using libraries like spaCy and NLTK.",
    "TensorFlow": "Learn TensorFlow by building machine learning and deep learning models.",
    "Scikit-learn": "Enhance your ML knowledge using scikit-learn for classification, regression, and clustering tasks.",
    "AWS": "Pursue AWS certifications to master cloud computing services like EC2, S3, and Lambda.",
    "Google Cloud": "Explore Google Cloud services like BigQuery, App Engine, and Cloud Functions.",
    "Azure DevOps": "Streamline DevOps processes by using Azure DevOps for CI/CD and project management.",
    "Postman": "Master API testing and development with Postman collections and workflows.",
    "Jenkins": "Use Jenkins to set up and manage CI/CD pipelines for automated builds and deployments.",
    "CI/CD Pipeline": "Learn to build robust CI/CD pipelines using Jenkins, GitHub Actions, or Azure DevOps.",
    "Cloud Security": "Improve your cloud security knowledge by studying IAM policies and encryption techniques.",
    "Database Design": "Focus on designing efficient database schemas to optimize performance and scalability.",
    "NoSQL": "Explore NoSQL databases like MongoDB and Cassandra for unstructured data storage.",
    "Relational Databases": "Master relational databases like MySQL, PostgreSQL, and Oracle for structured data management.",
    "MongoDB": "Learn MongoDB by creating NoSQL databases for scalable applications.",
    "MySQL": "Enhance your MySQL skills by working on database queries and optimizations.",
    "PostgreSQL": "Practice advanced PostgreSQL features like indexing and partitioning.",
    "Oracle": "Dive into Oracle database management with PL/SQL scripting.",
    "SQLAlchemy": "Use SQLAlchemy to integrate databases with Python applications.",
    "Web Scraping": "Learn web scraping techniques using libraries like BeautifulSoup and Scrapy.",
    "SEO": "Optimize web pages for search engines by learning SEO principles and best practices.",
    "Selenium": "Automate browser tasks with Selenium for functional testing and web scraping.",
    "Tableau": "Use Tableau to create dynamic dashboards and data visualizations.",
    "Power BI": "Develop interactive business intelligence reports using Power BI.",
    "Business Intelligence": "Build BI solutions using tools like Tableau, Power BI, or QlikView.",
    "Game Development": "Create games using engines like Unity or Unreal Engine.",
    "API Development": "Develop APIs using frameworks like Flask, FastAPI, or Express.js.",
    "Project Management": "Improve project management skills with tools like Jira, Trello, and Asana.",
    "Agile Methodology": "Gain practical experience in Agile by managing sprints and using Scrum boards.",
    "Kanban": "Enhance productivity using Kanban boards to manage workflow effectively.",
    "Jira": "Use Jira for managing Agile projects and tracking issues.",
    "Lean": "Apply Lean principles to streamline processes and eliminate waste in projects.",
    "Software Testing": "Improve your software testing skills by learning manual and automated testing techniques.",
    "Unit Testing": "Practice unit testing in various languages using frameworks like JUnit, pytest, or NUnit.",
    "JUnit": "Use JUnit for testing Java applications and automating unit test cases.",
    "Mockito": "Master Mockito for mocking in Java testing scenarios.",
    "Test-Driven Development (TDD)": "Adopt TDD by writing tests before implementing features.",
    "Microservices Architecture": "Learn microservices architecture for building scalable and maintainable applications.",
    "Cloud Infrastructure": "Gain expertise in cloud infrastructure management using Terraform or Ansible.",
    "Serverless Architecture": "Explore serverless computing with platforms like AWS Lambda and Azure Functions.",
    "Containerization": "Learn containerization techniques with Docker to streamline development and deployment.",
    "Infrastructure as Code": "Use tools like Terraform or CloudFormation to automate infrastructure provisioning.",
    "ElasticSearch": "Enhance your search capabilities by implementing ElasticSearch in projects.",
    "Nginx": "Use Nginx for load balancing and serving static web content.",
    "Apache": "Configure Apache servers for hosting websites and applications.",
    "Spring Framework": "Build enterprise-grade applications using the Spring Framework and its ecosystem.",
    "Spring Boot": "Develop RESTful APIs and microservices using Spring Boot.",
    "FastAPI": "Create high-performance APIs with Python using FastAPI.",
    "Flask": "Build lightweight web applications and APIs with Flask.",
    "Django": "Develop robust web applications using Django's ORM and MVC architecture.",
    "Node.js": "Build scalable backend systems using Node.js and its ecosystem.",
    "Ruby on Rails": "Create web applications using Ruby on Rails for rapid development.",
    "React": "Develop dynamic UIs using React and its component-based architecture.",
    "Vue.js": "Build lightweight web applications using Vue.js.",
    "Angular": "Learn Angular to develop feature-rich, single-page web applications.",
    "Flutter": "Create cross-platform mobile applications using Flutter and Dart.",
    "Laravel": "Develop PHP web applications with the Laravel framework.",
    "Docker Compose": "Learn Docker Compose for managing multi-container applications.",
    "Terraform": "Automate infrastructure provisioning using Terraform scripts.",
    "Chef": "Use Chef for configuration management and automated server setup.",
    "Puppet": "Implement infrastructure as code using Puppet for automated configurations.",
    "Firebase": "Develop real-time applications using Firebase's suite of tools.",
    "Solidity": "Learn Solidity to develop smart contracts on Ethereum blockchain.",
    "Python": "Practice Python coding exercises to improve your programming skills.",
    "Java": "Build enterprise applications using Java and its frameworks.",
    "C#": "Develop applications for Windows or Unity games using C#.",
    "C++": "Enhance C++ skills by implementing data structures and algorithms.",
    "JavaScript": "Develop interactive web applications using JavaScript and its frameworks.",
    "TypeScript": "Learn TypeScript to add strong typing to your JavaScript applications.",
    "Ruby": "Build applications using Ruby and frameworks like Rails.",
    "PHP": "Develop dynamic web pages using PHP and MySQL.",
    "Swift": "Create iOS applications using Swift and Xcode.",
    "Kotlin": "Develop Android applications with Kotlin for modern programming features.",
    "Go": "Learn Go for building scalable, high-performance applications.",
    "Rust": "Explore Rust for system programming and memory safety.",
    "Perl": "Use Perl for text processing and scripting tasks.",
    "Scala": "Leverage Scala for functional programming and big data projects.",
    "Hadoop": "Process large datasets using Apache Hadoop's ecosystem.",
    "Spark": "Perform distributed data processing with Apache Spark."
}

def calculate_missing_skill(resume_skills, job_skills, recommendations_dict):
    resume_set = set(resume_skills)
    job_set = set(job_skills)
    
    missing_skills = job_set - resume_set
    
    recommendations = []
    for skill in missing_skills:
        if skill in recommendations_dict:
            recommendations.append(recommendations_dict[skill])
        else:
            recommendations.append(f"Consider exploring resources to learn {skill}.")
    
    return list(missing_skills), recommendations

resume_skills = []  
job_skills = []  

def is_valid_url(url):
    regex = re.compile(
        r'^(https?://)?(www\.)?([A-Za-z0-9]+)\.(com|org|net|gov|edu|io|in)(/[\w-]*)*$', re.IGNORECASE
    )
    return re.match(regex, url) is not None

if uploaded_file and job_url:
    try:
        if not is_valid_url(job_url):
            st.error("Please enter a valid job description URL starting with http:// or https://!")
        else:

            st.success("Resume uploaded and job description URL provided!")

            file_bytes = uploaded_file.getvalue()

            try:
                resume_skills = extract_skills(file_bytes)
            except UnicodeDecodeError as e:
                st.error("There was an issue processing your resume. Please upload another resume file.")
                st.warning("The file might not be in the correct format or may be corrupted. Please try again.")
                st.stop()

            try:
                job_skills = extract_skills_from_job_description(job_url)
            except ConnectionError as e:
                st.error("Could not connect to the job description URL. Please check your internet connection or the URL.")
            except Exception as e:
                st.error(f"Error extracting skills from job description: {e}")
            
            missing_skills, recommendations = calculate_missing_skill(resume_skills, job_skills, recommendations_dict)

    except InvalidURL as e:
        st.error("Please enter a valid job description URL. Ensure it starts with http:// or https://.")
    except requests.exceptions.MissingSchema as e:
        st.error("Missing URL scheme. Please ensure the URL starts with http:// or https://.")
    except requests.exceptions.RequestException as e:
        st.error("Network Error: Please check your internet connection or the job description URL.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.warning("Please upload your resume and provide a job description URL.")

missing_skills, recommendations = calculate_missing_skill(resume_skills, job_skills, recommendations_dict)

if resume_skills and job_skills:
    resume_text = ' '.join(resume_skills)  
    job_description_text = ' '.join(job_skills)  
    combined_skills = resume_text + ' ' + job_description_text

    combined_skills_set = set(combined_skills.split())

    resume_set = set(resume_skills)
    job_set = set(job_skills)        
    total_skills = resume_set.union(job_set)  
    common_skills = resume_set.intersection(job_set) 
    X = tokenizer.texts_to_sequences([combined_skills]) 
    X = pad_sequences(X, padding='post') 
    match_score_model = model.predict(X)

    st.markdown("""
        <style>
        h4,ul,li{
            font-family: Cursive, Lucida-Handwriting; 
        }
        .match-score {
            backdrop-filter: blur(10px);
            border-radius: 10px;
            color: #333;
            padding: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            text-align: center;
        }
        .missing-skills {
            backdrop-filter: blur(10px);
            border-radius: 10px;
            color: #333;
            padding: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .recommendations {
            backdrop-filter: blur(10px);
            border-radius: 10px;
            color: #333;
            padding: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    if len(total_skills) > 0:
        match_score_skills = len(common_skills) / len(total_skills)
    else:
        match_score_skills = 0.0
    match_score=(match_score_model+match_score_skills)/2
    formatted_score = f"{float(match_score[0]):.2f}"
    st.markdown(f'<div class="match-score"><h4>Match Score: {formatted_score}</h4></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="missing-skills"><h4>Skills Found to be Missing in your Resume:</h4><ul style="list-style-type: square;">{"".join([f"<li>{skill}</li>" for skill in missing_skills])}</ul></div>', unsafe_allow_html=True)

    st.markdown('<div class="recommendations"><h4>Next Steps for Improvement:</h4><ul style="list-style-type: square;">' + ''.join([f'<li>{rec}</li>' for rec in recommendations]) + '</ul></div>', unsafe_allow_html=True)

    st.markdown(
    """
    <div style="backdrop-filter: blur(10px);
                text-align: center;
                padding: 10px;">
    <p style="font-size: 16px; color: #333;">Enhance your work, focus on progess and good luck ahead!</p>
    </div>
    """, 
    unsafe_allow_html=True
)