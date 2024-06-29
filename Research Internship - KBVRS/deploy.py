from flask import Flask, render_template, request
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Define the directory containing the transcribed text files
transcribed_directory = r'D:\SRM\3rd_Year\6th_Semester\Research\Video_Retriveal_System\Data\Output'
video_directory = r'D:\SRM\3rd_Year\6th_Semester\Research\Video_Retriveal_System\Data\Video'

# Reference text for similarity calculation
reference_text = "linear regression"

# Function to calculate cosine similarity
def calculate_cosine_similarity(transcribed_text, reference_text):
    vectorizer = CountVectorizer().fit([transcribed_text, reference_text])
    vectorized_text = vectorizer.transform([transcribed_text, reference_text])
    cosine_sim = cosine_similarity(vectorized_text)[0][1]
    return cosine_sim

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        similarity_scores = []

        # Iterate through each text file in the directory
        for filename in os.listdir(transcribed_directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(transcribed_directory, filename)
                
                # Read the content of the text file
                with open(file_path, 'r') as file:
                    transcribed_text = file.read()

                # Calculate cosine similarity
                cosine_sim = calculate_cosine_similarity(transcribed_text, input_text)

                # Append the filename and cosine similarity score to the list
                similarity_scores.append((filename, cosine_sim))

        # Sort the list of similarity scores based on cosine similarity
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        # Filter out files with cosine similarity of 0
        similarity_scores = [(filename, cosine_sim) for filename, cosine_sim in similarity_scores if cosine_sim > 0]

        # Construct the list of video filenames
        video_files = [os.path.join(video_directory, os.path.splitext(filename)[0] + '.mp4') for filename, _ in similarity_scores]

        return render_template('index.html', video_files=video_files)

    return render_template('index.html', video_files=None)

if __name__ == '__main__':
    app.run(debug=True)




