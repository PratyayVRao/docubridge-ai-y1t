import os
from flask import Flask, request, render_template
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import pandas as pd

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecret")
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
client = InferenceClient(token=HF_API_TOKEN)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("excel_file")
    question = request.form.get("user_question")
    if not file or not question:
        return "Missing file or question."

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    df = pd.read_excel(filepath)
    summary = df.describe(include='all').to_string()
    preview = df.head().to_html()

    prompt = f"Dataset summary:\n{summary}\n\nQuestion: {question}\nAnswer:"

    response = client.text_generation(
        model="tiiuae/falcon-7b-instruct",
        prompt=prompt,
        max_new_tokens=150
    )

    generated_text = response.generated_text
    answer = generated_text[len(prompt):].strip()

    return render_template("qa.html", preview=preview, question=question, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
