import os
import pickle
from flask import Flask, render_template_string, request
import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Model configuration using your specified model file name
MODEL_PATH = "model (2).keras"
TOKENIZER_PATH = "tokenizer.pkl"

# Load the Sequential spam classification model and tokenizer safely
model = keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as handle:
  tokenizer = pickle.load(handle)

MAX_LEN = 50  # Must match your model's input shape configuration[cite: 1]

# Embedded HTML Template containing the entire frontend UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Spam Detector</title>
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f3f4f6;
            --card-bg: #ffffff;
            --text-main: #1f2937;
            --text-muted: #6b7280;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            width: 100%;
            max-width: 600px;
            background: var(--card-bg);
            padding: 2.5rem;
            border-radius: 1rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
            margin: 1rem;
        }
        h1 {
            font-size: 1.875rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
            color: var(--primary);
        }
        p.subtitle {
            text-align: center;
            color: var(--text-muted);
            margin-bottom: 2rem;
            font-size: 0.95rem;
        }
        .form-group {
            margin-bottom: 1.5rem;
        }
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 0.5rem;
            font-size: 0.875rem;
        }
        textarea {
            width: 100%;
            height: 150px;
            padding: 0.75rem;
            border: 1px solid #d1d5db;
            border-radius: 0.5rem;
            font-family: inherit;
            font-size: 1rem;
            resize: vertical;
            box-sizing: border-box;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        textarea:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }
        button {
            width: 100%;
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 0.5rem;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        .result-card {
            margin-top: 2rem;
            padding: 1.25rem;
            border-radius: 0.5rem;
            text-align: center;
            animation: fadeIn 0.3s ease-in-out;
        }
        .spam {
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #f87171;
        }
        .not-spam {
            background-color: #d1fae5;
            color: #065f46;
            border: 1px solid #34d399;
        }
        .result-card h3 {
            margin: 0 0 0.25rem 0;
            font-size: 1.25rem;
        }
        .result-card p {
            margin: 0;
            font-size: 0.875rem;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(5px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ Spam Detector</h1>
        <p class="subtitle">Analyze text or email contents instantly using your Sequential RNN spam classification model.</p>
        
        <form method="POST">
            <div class="form-group">
                <label for="email_text">Paste Email or Text Content:</label>
                <textarea name="email_text" id="email_text" placeholder="Type or paste your message here..." required>{{ email_text }}</textarea>
            </div>
            <button type="submit">Analyze Message</button>
        </form>

        {% if prediction %}
            <div class="result-card {% if prediction == 'SPAM' %}spam{% else %}not-spam{% endif %}">
                <h3>Result: {{ prediction }}</h3>
                <p>Spam Confidence Probability: <strong>{{ probability }}%</strong></p>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
  prediction = None
  probability = None
  email_text = ""

  if request.method == "POST":
    email_text = request.form.get("email_text", "")
    if email_text.strip():
      # Preprocess text sequence
      sequence = tokenizer.texts_to_sequences([email_text])
      padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")

      # Predict using the loaded Sequential model
      score = float(model.predict(padded_sequence)[0][0])
      probability = round(score * 100, 2)

      # Threshold evaluation (0.5)
      if score > 0.5:
        prediction = "SPAM"
      else:
        prediction = "NOT SPAM"

  return render_template_string(
      HTML_TEMPLATE,
      prediction=prediction,
      probability=probability,
      email_text=email_text,
  )


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host="0.0.0.0", port=port)
