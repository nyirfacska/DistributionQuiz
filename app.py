from flask import Flask, render_template, request, session, redirect, url_for, send_file
import pandas as pd
import numpy as np
import uuid
import time
import csv
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key'

# Kérdésbank betöltése
QUESTION_FILE = 'questions4.csv'
questions_df = pd.read_csv(QUESTION_FILE, delimiter=';', encoding='utf-8-sig')

# Eredményfájl
RESULTS_FILE = 'test_results.csv'
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        header = ['user_id', 'neptun', 'clicks', 'total_time', 'avg_time_per_question', 'correct_answers', 'total_questions', 'language', 'level', 'distribution']
        for i in range(1, 51):
            header.extend([f'Question{i}', f'Answer{i}', f'CorrectAnswer{i}'])
        writer.writerow(header)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'gdpr' not in request.form:
            return "GDPR must be accepted.", 400

        session['user_id'] = str(uuid.uuid4())
        session['neptun'] = request.form['neptun']
        session['language'] = request.form['language']
        session['level'] = request.form['level']
        session['distribution'] = request.form['distribution']
        session['clicks'] = 0
        session['start_time'] = time.time()
        session['quiz_details'] = []
        session['asked_questions'] = []

        return redirect(url_for('test'))

    languages = questions_df['Language'].unique()
    levels = questions_df['Level'].unique()
    return render_template('index.html', languages=languages, levels=levels)

@app.route('/test', methods=['GET', 'POST'])
def test():
    language = session['language']
    level = session['level']
    asked_questions = session.get('asked_questions', [])

    available_questions = questions_df[(questions_df['Language'] == language) & (questions_df['Level'] == level)]

    if request.method == 'POST':
        session['clicks'] += 1
        selected_answer = request.form['answer']

        if 'current_question' in session and session['quiz_details']:
            correct_answer = session['current_question']['CorrectAnswer']
            session['quiz_details'][-1]['selected_answer'] = selected_answer
            session['quiz_details'][-1]['correct_answer'] = correct_answer
            session['quiz_details'][-1]['difficulty'] = session['current_question']['Difficulty']

        if len(session['asked_questions']) >= len(available_questions):
            return redirect(url_for('result'))

    remaining_questions = available_questions[~available_questions['Question'].isin(asked_questions)]

    if remaining_questions.empty:
        return redirect(url_for('result'))

    distribution = session['distribution']
    difficulties = remaining_questions['Difficulty'].unique()
    if distribution == 'uniform':
        chosen_difficulty = np.random.choice(difficulties)
    elif distribution == 'normal':
        chosen_difficulty = int(np.clip(np.random.normal(loc=3, scale=1), 1, 5))
    elif distribution == 'binomial':
        chosen_difficulty = np.random.binomial(n=4, p=0.5) + 1
    elif distribution == 'exponential':
        chosen_difficulty = int(np.clip(np.random.exponential(scale=1.5) + 1, 1, 5))
    elif distribution == 'poisson':
        chosen_difficulty = int(np.clip(np.random.poisson(lam=2) + 1, 1, 5))
    else:
        chosen_difficulty = np.random.choice(difficulties)

    matching_questions = remaining_questions[remaining_questions['Difficulty'] == chosen_difficulty]
    if matching_questions.empty:
        question = remaining_questions.sample(1).iloc[0].to_dict()
    else:
        question = matching_questions.sample(1).iloc[0].to_dict()

    session['current_question'] = question
    question_text = question['Question']
    if question_text not in session['asked_questions']:
        session['asked_questions'].append(question_text)
        session['quiz_details'].append({'question': question['Question'], 'difficulty': question['Difficulty']})

    return render_template('test.html', question=question, progress=len(session['quiz_details']), total=len(available_questions))

@app.route('/result')
def result():
    total_time = int(time.time() - session['start_time'])
    total_questions = len(session['quiz_details'])
    correct_answers = sum(1 for q in session['quiz_details'] if q.get('selected_answer') == q.get('correct_answer'))
    avg_time = total_time / total_questions if total_questions > 0 else 0

    labels = ['Correct', 'Incorrect']
    values = [correct_answers, total_questions - correct_answers]
    plt.figure()
    plt.bar(labels, values)
    plt.title("Eredmények összesítése")
    plt.ylabel("Darabszám")
    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    diff_counts = {}
    for q in session['quiz_details']:
        d = q.get('difficulty', -1)
        if d not in diff_counts:
            diff_counts[d] = 0
        diff_counts[d] += 1

    plt.figure()
    plt.bar(diff_counts.keys(), diff_counts.values())
    plt.title("Kérdések eloszlása nehézség szerint")
    plt.xlabel("Nehézségi szint")
    plt.ylabel("Darabszám")
    plt.tight_layout()
    img2 = BytesIO()
    plt.savefig(img2, format='png')
    img2.seek(0)
    diff_plot_url = base64.b64encode(img2.getvalue()).decode()
    plt.close()

    row = [
        session['user_id'],
        session['neptun'],
        session['clicks'],
        total_time,
        round(avg_time, 2),
        correct_answers,
        total_questions,
        session['language'],
        session['level'],
        session['distribution']
    ]

    for q in session['quiz_details']:
        row.extend([
            q.get('question', ''),
            q.get('selected_answer', ''),
            q.get('correct_answer', '')
        ])

    for _ in range(50 - len(session['quiz_details'])):
        row.extend(['', '', ''])

    with open(RESULTS_FILE, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    return render_template('result.html', quiz=session['quiz_details'], correct=correct_answers, total=total_questions, plot_url=plot_url, diff_plot_url=diff_plot_url)

@app.route('/download_results')
def download_results():
    return send_file(RESULTS_FILE, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5050)
