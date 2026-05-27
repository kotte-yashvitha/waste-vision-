from flask import Flask, render_template, request, redirect, url_for, session
import os
import cv2
import numpy as np
import joblib
import mysql.connector
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'secret123'  # Use a secure secret key in production

app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
svm_model = joblib.load('cnn_svm_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
cnn_model = MobileNetV2(include_top=False, weights='imagenet', pooling='avg', input_shape=(128, 128, 3))

# MySQL connection
def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="Classification"
    )

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        conn = get_db()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
            conn.commit()
            return redirect(url_for('login'))
        except mysql.connector.errors.IntegrityError:
            return render_template('signup.html', message="Username already exists.")
        finally:
            cursor.close()
            conn.close()
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password_input = request.form['password']

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username=%s", (username,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row and check_password_hash(row[0], password_input):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', message="Invalid credentials.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))

    prediction = ""
    image = ""

    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = cv2.imread(filepath)
            img = cv2.resize(img, (128, 128))
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            features = cnn_model.predict(img)
            pred = svm_model.predict(features)
            prediction = label_encoder.inverse_transform(pred)[0]
            image = filepath

    return render_template('index.html', prediction=prediction, image=image, user=session['username'])

if __name__ == '__main__':
    app.run(debug=True)
