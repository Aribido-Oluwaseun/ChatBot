from flask import Flask, render_template
from flask import session
from flask import request
import classiersfortext

answer="default ans"
app = Flask(__name__)
classifier = None
tiebot = None

@app.route('/')
def start_bot():
	global classifier
	global tiebot
	classifier, tiebot = classiersfortext.train()
	return render_template('home.html')

@app.route('/question', methods=['POST', 'GET'])
def print_something():
	global answer
	if request.method == 'POST':
		x=request.form['question']
		answer=classiersfortext.predict(classifier=classifier, tiebot=tiebot, question=x)
		return render_template('home.html')
	if request.method == 'GET':
		return answer
	
@app.route('/feedback', methods=['POST', 'GET'])
def send_feedback():
	if request.method == 'POST':
		question = request.form['question']
		answer = request.form['feedback']
		checkFeedbadk = classiersfortext.takeFeedBack(question, answer)
	if request.method == 'GET':
		return render_template('home.html')

if __name__=='__main__':
    app.run()
