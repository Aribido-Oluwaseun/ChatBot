from flask import Flask, render_template
from flask import session
from flask import request
import classiersfortext

answer="default ans"
app = Flask(__name__)

@app.route('/')
def start_bot():
	classiersfortext.main(TRAIN_TIEBOT=True)
	return render_template('home.html')

@app.route('/question', methods=['POST', 'GET'])
def print_something():
	global answer
	if request.method == 'POST':
		x=request.form['question']
		answer=classiersfortext.main(x)
		return render_template('home.html')
	if request.method == 'GET':
		return answer
	
@app.route('/feedback', methods=['POST'])
def send_feedback():
	x = request.form['feedback']
	print x
	return render_template('home.html')


if __name__=='__main__':
    app.run()
