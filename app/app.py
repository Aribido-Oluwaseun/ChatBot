from flask import Flask, render_template
from flask import session
from flask import request
import classiersfortext

answer="default ans"
app = Flask(__name__)

@app.route('/')
def hello_world(): 	
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
	


if __name__=='__main__':
    app.run()
