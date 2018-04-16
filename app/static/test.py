from flask import Flask, render_template
from flask import session
answer= "this is answer"

app = Flask(__name__)

@app.route('/')
def hello_world():
   
    return render_template('home.html', value=answer)


if __name__=='__main__':
    app.run()
