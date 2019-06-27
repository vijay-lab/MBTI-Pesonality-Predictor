from flask import Flask, render_template, url_for, request
from mbti_pred import get_prediction

app = Flask(__name__)

@app.route("/home")
def home_page():
	return render_template("persona-pred.html")


@app.route("/result", methods=["POST"])
def output():
	data = request.form
	message = data["msg"]

	if message != "":
		message_type = get_prediction(message)
		if message_type == "ENFJ":
			return render_template('enfj.html', message_type=message_type)
		elif message_type == "ENFP":
			return render_template('enfp.html', message_type=message_type)
		elif message_type == "ENTJ":
			return render_template('entj.html', message_type=message_type)
		elif message_type == "ENTP":
			return render_template('entp.html', message_type=message_type)
		elif message_type == "ESFJ":
			return render_template('esfj.html', message_type=message_type)
		elif message_type == "ESFP":
			return render_template('esfp.html', message_type=message_type)
		elif message_type == "ESTJ":
			return render_template('estj.html', message_type=message_type)
		elif message_type == "ESTP":
			return render_template('estp.html', message_type=message_type)
		elif message_type == "INFJ":
			return render_template('infj.html', message_type=message_type)
		elif message_type == "INFP":
			return render_template('infp.html', message_type=message_type)
		elif message_type == "INTJ":
			return render_template('intj.html', message_type=message_type)
		elif message_type == "INTP":
			return render_template('intp.html', message_type=message_type)
		elif message_type == "ISFJ":
			return render_template('isfj.html', message_type=message_type)
		elif message_type == "ISFP":
			return render_template('isfp.html', message_type=message_type)
		elif message_type == "ISTJ":
			return render_template('istj.html', message_type=message_type)
		elif message_type == "ISTP":
			return render_template('istp.html', message_type=message_type)
	else:
		message_type = "Please Enter Message Properly"

if __name__ == "__main__":
	app.run(debug=0)