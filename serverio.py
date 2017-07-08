from __future__ import print_function
from flask import Flask, render_template,request, redirect
from flask_socketio import SocketIO
import transaction_retriever
from werkzeug import secure_filename
from OCR import *
from transaction_retriever import *
import sys
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
from flask import request
from flask_socketio import send, emit
from os import environ
import re
from rnn import *


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

@socketio.on('message')
def handle_message(message):
	print(message, file=sys.stderr)
	send(message)

@socketio.on('json')
def handle_json(json):
    print('received json: ' + str(json))
    send(json, json=True)


@socketio.on_error()        # Handles the default namespace
def error_handler(e):
    pass
@socketio.on_error('/chat') # handles the '/chat' namespace
def error_handler_chat(e):
    pass
@socketio.on_error_default  # handles all namespaces without an explicit error handler
def default_error_handler(e):
    pass

@app.route('/api/mode')
def api_mode():
	print("request",file = sys.stderr)
	print (request,file = sys.stderr)
	mode = request.args.get('mode')
	print (mode, file = sys.stderr)

	socketio.emit('api_mode',{'mode': mode})
	return render_template("hotel.html")

@app.route('/api/facility')
def api_facility():
	print("request",file = sys.stderr)
	print (request,file = sys.stderr)
	action = request.args.get('action')
	facility = request.args.get('facility')
	print (action, file = sys.stderr)

	socketio.emit('api_facility',{'action': action, 'facility' : facility})
	return render_template("hotel.html")
@socketio.on('connect')
def connect():
	@app.route('/api/OCR', methods = ['POST'])
	def OCR():

		f = request.files['image']
		f.save(secure_filename('receipt.jpg'))
		print (ocr_space_file(filename='receipt.jpg'), file=sys.stderr)
		# print (request.files['image'], file= sys.stderr)
		floatList = []
		for word in ocr_space_file(filename='receipt.jpg').split('\r\n'):
			if is_number(word):
				floatList.append(float(word))
			print (word)
		total = max(floatList)
		print (total	)
		socketio.emit('receipt', {'total': total})
		return redirect('/')

@socketio.on('getPersonalAnalysis')
def handleAnalysis(name):
	print (getPersonalAnalysis(name))
	return getPersonalAnalysis(name)

@socketio.on('getComparison')
def handleComparison(name, criteria):
    return getComparison(name, criteria)

@socketio.on('coupon')
def coupon():
	print(predict())
	return predict()

@app.route('/')
def index():
	return render_template("index.html")

if __name__ == '__main__':
	HOST = environ.get('SERVER_HOST', 'localhost')
	try:
		PORT = int(environ.get('PORT','5009'))
	except ValueError:
		PORT = 5009

	print('server running on ' + str(PORT), file=sys.stderr)
	# print(getPersonalAnalysis('Rachel Trujillo'), file=sys.stderr)
    #os.system("python transaction_retriever.py")
    #print('File loaded.')
	socketio.run(app, port =  PORT, host= '0.0.0.0')
 	