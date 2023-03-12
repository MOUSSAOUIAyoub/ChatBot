from flask import Flask, jsonify, request , render_template
from flask_cors import CORS
import appv2
app = Flask(__name__)
CORS(app)
@app.route('/api/talk',methods=['POST'])
def index():
    user_input = request.json['user_input']
    return jsonify({'msg':str(appv2.handle_conversation(user_input))})

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=3000, debug=True)