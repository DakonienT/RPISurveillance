import cv2
from flask import Flask, render_template, Response, request
import logging
from xmlrpc.client import boolean
import psutil
from datetime import datetime
import socket
app = Flask(__name__, template_folder=r'C:\Users\simon\Desktop\RPISurveillance\templates')
camera = cv2.VideoCapture(0)

scale_percent = 120

class Recorder:
    color = (255, 0, 0)
    record = False
    image = None
    current_record_path = ""
    writer = cv2.VideoWriter()
    def getFileName(self):
        date_string = datetime.now().strftime("%d-%m-%y--%H-%M-%S")
        filename = 'Record_' + date_string + '.avi'
        return filename
    def create_writer(self):
        out_path = self.getFileName()
        self.current_record_path = out_path
        self.writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), 20, (self.image.shape[1], self.image.shape[0]))
    def release_writer(self):
        self.writer.release()
        
        
    
recorder_object = Recorder()
def genFrames():
    while True:
        success, frame = camera.read()
        #print(frame[0])
        #width = int(frame[1] * scale_percent / 100)
        #height = int(frame[0] * scale_percent / 100)
        #dsize = (width, height)
        #frame = cv2.resize(frame, dsize)
        #Put date & time        
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        #logging.info(recorder_object.color)
        frame = cv2.putText(frame, dt_string, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, recorder_object.color, 2,cv2.LINE_AA)
        recorder_object.image = frame
        if(recorder_object.record):
            recorder_object.writer.write(frame)
        if not success:
            break
            logging.error("Error : cannot get image")
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/', methods = ['GET', 'POST'])
def index():
    
    if request.method == 'POST':
        if request.form.get('Start recording') == 'Start recording':

            logging.info("Start recording")
            recorder_object.color = (0, 0, 255)
            recorder_object.record = True
            logging.info("Video will be stored as " + str(recorder_object.getFileName()))
            recorder_object.create_writer()
        elif request.form.get('Stop recording')== 'Stop recording':

            logging.info("Stop recording...")
            recorder_object.color = (255, 0 ,0)
            recorder_object.record = False
            recorder_object.release_writer()            
        else:
            pass
        #logging.info(recorder_object.color)

    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(genFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.root.setLevel(logging.NOTSET)
    logging.basicConfig(level=logging.NOTSET)
    logging.info("Hostname : %s", hostname)
    logging.info("IP : %s", IPAddr)
    logging.info("Loading FLASK appliction...")
    app.run(debug=False,host=IPAddr)
    logging.info("Server ready !")