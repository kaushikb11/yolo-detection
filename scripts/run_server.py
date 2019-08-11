import os
import signal
import subprocess

parent_directory = os.path.dirname(os.getcwd())

os.chdir(parent_directory + "/server")
tf_ic_server = ""
flask_server = ""

tf_serving_command = "tensorflow_model_server --model_base_path={}/darkflow --rest_api_port=9000 --model_name=darkflow".format(parent_directory)

flask_run_command = "python app.py"

try:
    tf_ic_server = subprocess.Popen([tf_serving_command],
        stdout=subprocess.DEVNULL,
        shell=True,
        preexec_fn=os.setsid)
    print("Started TensorFlow Serving Object Detection server!")

    flask_server = subprocess.Popen([flask_run_command],
        stdout=subprocess.DEVNULL,
        shell=True,
        preexec_fn=os.setsid)
    print("Started Flask server!")

    while True:
        print("Type 'exit' and press 'enter' OR press CTRL+C to quit: ")
        in_str = input().strip().lower()
        if in_str == 'q' or in_str == 'exit':
            print('Shutting down all servers...')
            os.killpg(os.getpgid(tf_ic_server.pid), signal.SIGTERM)
            os.killpg(os.getpgid(flask_server.pid), signal.SIGTERM)
            print('Servers successfully shutdown!')
            break
        else:
            continue

except KeyboardInterrupt:
    print('Shutting down all servers...')
    os.killpg(os.getpgid(tf_ic_server.pid), signal.SIGTERM)
    os.killpg(os.getpgid(flask_server.pid), signal.SIGTERM)
    print('Servers successfully shutdown!')
