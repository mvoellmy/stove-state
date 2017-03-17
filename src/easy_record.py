import picamera
import time
import _thread

path = '../data/recorded/'

# Thread listening for a keyboard press and appending it to a list
def input_thread(list):
    list.append(input())

# Function containing a loop which gets exited 
def record():
    list = []
    _thread.start_new_thread(input_thread, (list,))
    while not list:
        camera.wait_recording()

# Main Loop
while True:
        
        file_name = input("Enter filename or simply press enter to start recording: ")

        with picamera.PiCamera() as camera:
                camera.resolution = (1640, 1232) # full FOV  
                file_name = time.strftime("%Y-%m-%d-%H:%M:%S") + '_' + file_name + '.h264'
                path = path + file_name
                camera.start_recording(path)

                print("Press any key to stop recording...")
                record()
                camera.stop_recording()
                print('The video has been saved as ' + file_name)

