import configparser
import picamera
import picamera.mmal as mmal
import time
import threading


# Read Config data
config = configparser.ConfigParser()
config.read('../cfg/cfg.txt')

path_videos = config.get('paths', 'videos')
path_labels = config.get('paths', 'labels')

class InputThread (threading.Thread):
    def __init__(self, t_start, path_labels):
        threading.Thread.__init__(self)
        self.t_start = t_start
        self.stop_flag = False
        self.list = []

        # Init labels file
        self.path_label_file = path_labels + 'L_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.gmtime(self.t_start)) + '.csv'
        self.label_file = open(self.path_label_file, "w")
        self.label_file.write("Timestamp, PlateNr, Content, Start/End [0,1]")


        
    # Thread listening for a keyboard press and appending it to a list
    def run(self):

        # Open label file

        # Loop waiting for labels or commands to be input
        while self.stop_flag == False: 
            self.list.append(input())
            # Parse commands
            self.parse_input()


        print("Thread terminating")

    def parse_input(self):
        if 'stop' in self.list:
            self.stop_flag = True
        elif self.list:
            t_delta = time.time() - self.t_start
            self.label_file.write("{:.2f}, {} \n".format(t_delta, self.list[0]))
            #self.label_file.write("{}".format(t_delta))
           
        self.list = []

# Function containing a loop which gets exited 
def record(t_start):

    # Starting Input Thread
    input_thread = InputThread(t_start, path_labels)
    input_thread.start()

    print("Type 'stop' to save the recording...")
    while input_thread.is_alive():
        camera.wait_recording()
    

# Main Loop
while True:

    file_name = input("Enter filename or simply press enter to start recording: ")

    with picamera.PiCamera() as camera:
            camera.resolution = (1640, 1232) # full FOV
            
            t_start = time.time()
            file_name = time.strftime("%Y-%m-%d-%H_%M_%S", time.gmtime(t_start)) + '_' + file_name + '.h264'
            path_video = path_videos + file_name

            camera.start_recording(path_video)

            record(t_start)
            camera.stop_recording()
            t_stop = time.time()
            print('The video has been saved as ' + file_name)
            print('The video is {:.0f} seconds long'.format(t_stop - t_start))


