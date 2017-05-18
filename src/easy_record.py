import configparser
import picamera
import time
import threading

# Read Config data
config = configparser.ConfigParser()
config.read('../cfg/cfg.txt')

path_videos = config.get('paths', 'videos')
path_labels = config.get('paths', 'labels')
stove_type = config.get('misc', 'stove_type')


class InputThread(threading.Thread):
    """ Thread continuously listening for keyboard inputs and so they can be used for labeling.
    """

    def __init__(self, _t_start, path_labels, rec_name):
        threading.Thread.__init__(self)
        self.t_start = _t_start
        self.stop_flag = False
        self.list = []

        # Init labels file
        self.path_label_file = path_labels + stove_type + time.strftime("_%Y-%m-%d-%H_%M_%S_", time.gmtime(self.t_start)) + rec_name + '.csv'
        self.label_file = open(self.path_label_file, "w")
        self.label_file.write("Timestamp PlateNr Content Start/End[1/0]\n")

    def run(self):
        # Loop waiting for labels or commands to be input
        while not self.stop_flag:
            self.list.append(input())
            # Parse commands
            self.parse_input()

        print("Thread terminating")

    def parse_input(self):
        """ Manages what happens with the input. Checks for 'stop' command and otherwise writes the input into the
        labeling file.
        """

        t_delta = time.time() - self.t_start
        if 'stop' in self.list:
            self.stop_flag = True
            self.label_file.write("{:.2f} stop".format(t_delta, self.list[0]))         
        elif self.list:
            self.label_file.write("{:.2f} {} \n".format(t_delta, self.list[0]))
            # self.label_file.write("{}".format(t_delta))

        self.list = []


# Function containing a loop which gets exited 
def record(t_start, rec_name):

    # Starting Input Thread
    input_thread = InputThread(t_start, path_labels, rec_name)
    input_thread.start()
  
    print("Type 'stop' to save the recording...\n")

    print("Labeling columns:")
    print("PlateNr Content Start/End[1/0]")
    print("------------------------------")
    while input_thread.is_alive():
        camera.wait_recording()



print("          __          ")
print(" ________/__\________ ")
print("/____________________\\")
print("|                    |")
print("|   WELCOME TO THE   |")
print("| COOKOMAT 3 MILLION |")
print("|                    |")
print("|____________________|\n")
print("######################")
print("Golden Recording Rules")
print("- only pans on stove (no lids, plates, etc.)")
print("- only short sleeved, non skin coloured shirt")
print("- always use the same tools, pans, lids")
print("- only cook alone")
print("######################")

# Main Loop
while True:
      
    rec_name = input("Enter filename or simply press enter to start recording: ")

    with picamera.PiCamera() as camera:
            camera.resolution = (1640, 1232) # full FOV
            camera.framerate = 25
            
            t_start = time.time()
            file_name = stove_type + time.strftime("_%Y-%m-%d-%H_%M_%S_", time.gmtime(t_start)) + rec_name + '.h264'
            path_video = path_videos + file_name

            camera.start_recording(path_video)

            record(t_start, rec_name)
            camera.stop_recording()
            t_stop = time.time()
            print('The video has been saved as ' + file_name)
            print('The video is {:.0f} seconds long'.format(t_stop - t_start))
