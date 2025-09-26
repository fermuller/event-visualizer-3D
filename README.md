# event-visualizer-3D
# Creates a video in mpeg format or animated gif from the input in the file folder 
# By default, it is "mpeg", if ffmpeg not installed or the "gif" is set, the output format will be gif
#To set it to gif call : events_animation  --format gif
#
#The essential function create_events_animation builds the events animations and saves it to an mp4 or gif  file using Matplotlib. 
#The main parameters of this function are arrays of {t, x, y ,p} values from events. Time must be provided in microseconds.
#
#Use "--help" flag to check the different parameters that can modify the behaviour of the animation. 
#This event_visualizer-3D script is prepared to be fed with a CSV file containing four columns (x, y, t, and p) where each row includes these values for each event.
#
#Required packages:
# numpy
# matplotlib
# pillow
# pandas 
