#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2023 by Daniel Deniz, University of Granada. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# modified by Cornelia Fermuller to include GIF output if ffmpeg is not available
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import argparse
matplotlib.use("Agg")



def create_events_animation(t, x, y, p, animation_duration_s, filepath="animation", cmap="polarity", front_visualization=False, format="mpeg"):
    """Create an animation with asynchronous events plotted in a 3D scatter plot. Events are included in the visualization iteratively based on timestamp. 
    
    :param t: array of timestamps of events in microseconds
    :param x: array of x positions of events
    :param y: array of y positions of events
    :param p: array of polarity of events
    :param animation_duration_s: desired duration (in seconds) of created animation. Do not need to match with real event time window.
    :param filepath: destination path of generated video animation
    :param cmap: color map for ploting events. "polarity" colors the events in blue or red depending on polarity. Other cmaps such as "plasma" or "viridis" can be used.
    :param front_visualization: bool, True generates a visualization from perpendicular perpendicular to Time axis (such as a TS).
    :param format: str, either "mpeg" or "gif" to specify the output format. Defaults to "mpeg".
    :return:
    References:
        When Do Neuromorphic Sensors Outperform cameras? Learning from Dynamic Features in 2023 CISS:
        [Deniz et al., 2023](https://ieeexplore.ieee.org/abstract/document/10089678).
    """
    assert t.shape[0] == x.shape[0]  == y.shape[0] == p.shape[0], "{t, x, y, p} parameters must be an arrays with the same lenght"
    
    # We set the frame of the animation video to 20 FPS, it can be modified if necessary
    fps = 20
    
    # Number of frames required to get an animation with a duration specified by animation_duration_s
    n_frames = int(animation_duration_s * fps)
    
    # Creation of Pandas dataframe with all components of event data: time (t), x (position), y (position) and p (polarity)
    df = pd.DataFrame.from_dict({"t": t,
                                 "x": x,
                                 "y": y,
                                 "p": p})


    # Microseconds to seconds conversion. Also, time of first event is set to 0.
    df.t = (df.t - df.t.min())*1e-6
    
    # Time span to add new events in each new generated animation frame
    equal_time_span = df.t.max() / n_frames

    # Get the number of events that will be represented at each time span. (Elements in the list are used as index then) 
    n_elements = [np.argmax(df.t >= i*equal_time_span) for i in range(n_frames)]

    # Matplolib 3D Figure definition
    
    plt.tight_layout()
    fig = plt.figure(figsize=(8, 8 if front_visualization else 6.5))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Define the update function for the animation
    def update_graph(frame):
        ax.clear()
        ax.set_xlabel('Time (s)', labelpad=7)
        ax.set_xlim([0, df['t'].max()])
        
        ax.set_ylabel('X')
        ax.set_yticks([0, df['x'].max()])
        ax.set_ylim([0, df['x'].max()])
        
        ax.set_zlabel('Y')
        ax.set_zticks([0, df['y'].max()])
        ax.set_zlim([0, df['y'].max()])
          
        ax.scatter(df['t'][:n_elements[frame]], 
                   df['x'][:n_elements[frame]], 
                   df['y'][:n_elements[frame]], 
                   c=df['p'][:n_elements[frame]] if cmap =="polarity" else df['t'][:n_elements[frame]], 
                   cmap="bwr" if cmap =="polarity" else cmap, 
                   alpha=0.5,           # Points transparency 
                   s=0.4,               # Points size 
                   edgecolor='black',   # Points edge color 
                   linewidth=0.05)      # Points edge line width
        
        # Makes Time (s) axis larger than x and y positions axis. For example [1, 1, 1] will output a cube.
        ax.set_box_aspect([4, 1.5, 1.5])
        
        # Set pane colors to transparent using the modern API
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Set grid lines to be visible
        ax.grid(True)
        
        # This sets the point of view for the 3D scatter plot. 
        if front_visualization:
            ax.view_init(elev=8, azim=0)
            ax.set_xticks([0, df['t'].max()])
            ax.set_zticks([])
            ax.set_yticks([])
            
        else: 
            ax.view_init(elev=8, azim=290)           
            
        return ax

    # Create the animation with proper interval in milliseconds
    interval_ms = int(1000 / fps)  # Convert FPS to milliseconds per frame
    
    # Create animation with blit=False to ensure proper rendering
    ani = animation.FuncAnimation(fig, update_graph, frames=n_frames, 
                                interval=interval_ms, blit=False, repeat=False)
    
    # Draw the first frame to ensure the animation is properly initialized
    fig.canvas.draw()
    
    # Save animation in the specified format
    try:
        # Check if ffmpeg is available
        ffmpeg_available = 'ffmpeg' in animation.writers.list()
        
        # Use MP4 only if ffmpeg is available and format is mpeg
        if ffmpeg_available and format.lower() == "mpeg":
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='DanielDeniz'), bitrate=1800)
            ani.save(f'{filepath}.mp4', writer=writer)
            print(f"Successfully saved animation as {filepath}.mp4")
        # Fall back to GIF if either ffmpeg is not available or GIF was requested
        elif not ffmpeg_available or format.lower() == "gif":
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=fps)
            plt.draw()  # Ensure figure is drawn before saving
            ani.save(f'{filepath}.gif', writer=writer)
            print(f"Successfully saved animation as {filepath}.gif")
            if not ffmpeg_available and format.lower() == "mpeg":
                print("Note: FFmpeg not available, falling back to GIF format")
        else:
            raise ValueError(f"Unsupported format: {format}. Use either 'mpeg' or 'gif'.")
    except Exception as e:
        print(f"Error: Could not save animation as {format}. Error: {e}")
        raise
    finally:
        # Clean up
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates an animation of an events stream in a scatter 3D plot')
    parser.add_argument('--i', dest='csvfile', type=str, default="events_sample.csv",
                        help='Path to csv file containing x, y, t and p columns where each row represent an event. Time must be provided in microseconds.')
    parser.add_argument('--duration', dest='duration', type=float,
                        default=4,
                        help='Time in seconds of the desired duration of the animation (video) does not need to match with events time window')
    parser.add_argument('--dst', default="./animation_events", type=str,
                        help='Destination path of generated animation video (mp4)')
    parser.add_argument('--cmap', default="plasma", type=str,
                        help='Defines the matplotlib color map used to color events based on timestamp. cmap "polarity" uses blue and red to color events depending on polarity value [-1, 1]')
    parser.add_argument('--front', default=False, type=bool,
                        help='Whether to show a front or horizontal view of events based on time. Try both to check.')
    parser.add_argument('--format', default="mpeg", type=str, choices=['mpeg', 'gif'],
                        help='Output format for the animation. Either "mpeg" (default) or "gif".')
    
    args = parser.parse_args()

    # Load the data from the CSV file into a pandas dataframe
    df = pd.read_csv(args.csvfile)

    # Check that the required columns are present
    required_columns = ["x", "y", "t", "p"]
    for col in required_columns:
        if col not in df.columns:
            print(f"Column '{col}' is missing from the DataFrame. Please, make sure column names are: x, y, t and p")
            exit(1)
    
    create_events_animation(t=df.t, x=df.x, y=df.y, p=df.p, 
                            animation_duration_s=args.duration, 
                            filepath=args.dst, cmap=args.cmap, 
                            front_visualization=args.front,
                            format=args.format)