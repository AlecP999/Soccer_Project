from trackers import Tracker
import cv2


def read_video(video_path):
    frames = []

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        else:
            frames.append(frame)
    return frames

def save_video(video_frames, video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    out = cv2.VideoWriter(video_path, fourcc, 24, (video_frames[0].shape[1], video_frames[0].shape[0]))
    for frame in video_frames:
        out.write(frame)
    out.release()

def main():
    # read the video
    video_frames = read_video("Soccer_Project/input_videos/soccer.mp4")

    tracker = Tracker("Soccer_Project/models/best.pt")

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='Soccer_Project/stubs/track_stubs.pkl')

    tracker.add_position_to_tracks(tracks)

    team_ball_control = []

    output_video_frames = tracker.draw_annotation(video_frames, tracks, team_ball_control)

    # save the video
    save_video(output_video_frames, "Soccer_Project/output_videos/soccer_with_tracking.avi")
    

if __name__ == "__main__":
    main()