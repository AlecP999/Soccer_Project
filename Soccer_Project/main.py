from trackers import Tracker
import cv2
import numpy as np


def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def measure_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

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


class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = 999999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)

            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id
        return assigned_player

#TODO: team ball, camera, team assigner
def main():
    # read the video
    video_frames = read_video("Soccer_Project/input_videos/soccer.mp4")

    tracker = Tracker("Soccer_Project/models/best.pt")

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='Soccer_Project/stubs/track_stubs.pkl')

    tracker.add_position_to_tracks(tracks)
    
    #TODO: team ball control
    play_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = play_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
        else:
            team_ball_control.append(team_ball_control[-1])
    
    team_ball_control = np.array(team_ball_control)

    output_video_frames = tracker.draw_annotation(video_frames, tracks, team_ball_control)

    # save the video
    save_video(output_video_frames, "Soccer_Project/output_videos/soccer_with_tracking.avi")
    

if __name__ == "__main__":
    main()