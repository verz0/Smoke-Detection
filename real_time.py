from ultralytics import YOLO
import cv2
import math
import pygame
import moviepy.editor as mpe

# Initialize the alarm sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('alarm.wav')

# Initialize video capture
video = "VID_20240514202240.mp4"
cap = cv2.VideoCapture(video)
cap.set(3, 640)
cap.set(4, 480)

# Load YOLO model
model = YOLO("best(8s).pt")

# Object classes
classNames = ["yellow smoke"]

# Alarm settings
SMOKE_ALARM_THRESHOLD = 20  # Number of consecutive frames with smoke to trigger alarm
SMOKE_ALARM_BUFFER = 5  # Number of consecutive frames without smoke to stop the alarm
smoke_frames_count = 0
no_smoke_buffer = 0
alarm_playing = False

# Alarm duration (seconds)
ALARM_DURATION = 10  

# Timer for alarm duration
alarm_timer = 0

# Video writer setup
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_output_path = 'output.avi'
out = cv2.VideoWriter(video_output_path, fourcc, 30, (frame_width, frame_height))

while True:
    success, img = cap.read()
    if not success:
        break

    # Perform object detection
    results = model(img, stream=True)

    smoke_detected = False

    # Process each detection result
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red color, thin rectangle

            # Confidence score
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence:", confidence)

            # Class name
            cls = int(box.cls[0])
            class_name = classNames[cls]
            print("Class name:", class_name)

            # Display class name
            org = (x1, y1 - 10)  # Adjust text position above the box
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5  # Small font size
            color = (255, 0, 0)  # Blue color for text
            thickness = 1

            cv2.putText(img, class_name, org, font, fontScale, color, thickness)

            # Check if yellow smoke is detected
            if class_name == "yellow smoke":
                smoke_detected = True

    # Update smoke frames count and buffer
    if smoke_detected:
        smoke_frames_count += 1
        no_smoke_buffer = 0
        alarm_timer = ALARM_DURATION  # Reset alarm timer on smoke detection
    else:
        no_smoke_buffer += 1

    # Trigger or stop the alarm based on detection and timer
    if alarm_timer > 0:  # Check if alarm timer is active
        if not alarm_playing:
            print("Alarm triggered!")
            alarm_sound.play(loops=-1)  # Loop the alarm sound
            alarm_playing = True
        alarm_timer -= 1  # Decrement alarm timer
    else:
        if alarm_playing:
            print("Alarm stopped!")
            alarm_sound.stop()  # Stop the alarm sound
            alarm_playing = False
        smoke_frames_count = 0  # Reset smoke frames count after alarm stops

    # Debugging information
    print(f"Smoke frames count: {smoke_frames_count}, No smoke buffer: {no_smoke_buffer}, Alarm playing: {alarm_playing}")

    # Write the frame to the output video
    out.write(img)

    # Display annotated frame
    cv2.imshow('cam', img)

    # Check for 'q' key to quit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

# Add the alarm sound to the video using moviepy
video_clip = mpe.VideoFileClip(video_output_path)
audio_clip = mpe.AudioFileClip("alarm.wav").subclip(0, video_clip.duration)
combined = video_clip.set_audio(audio_clip)
combined.write_videofile("output_with_alarm.mp4", codec='libx264', fps=30)
