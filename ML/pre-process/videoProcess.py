import cv2
import os
import matplotlib.pyplot as plt

def extract_and_display_frames(video_path, output_folder, frame_interval=20):
    """
    Extract frames from a video, save them as images, and display them using subplots.

    Parameters:
    - video_path (str): Path to the video file.
    - output_folder (str): Directory to save the extracted images.
    - frame_interval (int): Interval to capture frames.
    """

    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    video_obj = cv2.VideoCapture(video_path)
    total_frames = int(video_obj.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Total frames in the video: {total_frames}")
    print(f"Extracting every {frame_interval}th frame...")

    frame_count = 0
    extracted_count = 0
    frames = []

    while True:
        ret, frame = video_obj.read()

        # If frame read is unsuccessful, break the loop
        if not ret:
            break

        if frame_count % frame_interval == 0:
            img_name = f"frame_{frame_count}.png"
            img_path = os.path.join(output_folder, img_name)

            # Convert frame from BGR to RGB for displaying with matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cv2.imwrite(img_path, frame)
            frames.append(frame_rgb)
            extracted_count += 1
            print(f"Extracted frame {frame_count} as {img_name}")

        frame_count += 1

    video_obj.release()

    # # Display frames using subplots
    rows = len(frames) // 2 + len(frames) % 2
    fig, axs = plt.subplots(rows, 2, figsize=(10, 10))

    for i, ax in enumerate(axs.ravel()):
        if i < len(frames):
            ax.imshow(frames[i])
            ax.set_title(f"Frame {i * frame_interval}")
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# Example usage:
video_source = "/content/Video0003.3gp"
output_dir = "extracted_frames_3gp"
extract_and_display_frames(video_source, output_dir, frame_interval=250)