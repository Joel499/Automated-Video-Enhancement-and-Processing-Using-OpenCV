import cv2
import numpy as np
import os
from collections import deque

# Function to detect daytime or nightime through histogram
def is_nighttime(frame, dark_threshold=100, dark_proportion=0.5):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
    total_pixels = gray_frame.size
    dark_pixels = np.sum(hist[:dark_threshold])
    dark_pixel_proportion = dark_pixels / total_pixels

    # Nighttime if more than half the pixels with values less than dark threshold
    return dark_pixel_proportion > dark_proportion

# Function to increase brightness of the video
def increase_brightness(frame, value=30):
    frame_float = frame.astype(np.float64)
    # Add constant of 30 to all pixels
    frame_float += value
    # Clip the values to be in the range [0, 255]
    frame_float = np.clip(frame_float, 0, 255)
    brightened_frame = frame_float.astype(np.uint8)
    
    return brightened_frame

# Function to change non-black pixels to a lighter gray in the watermark
def blend_watermark_with_gray_color(watermark_path):
  
    watermark = cv2.imread(watermark_path, cv2.IMREAD_COLOR)
    
    # Create a mask where non-black pixels are marked as 255 (white)
    mask = np.all(watermark != [0, 0, 0], axis=-1).astype(np.uint8)
  

    gray_color = [200,200,200]
    # Convert gray_color to a BGR format for the watermark
    gray_bgr = np.array(gray_color, dtype=np.uint8)[::-1]

    # Create new watermark with zeroes
    blended_watermark = np.zeros_like(watermark)
    
    # Apply the color to each channel (BGR) of new watermark
    for i in range(3): 
        blended_watermark[:, :, i] = gray_bgr[i] * mask

    return blended_watermark, mask


# Function to create 64x64 logo
def create_camera_logo():
    
    # Create 64x64 logo with gray background
    logo = np.ones((64, 64, 3), dtype=np.uint8) * 105
    
    # Specify the color of elements within logo
    camera_body_color = (0, 0, 0)         # Black body
    lens_color = (200, 200, 200)          # Light gray lens
    lens_border_color = (0, 0, 0)        # Black lens border
    button_color = (0, 0, 255)           # Red button
    flash_color = (255, 255, 0)         # Turquoise flash
      
    # Draw the camera body
    cv2.rectangle(logo, (10, 20), (54, 50), camera_body_color, thickness=cv2.FILLED)
     
    # Draw the lens and its border
    cv2.circle(logo, (32, 35), 10, lens_color, thickness=cv2.FILLED)
    cv2.circle(logo, (32, 35), 12, lens_border_color, thickness=2)
    
    # Draw the lens reflection
    cv2.circle(logo, (32, 32), 5, (255, 255, 255), thickness=cv2.FILLED)
    
    # Draw the camera's flash
    cv2.circle(logo, (50, 22), 2, flash_color, thickness=cv2.FILLED)
    
    # Draw a red button on the camera body
    cv2.circle(logo, (32, 45), 4, button_color, thickness=cv2.FILLED)
    return logo


# Function to place watermarks and logo accordingly onto background video
def overlay_watermark_and_logo_on_frame(background_frame, watermark_frame, mask, alpha=0.5, logo_frame=None, logo_position=(180, 100)):
    # Create an inverse mask for the background region
    mask_inv = cv2.bitwise_not(mask)

    # Keep the background region 
    background_region = cv2.bitwise_and(background_frame, background_frame, mask=mask_inv)

    # Extract the watermark region
    watermark_region = cv2.bitwise_and(watermark_frame, watermark_frame, mask=mask)

    # Combine the background and watermark regions with a specified alpha for the watermark for transparency
    combined_frame = cv2.addWeighted(background_region, 1, watermark_region, alpha, 0)


    if logo_frame is not None:
        logo_height, logo_width = logo_frame.shape[:2]
        
        # Calculate the position for the logo (Top-Right of background video)
        x_offset = background_frame.shape[1] - logo_width - logo_position[0]
        y_offset = logo_position[1]
        
        # Position the dimensions for the logo on the combined frame
        combined_frame[y_offset:y_offset + logo_height, x_offset:x_offset + logo_width] = logo_frame

    return combined_frame


# Function to change fps of endscreen to match fps of the background video
def resample_endscreen_video(video_path, target_fps, output_path):
   
    # Capturing endscreen video
    cap = cv2.VideoCapture(video_path)
        
    # Obtaining endscreen video resolution
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Modifying output framerate to background video framerate with same resolution
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Writing frames of new endscreen video with different fps
        out.write(frame)
        
    cap.release()
    out.release()

# Function to apply fade-in and fade-out effects in video
def apply_fade_effects(frame, alpha):
    
    # Alters the weighted presence of background video frame and black frame on screen
    return cv2.addWeighted(frame, alpha, np.zeros_like(frame), 1 - alpha, 0)

# Main function to overlay "talking.mp4" and initiate face blurring within video
def process_and_combine_videos(overlay_path, background_path, endscreen_path, output_path, watermark1_path, watermark2_path):
  
    # Capture overlayed video "talking.mp4"
    overlay_cap = cv2.VideoCapture(overlay_path)
   
    # Capture background video 
    bg_cap = cv2.VideoCapture(background_path)

    #Capture endscreen video
    endscreen_cap = cv2.VideoCapture(endscreen_path)
    
    # Obtaining resolution and fps for background video
    bg_width = int(bg_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    bg_height = int(bg_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = bg_cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (bg_width, bg_height))

    # Check if video is recorded in nighttime for the first five frames
    initial_frame_count = 5
    nighttime_detected = False
    
    # Looping through nighttime function for initial five frames
    for frame_count in range(initial_frame_count):
        success, frame = bg_cap.read()
        if not success:
            break
        
        # Condition if video is detected at nighttime
        if is_nighttime(frame):
            
            nighttime_detected = True
            print("Video is recorded at nighttime, increasing brightness")
            break
        else:
            print("Video is recorded at daytime, no change needed")
            break
        
    # Resseting the frames back to starting frame
    bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # File for face detection 
    face_cascade = cv2.CascadeClassifier('face_detector.xml')
    
    # Initialize video capture for future frames for face detecton
    future_vid = cv2.VideoCapture(background_path)
    # Skip the first frame
    future_vid.read() 
    
    # Initialize variables for face tracking
    max_disappeared = 10 # For faces to disappear after a predefined number of frames
    buffer_size = 3 # Initialize number of frames to be stored in buffer
    face_positions = {} # Dictionary that maps each face id to a face position
    face_disappeared = {} # Dictionary for disappeared faces
    face_id = 0
    face_buffer = deque(maxlen=buffer_size)
    
    # Initialize creation of logo
    logo = create_camera_logo()
    
    # Changing colour of watermarks to gray
    watermark1_gray, mask1 = blend_watermark_with_gray_color(watermark1_path)
    watermark2_gray, mask2 = blend_watermark_with_gray_color(watermark2_path)


    # Resizing all the watermarks and masks to match the size of background video resolution
    watermark1_resized = cv2.resize(watermark1_gray, (bg_width, bg_height))
    mask1_resized = cv2.resize(mask1, (bg_width, bg_height))
    watermark2_resized = cv2.resize(watermark2_gray, (bg_width, bg_height))
    mask2_resized = cv2.resize(mask2, (bg_width, bg_height))
    
    # Keep track of the number of frames as the background video progresses
    frame_count = 0
    
    # Initialize duration of fade effect
    fade_duration = 2
    
    # Initialize the frames for fade-in effect to occur
    fade_in_frames = int(fade_duration * fps)
    
    # Initialize the frames for fade-out effect to occur
    fade_out_frames = int(fade_duration * fps)
    
    # Initialize the frames for fade-out effect to occur for endscreen
    end_fade_out_frames = int(fade_duration * fps + 30)
    
    # Obtain total number of frames in bacground video
    total_bg_frames = int(bg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Variable to start video effects after 5 seconds of background video
    effects_start_frame = int(5 * fps)
    
    # Variable to stop video effects before 3 seconds of ending background video
    effects_stop_frame = total_bg_frames - int(3 * fps)  
    
    # Calculating the frame at which the fade-out effect starts
    fade_out_start_frame = total_bg_frames - fade_out_frames 
    

    # Main loop for the effects
    while True:
        ret_bg, bg_frame = bg_cap.read()
        ret_ol, ol_frame = overlay_cap.read()

        if not ret_bg:
            break

        if not ret_ol:
            # Resseting the overlay video to starting frame and repeating if frames run out
            overlay_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_ol, ol_frame = overlay_cap.read()
            if not ret_ol:
                break

        # Increasing brightness if video is recorded at night
        if nighttime_detected:
            bg_frame = increase_brightness(bg_frame)
            
        # Reading next frame within video and appending faces to buffer
        future_ret, future_frame = future_vid.read()
        if future_ret:
            gray_future = cv2.cvtColor(future_frame, cv2.COLOR_BGR2GRAY)
            # Detecting faces in the next frame
            future_faces = face_cascade.detectMultiScale(gray_future, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            face_buffer.append(future_faces)

        # Start effects within five seconds and end effects three seconds before background video ends
        if effects_start_frame <= frame_count < effects_stop_frame:
            
            
            # Detecting faces in the current frame and adding to buffer
            gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            face_buffer.append(faces)
            
            # Combine all faces within buffer
            all_faces = [face for faces in face_buffer for face in faces]
            
            # Removing duplicate faces by checking if distance lower than width and height to assume is the same face
            unique_faces = []
            for (x, y, w, h) in all_faces:
                if not any(abs(ux - x) < w and abs(uy - y) < h for (ux, uy, uw, uh) in unique_faces):
                    unique_faces.append((x, y, w, h))
            
            # Update the new face positions
            updated_face_positions = {}
            for (x, y, w, h) in unique_faces:
                matched = False
                # Assigning face id for the faces present within the list
                for fid, (fx, fy, fw, fh) in face_positions.items():
                    if abs(fx - x) < w and abs(fy - y) < h:
                        updated_face_positions[fid] = (x, y, w, h)
                        face_disappeared[fid] = 0
                        matched = True
                        break
                # Adding face_id for when a new face has been detected
                if not matched:
                    updated_face_positions[face_id] = (x, y, w, h)
                    # Initialize face disappeared for new detected face
                    face_disappeared[face_id] = 0
                    # Increment face_id by 1
                    face_id += 1

            # Increment disappearance count by 1 for unmatched faces in the list
            for fid in face_positions.keys():
                if fid not in updated_face_positions:
                    face_disappeared[fid] += 1
                    # If count is more than predefined limit face is removed 
                    if face_disappeared[fid] > max_disappeared:
                        del face_disappeared[fid]
                    else:
                        # Adding the faces that were not detected by within predefined limit to list
                        updated_face_positions[fid] = face_positions[fid]
            
            # Update original list of faces with new detected face positions
            face_positions = updated_face_positions
            
            # Blur faces based on positions within values in face positions dictionary
            for (x, y, w, h) in face_positions.values():
                face_roi = bg_frame[y:y+h, x:x+w]
                blurred_face = cv2.GaussianBlur(face_roi, (23, 23), 30)  # Adjust parameters for better blur
                bg_frame[y:y+h, x:x+w] = blurred_face
                
            # Resize the overlay video with predefined resolution
            resized_ol_frame = cv2.resize(ol_frame, (480, 360))
            
            # Convert to from BGR to HSV
            hsv = cv2.cvtColor(resized_ol_frame, cv2.COLOR_BGR2HSV)
            
            # Define the range of pixel values for green screen
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Inverse the mask so green screen becomes black
            mask_inv = cv2.bitwise_not(mask)
            
            # Variables for position of overlay
            ol_height, ol_width = resized_ol_frame.shape[:2]
            y1 = bg_height - ol_height
            x1 = 0 
            y2 = y1 + ol_height
            x2 = x1 + ol_width

            # Define region of interest of overlay on background video
            bg_roi = bg_frame[y1:y2, x1:x2]
            bg_roi = bg_roi.astype(float)
            mask_float = mask_inv.astype(float) / 255.0

            # Blend the masked overlay and background region of interest in all 3 channels
            for c in range(0, 3):
                bg_roi[:, :, c] = (mask_float * resized_ol_frame[:, :, c] + (1 - mask_float) * bg_roi[:, :, c])
                
    
            # Place background region of interest onto background frame
            bg_frame[y1:y2, x1:x2] = bg_roi.astype(np.uint8)

            # Switch watermarks at intervals of five seconds of the video
            watermark_resized = watermark1_resized if (frame_count // int(fps * 5)) % 2 == 0 else watermark2_resized
            mask_resized = mask1_resized if (frame_count // int(fps * 5)) % 2 == 0 else mask2_resized

            combined_frame = overlay_watermark_and_logo_on_frame(bg_frame, watermark_resized, mask_resized, logo_frame=logo)
        else:
            # Frame with no effects
            combined_frame = bg_frame

        # Condition for fade-in effect at beginning of background video
        if frame_count < fade_in_frames:
            # Value of alpha incrementing to 1 (non-transparent)
            alpha = frame_count / fade_in_frames
            combined_frame = apply_fade_effects(combined_frame, alpha)

        # Condition for fade-out effect before background video ends
        if frame_count >= fade_out_start_frame:
            fade_frame_count = frame_count - fade_out_start_frame
            # Value of alpha decreasing to 0 (transparent)
            alpha = 1 - (fade_frame_count / fade_out_frames)
            combined_frame = apply_fade_effects(combined_frame, alpha)

        out.write(combined_frame)
        frame_count += 1
        
    # Apply fade-out effect for the final part of the background video
    while frame_count < total_bg_frames:
        ret_bg, bg_frame = bg_cap.read()
        if not ret_bg:
            break

        if frame_count >= fade_out_start_frame:
            fade_frame_count = frame_count - fade_out_start_frame
            alpha = 1 - (fade_frame_count / fade_out_frames)
            combined_frame = apply_fade_effects(bg_frame, alpha)
            out.write(combined_frame)

        frame_count += 1

    # Fade-in effect for endscreen
    for fade_frame_count in range(fade_in_frames):
        ret_es, es_frame = endscreen_cap.read()
        if not ret_es:
            break
        alpha = fade_frame_count / fade_in_frames
        # Resize endscreen frame to match resolution of background frame
        endscreen_resized = cv2.resize(es_frame, (bg_width, bg_height))
        combined_frame = apply_fade_effects(endscreen_resized, alpha)
        out.write(combined_frame)
        
        endscreen_fps = endscreen_cap.get(cv2.CAP_PROP_FPS)
        # Condition if endscreen and background dont match 
        if fps != endscreen_fps:
            # Create new endscreen with adjusted fps
            temp_endscreen_path = 'temp_endscreen.mp4'
            resample_endscreen_video(endscreen_path, fps, temp_endscreen_path)
            endscreen_path = temp_endscreen_path
            endscreen_cap = cv2.VideoCapture(endscreen_path)


    while endscreen_cap.isOpened():
        ret, frame = endscreen_cap.read()
        if not ret:
            break
        
        # Resize endscreen frame with altered fps to match background resolution
        resized_frame = cv2.resize(frame, (bg_width, bg_height))
        out.write(resized_frame)
    
    # Fade-out effect for endscreen
    for end_frame_count in range(end_fade_out_frames):
        alpha = 1 - (end_frame_count / fade_out_frames)
        combined_frame = apply_fade_effects(resized_frame, alpha)
        out.write(combined_frame)
        
    endscreen_cap.release()
    out.release()

    print(f"Processed and combined video saved as {output_path}")

# Paths for the videos and watermark images
overlay_path = 'talking.mp4'
background_paths = ['street.mp4', 'office.mp4', 'traffic.mp4', 'singapore.mp4']
endscreen_path = 'endscreen.mp4'
watermark1_path = 'watermark1.png'
watermark2_path = 'watermark2.png'


# Process videos with seperate names accordingly
for bg_path in background_paths:
    base_name = os.path.splitext(os.path.basename(bg_path))[0]
    processed_output_path = f"processed_{base_name}.mp4"
    process_and_combine_videos(overlay_path, bg_path, endscreen_path, processed_output_path, watermark1_path, watermark2_path)
