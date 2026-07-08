# UI
cam_count_down = 3  # seconds
ui_update_state_interval = 1 # seconds
ui_cam_max_width = 1200 # px

# Pipeline
# YOLO+rembg center-person cutout before sending the photo to ComfyUI.
enable_person_segmentation = True

# Upload server
upload_results = True
upload_gen_prefix = "cruilla26_tctb_gen_"
upload_inp_prefix = "cruilla26_tctb_inp_"