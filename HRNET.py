 
import os
import torch
import cv2
import numpy as np
from argparse import Namespace
from torchvision import transforms
import sys
import HRNet

def body_measure(front_image_path,side_image_path,angle_image_path,user_height_mm):
        # Append HRNet path to system path

    if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    
        # Add HRNet lib to sys.path dynamically
    lib_path = os.path.join(base_path, "HRNet", "lib")
    if lib_path not in sys.path:
        sys.path.append(lib_path)
    config_path = os.path.join(base_path, "HRNet", "experiments", "coco", "hrnet", "w48_384x288_adam_lr1e-3.yaml")
    

    model_weights = os.path.join(lib_path, "models", "pose_hrnet_w48_384x288.pth")
      
    
    from models import pose_hrnet
    from config import cfg
    from config import update_config
    
    # Load HRNet configuration
   
    
    cfg = cfg.clone()
    args = Namespace(cfg=config_path, opts=[], modelDir="", logDir="", dataDir="")
    
    cfg.defrost()
    update_config(cfg, args)
    cfg.freeze()

    
    # Load HRNet model
    model = pose_hrnet.get_pose_net(cfg, is_train=False)
    model.load_state_dict(torch.load(model_weights, map_location="cuda"))
    model = model.cuda()
    model.eval()
    
    print("HRNet Model Running on:", next(model.parameters()).device)
    
    # Image transformation
    hrnet_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # COCO Skeleton keypoint pairs
    COCO_SKELETON = [
        (5, 7), (7, 9),  # Left arm (Shoulder -> Elbow -> Wrist)
        (6, 8), (8, 10), # Right arm (Shoulder -> Elbow -> Wrist)
        (5, 6),          # Shoulders (Left Shoulder -> Right Shoulder)
        (11, 13), (13, 15), # Left leg (Hip -> Knee -> Ankle)
        (12, 14), (14, 16), # Right leg (Hip -> Knee -> Ankle)
        (11, 12),           # Hips (Left Hip -> Right Hip)
        (5, 11), (6, 12)    # Torso (Shoulders to Hips)
    ]
    
    def resize_with_padding(image, target_size=(384, 384)):
        """
        Resizes the image while maintaining the aspect ratio and adds padding to match the target size.
        """
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = target_size
    
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    
        image_resized = cv2.resize(image, (new_w, new_h))
    
        delta_w = target_w - new_w
        delta_h = target_h - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
    
        image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
        return image_padded, scale, left, top
    
    def extract_keypoints(image_path, save_txt_path, save_img_path):
        print(f"\nProcessing Image: {image_path}")
    
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not load {image_path}")
            return
    
        orig_h, orig_w = image.shape[:2]
        print(f"✔ Original Image Size: {orig_w} x {orig_h}")
    
        # Resize with aspect ratio preserved
        resized_image, scale, pad_x, pad_y = resize_with_padding(image)
        print(f"✔ Resized Image Size (with padding): {resized_image.shape[1]} x {resized_image.shape[0]}")
    
        input_tensor = hrnet_transform(resized_image).unsqueeze(0).cuda()
    
        with torch.no_grad():
            output = model(input_tensor)  # HRNet inference
    
        output_np = output.cpu().numpy()
        heatmap_h, heatmap_w = output_np.shape[2], output_np.shape[3]
    
        keypoints_np = np.zeros((output_np.shape[1], 3))  # num_keypoints x (x, y, confidence)
        for i in range(output_np.shape[1]):
            heatmap = output_np[0, i, :, :]
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    
            # Adjust back to original image size
            keypoints_np[i, 0] = ((x * resized_image.shape[1] / heatmap_w) - pad_x) / scale
            keypoints_np[i, 1] = ((y * resized_image.shape[0] / heatmap_h) - pad_y) / scale
            keypoints_np[i, 2] = np.max(heatmap)
    
            print(f"Keypoint {i}: Heatmap (x, y) = ({x:.2f}, {y:.2f})")
            print(f"Scaled to Original (x, y) = ({keypoints_np[i,0]:.2f}, {keypoints_np[i,1]:.2f})")
    
        np.savetxt(save_txt_path, keypoints_np, fmt="%.5f")
        print(f"✔ Saved keypoints to {save_txt_path}")
    
        visualize_skeleton(image, keypoints_np, save_img_path)
    
    def visualize_skeleton(image, keypoints, save_path):
        """
        Draws keypoints and skeleton connections on the image.
        """
        keypoints = keypoints.squeeze()
    
        for pt1, pt2 in COCO_SKELETON:
            if pt1 < len(keypoints) and pt2 < len(keypoints):  
                x1, y1 = keypoints[pt1][:2]
                x2, y2 = keypoints[pt2][:2]
    
                if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
                    print(f"❌ Skipping invalid keypoints: {(x1, y1)} -> {(x2, y2)}")
                    continue
    
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
        cv2.imwrite(save_path, image)
        print(f"✔ Saved skeleton visualization to {save_path}")
    
    
    # Define output directory
    output_dir = "C:\\Users\\AnbuC\\OneDrive\\Desktop\\input and output\\output"
    input_dir = "C:\\Users\\AnbuC\\OneDrive\\Desktop\\input and output\\input"
    
    extract_keypoints(
        front_image_path,
        os.path.join(output_dir, "front_keypoints.txt"),
        os.path.join(output_dir, "front_skeleton.jpg")
    )
    
    
    extract_keypoints(
        side_image_path,
        os.path.join(output_dir, "side_keypoints.txt"),
        os.path.join(output_dir, "side_skeleton.jpg")
    )
    extract_keypoints(
        angle_image_path,
        os.path.join(output_dir, "45.txt"),
        os.path.join(output_dir, "45_skeleton.jpg")
    )
    
    
    
    
    
    
    
    
    #BODY MEASUREMENTS CALCULATION PART
    
    
    def parse_hrnet_keypoints(txt_file):
        """Reads HRNet keypoints from a file, ignoring confidence values."""
        keypoints = []
        with open(txt_file, 'r') as f:
            for line in f.readlines():
                x, y, _ = line.strip().split()  # Ignore confidence
                keypoints.append((float(x), float(y)))
        return keypoints
    
    def pixel_distance(point1, point2, keypoints):
        """Computes Euclidean distance between two keypoints."""
        return np.sqrt((keypoints[point1][0] - keypoints[point2][0])**2 + 
                       (keypoints[point1][1] - keypoints[point2][1])**2)
    
    def calculate_ellipse_circumference(a, b):
        """Approximates ellipse circumference using Ramanujan’s formula."""
        return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
    
    def calculate_average_scale_factor(front_kp, side_kp, angle45_kp, known_height_mm):
        """Computes an average scale factor using all three views."""
        def height_from_kp(kp):
            top_y = kp[0][1]  # Nose
            ankle_y = (kp[15][1] + kp[16][1]) / 2  # Average ankle height
            return ankle_y - top_y  # Height in pixels
    
        front_scale = known_height_mm / height_from_kp(front_kp)
        side_scale = known_height_mm / height_from_kp(side_kp)
        angle45_scale = known_height_mm / height_from_kp(angle45_kp)
    
        return (front_scale + side_scale + angle45_scale) / 3  # Average scale factor
    
        
        return scale_factor
    
    def calculate_other_measurements(keypoints, scale_factor):
        """Computes shoulder width, torso length, arm length, and leg length."""
        shoulder_width_mm = pixel_distance(5, 6, keypoints) * scale_factor 
        hip_y = (keypoints[11][1] + keypoints[12][1]) / 2  # Midpoint of hips
        torso_length_mm = ((hip_y - keypoints[0][1]) * scale_factor) 
        left_arm_pixels = pixel_distance(5, 7, keypoints) + pixel_distance(7, 9, keypoints)  
        arm_length_mm = ((left_arm_pixels) * scale_factor) 
        left_leg_pixels = pixel_distance(11, 13, keypoints) + pixel_distance(13, 15, keypoints)  
        leg_length_mm = ((left_leg_pixels) * scale_factor)  
        
        return shoulder_width_mm, torso_length_mm, arm_length_mm, leg_length_mm
    
    def calculate_circumferences(front_kp, side_kp, angle45_kp, scale_factor):
        """Computes body circumferences using multi-view keypoints."""
        
        def midpoint(point1, point2, keypoints):
            """Finds midpoint between two keypoints."""
            return ((keypoints[point1][0] + keypoints[point2][0]) / 2,
                    (keypoints[point1][1] + keypoints[point2][1]) / 2)
    
        # Bicep Circumference
        #front_bicep_width = abs(front_kp[7][0] - front_kp[5][0])  # Horizontal span
        #side_bicep_depth = abs(side_kp[7][0] - side_kp[5][0])
        #angle45_bicep = abs(angle45_kp[7][0] - angle45_kp[5][0])
        #bicep_circumference = (calculate_ellipse_circumference(front_bicep_width / 2, 
                                                              # max(side_bicep_depth / 2, angle45_bicep / 2)) * scale_factor)
        
        # Chest Circumference
        front_chest_width = abs(front_kp[5][0] - front_kp[6][0])
        side_chest_depth = abs(side_kp[5][0] - side_kp[6][0])
        angle45_chest = abs(angle45_kp[5][0] - angle45_kp[6][0])
        chest_circumference = calculate_ellipse_circumference(front_chest_width / 2, 
                                                               max(side_chest_depth / 2, angle45_chest / 2)) * scale_factor
        
        # Waist Circumference
        waist_front_mid = midpoint(11, 12, front_kp)
        waist_side_mid = midpoint(11, 12, side_kp)
        waist_angle45_mid = midpoint(11, 12, angle45_kp)
        
        front_waist_width = abs(front_kp[11][0] - front_kp[12][0])
        side_waist_depth = abs(side_kp[11][0] - side_kp[12][0])
        angle45_waist = abs(angle45_kp[11][0] - angle45_kp[12][0])
        waist_circumference = calculate_ellipse_circumference(front_waist_width / 2, 
                                                               max(side_waist_depth / 2, angle45_waist / 2)) * scale_factor
        
        # Hip Circumference
        front_hip_width = abs(front_kp[11][0] - front_kp[12][0])
        side_hip_depth = abs(side_kp[11][0] - side_kp[12][0])
        angle45_hip = abs(angle45_kp[11][0] - angle45_kp[12][0])
        hip_circumference = calculate_ellipse_circumference(front_hip_width / 2, 
                                                             max(side_hip_depth / 2, angle45_hip / 2)) * scale_factor
    
        return chest_circumference, waist_circumference, hip_circumference
    
    
    
    # ---- Main Execution ----
    known_height_mm = user_height_mm
    
    # Load Keypoints
    front_keypoints = parse_hrnet_keypoints(r"C:\Users\AnbuC\OneDrive\Desktop\input and output\output\front_keypoints.txt")
    side_keypoints = parse_hrnet_keypoints(r"C:\Users\AnbuC\OneDrive\Desktop\input and output\output\side_keypoints.txt")
    angle45_keypoints = parse_hrnet_keypoints(r"C:\Users\AnbuC\OneDrive\Desktop\input and output\output\45.txt")
    
    # Compute scale factor using real height
    scale_factor = calculate_average_scale_factor(front_keypoints, side_keypoints, angle45_keypoints, known_height_mm)
    
    # Compute Measurements
    shoulder_width_mm, torso_length_mm, arm_length_mm, leg_length_mm = calculate_other_measurements(front_keypoints, scale_factor)
    chest_circ, waist_circ, hip_circ = calculate_circumferences(front_keypoints, side_keypoints, angle45_keypoints, scale_factor)
    
   
    height = round(known_height_mm / 10, 1)
    shoulder_width = round((shoulder_width_mm * 1.1) / 10, 1)
    torso_length = round((torso_length_mm * 0.6) / 10, 1)
    arm_length = round((arm_length_mm * 0.8) / 10, 1)
    leg_length = round((leg_length_mm * 0.9) / 10, 1)
    #bicep_circ = round((bicep_circ * 1.4) / 10, 1)
    chest_circ = round(chest_circ / 10, 1)
    waist_circ = round((waist_circ * 1.2) / 10, 1)
    hip_circ = round((hip_circ + 288) / 10, 1)

    return height,shoulder_width,torso_length ,arm_length,leg_length,chest_circ,waist_circ,hip_circ


