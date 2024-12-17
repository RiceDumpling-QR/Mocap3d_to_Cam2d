import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import cv2

np.set_printoptions(threshold=np.inf)

#referece:
#https://www.geeksforgeeks.org/mapping-coordinates-from-3d-to-2d-using-opencv-python/

def parse_camera_calibration(calibration_file):
    tree = ET.parse(calibration_file)
    root = tree.getroot()
    cameras = []

    for camera in root.findall(".//camera"):
        intrinsic = camera.find("intrinsic")
        transform = camera.find("transform")

        camera_params = {
            #the name of the camera
            "serial": camera.get("serial"),
            "active": camera.get("active"),
            #get rotation matrix
            "rotation": np.array([
                [float(transform.get("r11")), float(transform.get("r12")), float(transform.get("r13"))],
                [float(transform.get("r21")), float(transform.get("r22")), float(transform.get("r23"))],
                [float(transform.get("r31")), float(transform.get("r32")), float(transform.get("r33"))],
            ]),

            "translation": np.array([
                float(transform.get("x")),
                float(transform.get("y")),
                float(transform.get("z")),
            ]),
            
            "intrinsic": {
                "focal_length_u": float(intrinsic.get("focalLengthU")),
                "focal_length_v": float(intrinsic.get("focalLengthV")),
                "center_u": float(intrinsic.get("centerPointU")),
                "center_v": float(intrinsic.get("centerPointV")),
            }
            #ommit the distortion content because not used here 
        }
        cameras.append(camera_params)
    
    return cameras

def process_keypoints_file(keypoints_file_path):
    df = pd.read_csv(keypoints_file_path)
    keypoint_columns = [col for col in df.columns if 'X_keypoint_' in col]
    #print("keypoint",keypoint_columns)
    all_keypoints_3d = {}
    for i in range(1, len(keypoint_columns)+1):
        for index, row in df.iterrows():
            x = row[f"X_keypoint_{i}"]
            y = row[f"Y_keypoint_{i}"]
            z = row[f"Z_keypoint_{i}"]


            keypoint = f"keypoint_{i}"
            if keypoint in all_keypoints_3d:
                all_keypoints_3d[keypoint].append([x, y, z])
            else:
                all_keypoints_3d[keypoint] = [[x, y, z]]
    #print(all_keypoints_3d)
    return all_keypoints_3d


def parse_keypoints_npy_file(npy_file_path):
    keypoints_3d_data = np.load(npy_file_path)
    if len(keypoints_3d_data.shape) != 3 or keypoints_3d_data.shape[2] != 3:
        raise ValueError("The .npy file does not have the expected shape (num_frames, num_keypoints, 3).")

    num_frames, num_keypoints, _ = keypoints_3d_data.shape
    all_keypoints_3d = {}
    for keypoint_index in range(num_keypoints):
        keypoint_name = f"keypoint_{keypoint_index + 1}" 
        all_keypoints_3d[keypoint_name] = []
        for frame_index in range(num_frames):
            x, y, z = keypoints_3d_data[frame_index, keypoint_index]
            all_keypoints_3d[keypoint_name].append([x, y, z])
    return all_keypoints_3d


def compute_min_max(keypoints):
    num_frames = len(next(iter(keypoints.values())))
    results = np.zeros((num_frames, 4))
    for frame_idx in range(num_frames):
        frame_data = np.array([
            kp[frame_idx] for kp in keypoints.values() 
            if kp[frame_idx] is not None and len(kp[frame_idx]) > 0
        ])


        frame_data = frame_data.squeeze()
        x_min = np.min(frame_data[:, 0])
        x_max = np.max(frame_data[:, 0])
        y_min = np.min(frame_data[:, 1])
        y_max = np.max(frame_data[:, 1])
        results[frame_idx] = [x_min, x_max, y_min, y_max]
   # print(results)
    
    return results


def project_3d_to_2d(keypoints_3d, cameras):
    results = {}

    #building up the camera project matrix 
    for camera in cameras:
        #intrinsic matrix
        K = np.array([
            [camera["intrinsic"]["focal_length_u"], 0, camera["intrinsic"]["center_u"]],
            [0, camera["intrinsic"]["focal_length_v"], camera["intrinsic"]["center_v"]],
            [0, 0, 1]
        ])
        # print('k: ',K)
        R = camera["rotation"]
        #3x3
        t = camera["translation"].reshape(3, 1)
        
        Rt = np.hstack((R,t))
        P = np.dot(K,Rt)
        
        camera_serial = camera["serial"]
        #get camera serial name
        results[camera_serial] = {}


        for keypoint_name, keypoint_3d in keypoints_3d.items():
            keypoint_3d = np.matrix(keypoint_3d)
            # print(keypoint_name)
            # print(keypoint_3d.shape)
            #print(keypoint_name)
            #kp3dh: 4271x3
            keypoints_3d_h = np.hstack((keypoint_3d, np.ones((keypoint_3d.shape[0], 1))))
            #print(keypoints_3d_h)
            #kp3dh: 4271x4
            #matrix: 3x4
            keypoints_2d_h = np.dot(P, keypoints_3d_h.T).T
            #print(keypoints_2d_h)
            #4721x3 with the final column not normalized 
            keypoints_2d = keypoints_2d_h / keypoints_2d_h[:, 2]
            #normallized the final column and changed to 4721x2
            keypoints_2d[:, 1] = -keypoints_2d[:, 1]
            results[camera_serial][keypoint_name] = keypoints_2d[:,:2]

    return results






if __name__ == "__main__":

    calibration_file = "calibration.txt"
    cameras = parse_camera_calibration(calibration_file)
    i = 0
    
    keypoints_3d_file = "3D_camera_keypoints.csv"
    keypoints_3d = process_keypoints_file(keypoints_3d_file)
    #keypoints_3d = parse_keypoints_npy_file("t0_static_triangulated_keypoints.npy")

    keypoints_2d = project_3d_to_2d(keypoints_3d, cameras)
    #print(keypoints_2d)


    #print(type(keypoints_2d))
    cams = []
    for key in keypoints_2d:
        cams.append(key)
    # print(cams[3])
    cam = cams[len(cams)-1]
    keypoints_dict = keypoints_2d[cam]
    frame_index = 0
    x_coords = []
    y_coords = []
    for keypoint_id in keypoints_dict:
        keypoint_data = keypoints_dict[keypoint_id] 
        point = keypoint_data[frame_index]
        if point.ndim == 2:
            #print(type(point))
            point = np.array(point).flatten()
        #print(point)
        x, y = point
        x_coords.append(x)
        y_coords.append(y)

    from PIL import Image
    image = Image.open("sample_image.jpg")

    image_width, image_height = image.size

    x_coords_normalized = [(x - min(x_coords)) / (max(x_coords) - min(x_coords)) * (image_width*0.13) + 1290 for x in x_coords]
    y_coords_normalized = [(y - min(y_coords)) / (max(y_coords) - min(y_coords)) * (image_height*0.55) + 253 for y in y_coords]

    # Flip the y-axis
    y_coords_normalized = [image_height - y for y in y_coords_normalized]


    # Plot the image and overlay points
    x_min = min(x_coords_normalized)
    x_max = max(x_coords_normalized)
    y_min = min(y_coords_normalized)
    y_max = max(y_coords_normalized)

    # Plot the image and overlay points
    plt.figure(figsize=(12, 6))
    plt.imshow(image)  # Replace 'image' with your loaded image
    plt.scatter(x_coords_normalized, y_coords_normalized, color='red', label='Keypoints')

    # Draw the bounding box
    plt.plot([x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min],
            color='blue', linewidth=2, label='Bounding Box')

    # Add labels and save the image
    plt.legend()
    plt.axis("off")
    plt.savefig('new_with_bbox.png', dpi=500)

    # print(x_coords)
    # print(y_coords)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(image_rgb)
    # plt.scatter(x_coords, y_coords, color='red', label='Keypoints', s=1) 
    # plt.axis('off')
    # plt.legend()
    # plt.show()

    # second_key = next(iter(keypoints_2d[first_key]))
    output_file = "bbox.txt"
    with open(output_file, "w") as f:
        for camera in keypoints_2d:
            #print(camera)
            keypoint_data = keypoints_2d[camera]
            #print(keypoint_data.shape)
            # results = compute_min_max(keypoint_data)
            # #print(results)
            # f.write(f"Camera: {camera}\n")
            # np.savetxt(f, results, fmt="%.6f", header="x_min x_max y_min y_max", comments="")
            # f.write("\n")



