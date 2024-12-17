import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_motion_capture_data(csv_file):

    df = pd.read_csv(csv_file, skiprows=6)
    num_keypoints = df.shape[1] // 3
    # import pdb; pdb.set_trace()

    timestamps = df.iloc[:, 0].values
    #iloc allows you to access specific rows and columns of a DataFrame 
    #by providing the integer-based indices.
    #print(timestamps)
    keypoints = []
    for i in range(num_keypoints):
        # every 3 columns is a keypoint
        # import pdb; pdb.set_trace()
        keypoint_columns = df.iloc[:, (3 * i + 2):(3 * i + 5)].values
        keypoints.append(keypoint_columns)
    keypoints = np.stack(keypoints, axis=1)
    return keypoints, timestamps, num_keypoints
    # the matrix we want 
    # time stamps as an array

    
#the main function that does the matrix multiplication 
def transform_to_camera_coordinates(keypoints, transformation_matrix):
    """
    Transforms 3D keypoints for all cameras using the transformation matrix.
    """
    num_frames, num_keypoints, _ = keypoints.shape
    #4272 48 3
    #import pdb; pdb.set_trace()
    #add an additional one line of 1s
    keypoints_h = np.ones((num_frames, num_keypoints, 4))
    print(keypoints)
    keypoints_h[:, :, :3] = keypoints
    #print("keypoints_homogeneous",keypoints_homogeneous)
    #just time this 4272 x 48 x 4 matrix to the transformation matrix?
    #this part shall be fixed
    # transformed_keypoints_h = np.dot(keypoints_h,transformation_matrix)
    transformed_keypoints_h = keypoints_h @ np.linalg.inv(transformation_matrix)
    #print(transformed_keypoints)
    
    #transformed_keypoints = transformed_keypoints_h / transformed_keypoints_h[:, :, 3][:, :, None]
    #import pdb; pdb.set_trace()

    transformed_keypoints = transformed_keypoints_h[:, :, :3]
    # print(transformed_keypoints)
    return transformed_keypoints

def save_camera_keypoints_to_csv(camera_keypoints, timestamps, num_keypoints, output_csv):

    columns = ["frame"]
    data = [timestamps]
    for kp in range(num_keypoints):
        columns.extend([f"X_keypoint_{kp+1}", f"Y_keypoint_{kp+1}", f"Z_keypoint_{kp+1}"])
        # import pdb; pdb.set_trace()
        data.extend(camera_keypoints[:, kp, :].T)
    output_df = pd.DataFrame(np.array(data).T, columns=columns)
    output_df.to_csv(output_csv, index=False)


def main(motion_capture_csv, transformation_npy, output_csv):

    keypoints, timestamps, num_keypoints = load_motion_capture_data(motion_capture_csv)
    #print(keypoints.shape)
    
    print(f"Number of keypoints detected: {num_keypoints}")

    transformation_matrix = np.load(transformation_npy)

    for i in range(keypoints.shape[1]):
        
        plt.scatter(keypoints[0, i, 2], keypoints[0, i, 1])
    plt.savefig('mocap_cs.png', dpi=500)

    camera_keypoints = transform_to_camera_coordinates(keypoints, transformation_matrix)


    for i in range(camera_keypoints.shape[1]):
        plt.scatter(camera_keypoints[0, i, 1], camera_keypoints[0, i, 2])

    plt.savefig('test.png', dpi=500)

    # import pdb; pdb.set_trace()
    #print(camera_keypoints)

    save_camera_keypoints_to_csv(camera_keypoints, timestamps, num_keypoints, output_csv)
    #output file contains 48 keypoints


if __name__ == "__main__":
    motion_capture_csv = "t0_static_pose_001.csv"
    transformation_npy = "inverse_theia3dtransformation.npy" 
    output_csv = "3D_camera_keypoints.csv"

    main(motion_capture_csv, transformation_npy, output_csv)
