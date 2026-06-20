import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def T_from_list(T_list):
    return np.array(T_list).reshape(4, 4)

def plot_frame(ax, T, name, axis_len=0.03):
    R = T[:3, :3]
    t = T[:3, 3]

    # plot origin
    ax.scatter(t[0], t[1], t[2])

    # axes
    colors = ['r', 'g', 'b']
    for i in range(3):
        axis = R[:, i] * axis_len
        ax.plot(
            [t[0], t[0] + axis[0]],
            [t[1], t[1] + axis[1]],
            [t[2], t[2] + axis[2]],
            color=colors[i]
        )

    ax.text(t[0], t[1], t[2], name)

def main():
    data = load_yaml("/home/alejandro/VSLAM-LAB-Benchmark/HAMLYN/rectified01/calibration.yaml")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot body frame
    T_body = np.eye(4)
    plot_frame(ax, T_body, "body", axis_len=0.05)

    # cameras
    for cam in data.get("cameras", []):
        T = T_from_list(cam["T_BS"])
        plot_frame(ax, T, cam["cam_name"])

    # imus
    for imu in data.get("imus", []):
        T = T_from_list(imu["T_BS"])
        plot_frame(ax, T, imu["imu_name"])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Sensor Extrinsics")

    ax.set_box_aspect([1,1,1])
    # ax.set_xlim(-0.2, 0.2)
    # ax.set_ylim(-0.2, 0.2)
    # ax.set_zlim(-0.2, 0.2)
    plt.show()

if __name__ == "__main__":
    main()