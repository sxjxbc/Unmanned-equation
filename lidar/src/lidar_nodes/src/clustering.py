#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import rospy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from mapping.msg import ConeArray, Cone
import ros_numpy as rnp
import numpy as np
from scipy.spatial import cKDTree


# ============================================================
# 自适应DBSCAN参数
# ============================================================

# 公式(10)中的k近邻数量
K_NEIGHBORS = 5

# 公式(9)中的MinPts
MIN_PTS = 5


# ============================================================
# LiDAR置信度模型参数
# ============================================================

# 点数量范围
N_MIN = 3
N_MAX = 5

# 锥桶高度范围，单位：m
H_MIN = 0.15
H_MAX = 0.40

# XY平面等效半径范围，单位：m
R_MIN = 0.05
R_MAX = 0.30

# 最大有效检测距离，单位：m
D_MAX = 5.0
epsilon_min = 0.08
epsilon_max = 0.25

# 各特征权重
W_N = 0.6
W_H = 0
W_R = 0
W_D = 0.4


# ============================================================
# 原始PoseArray代码
# ============================================================

def get_PoseArray(idxs_by_cluster,
                  nparray_obstacle_xy,
                  starray_obstacle_xyz,
                  timestamp):

    clusters_keys = idxs_by_cluster.keys()

    pose_array = PoseArray()
    pose_array.header.stamp = timestamp
    pose_array.header.frame_id = "velodyne"

    for _, idx in enumerate(clusters_keys):

        pose = Pose()

        # Cluster centroid position
        cluster_indices = idxs_by_cluster[idx]

        pose.position.x = np.mean(
            nparray_obstacle_xy[cluster_indices, 0]
        )

        pose.position.y = np.mean(
            nparray_obstacle_xy[cluster_indices, 1]
        )

        pose.position.z = np.mean(
            starray_obstacle_xyz['z'][cluster_indices]
        )

        pose.orientation.w = 1.0
        pose_array.poses.append(pose)

    return pose_array


# ============================================================
# RVIZ Marker
# ============================================================

def get_cone_marker(timestamp, id, x, y, z, r, g, b, a):

    lidar_marker = Marker()

    lidar_marker.header.frame_id = "velodyne"
    lidar_marker.header.stamp = timestamp

    lidar_marker.ns = "aux_marker"
    lidar_marker.action = Marker.ADD
    lidar_marker.id = id
    lidar_marker.type = lidar_marker.CYLINDER

    lidar_marker.scale.x = 0.3
    lidar_marker.scale.y = 0.3
    lidar_marker.scale.z = 0.3

    lidar_marker.color.a = a
    lidar_marker.color.r = r
    lidar_marker.color.g = g
    lidar_marker.color.b = b

    lidar_marker.pose.orientation.w = 1.0

    lidar_marker.pose.position.x = x
    lidar_marker.pose.position.y = y
    lidar_marker.pose.position.z = z

    return lidar_marker


def get_LiDAR_marker(timestamp):

    lidar_marker = Marker()

    lidar_marker.header.frame_id = "velodyne"
    lidar_marker.header.stamp = timestamp

    lidar_marker.ns = "cone_markers"
    lidar_marker.action = Marker.ADD
    lidar_marker.id = 250
    lidar_marker.type = lidar_marker.CYLINDER

    lidar_marker.scale.x = 0.3
    lidar_marker.scale.y = 0.3
    lidar_marker.scale.z = 0.3

    lidar_marker.color.a = 1
    lidar_marker.color.r = 0
    lidar_marker.color.g = 1
    lidar_marker.color.b = 1

    lidar_marker.pose.orientation.w = 1.0

    lidar_marker.pose.position.x = 0
    lidar_marker.pose.position.y = 0
    lidar_marker.pose.position.z = 0

    return lidar_marker


def publish_markers_to_RVIZ(
        idxs_by_cluster,
        nparray_obstacle_xy,
        starray_obstacle_xyz):

    clusters_keys = idxs_by_cluster.keys()

    marker_array = MarkerArray()

    timestamp = rospy.Time.now()

    marker_array.markers.append(
        get_LiDAR_marker(timestamp)
    )

    for id, idx in enumerate(clusters_keys):

        marker = Marker()

        marker.header.frame_id = "velodyne"
        marker.header.stamp = timestamp

        marker.ns = "cone_markers"

        marker.lifetime = rospy.Duration(0.1)

        marker.action = Marker.ADD
        marker.id = id
        marker.type = marker.CYLINDER

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.4

        marker.color.a = 1
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        marker.pose.orientation.w = 1.0

        center = np.mean(nparray_obstacle_xy[idxs_by_cluster[idx]], axis=0)
        marker.pose.position.x = float(center[0])
        marker.pose.position.y = float(center[1])

        marker.pose.position.z = 0

        marker_array.markers.append(marker)

    publisher_rviz_markers.publish(marker_array)


# ============================================================
# 点云预处理
# ============================================================

def remove_field_num(a, i):

    names = list(a.dtype.names)

    new_names = names[:i] + names[i+1:]

    return a[new_names]


def preprocessing(msg):

    timestamp = msg.header.stamp

    starray_obstacle_xyz = \
        rnp.point_cloud2.pointcloud2_to_array(msg)

    starray_obstacle_xyz = starray_obstacle_xyz.reshape(-1)
    finite_mask = (np.isfinite(starray_obstacle_xyz['x']) &
                   np.isfinite(starray_obstacle_xyz['y']) &
                   np.isfinite(starray_obstacle_xyz['z']))
    starray_obstacle_xyz = starray_obstacle_xyz[finite_mask]
    nparray_obstacle_xy = np.column_stack((starray_obstacle_xyz['x'],
                                          starray_obstacle_xyz['y']))

    return timestamp, \
           nparray_obstacle_xy, \
           starray_obstacle_xyz


# ============================================================
# 自适应DBSCAN
# ============================================================

def adaptive_dbscan(points,
                    k=K_NEIGHBORS,
                    min_pts=MIN_PTS):

    """
    自适应DBSCAN

    公式(8)：
    N_e(p) = {q in P | dist(p,q) <= epsilon}

    公式(9)：
    |N_e(p)| >= MinPts

    公式(10)：
    epsilon_i =
        1/k * sum d(p_i,p_j)

    每个点根据自己的k近邻平均距离
    得到局部自适应邻域半径epsilon_i。
    """

    points = np.asarray(points)

    n_points = len(points)

    # 没有点
    if n_points == 0:
        return np.array([], dtype=int), \
               np.array([]), \
               np.array([], dtype=bool)

    # 只有一个点时无法计算k近邻
    if n_points == 1:

        labels = np.array([-1], dtype=int)
        epsilon = np.array([0.0])
        core_points = np.array([False])

        return labels, epsilon, core_points

    # 防止k超过实际点数量
    actual_k = min(
        k,
        n_points - 1
    )

    # KD tree avoids allocating an N-by-N distance matrix.
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=actual_k + 1)
    epsilon = np.clip(np.mean(distances[:, 1:], axis=1),
                      epsilon_min, epsilon_max)
    neighborhoods = [sorted(tree.query_ball_point(points[i], epsilon[i]))
                     for i in range(n_points)]

    # ========================================================
    # 公式(9)：判断核心点
    # ========================================================

    core_points = np.zeros(
        n_points,
        dtype=bool
    )

    for i in range(n_points):

        if len(neighborhoods[i]) >= min_pts:

            core_points[i] = True

    # ========================================================
    # 开始DBSCAN簇扩展
    # ========================================================

    # -1表示噪声点
    labels = np.full(
        n_points,
        -1,
        dtype=int
    )

    visited = np.zeros(
        n_points,
        dtype=bool
    )

    cluster_id = 0

    for i in range(n_points):

        # 已经访问过
        if visited[i]:
            continue

        visited[i] = True

        # 非核心点暂时作为噪声
        if not core_points[i]:
            continue

        # 当前点属于新的聚类
        labels[i] = cluster_id

        seeds = list(
            neighborhoods[i]
        )

        # 用于避免重复加入
        seed_set = set(seeds)

        j = 0

        while j < len(seeds):

            current_point = seeds[j]

            if not visited[current_point]:

                visited[current_point] = True

                # 如果该点也是核心点
                # 则继续扩展其邻域
                if core_points[current_point]:

                    for neighbor in \
                            neighborhoods[current_point]:

                        if neighbor not in seed_set:

                            seeds.append(
                                neighbor
                            )

                            seed_set.add(
                                neighbor
                            )

            # 原来被视为噪声的点
            # 如果密度可达，则加入当前簇
            if labels[current_point] == -1:

                labels[current_point] = \
                    cluster_id

            j += 1

        cluster_id += 1

    return labels, epsilon, core_points


# ============================================================
# 点数量得分
# ============================================================

def calculate_point_number_score(N_G):

    """
    s_N =
    clip(
        (N_G-N_MIN)/(N_MAX-N_MIN),
        0,1
    )
    """

    if N_MAX == N_MIN:
        return 0.0

    s_N = (
        N_G - N_MIN
    ) / float(
        N_MAX - N_MIN
    )

    return np.clip(
        s_N,
        0.0,
        1.0
    )


# ============================================================
# 高度得分
# ============================================================

def calculate_height_score(cluster_xyz):

    """
    H_G = Z_max - Z_min
    """

    z_values = np.asarray(
        cluster_xyz['z']
    ).reshape(-1)

    if len(z_values) == 0:
        return 0.0, 0.0

    H_G = \
        np.max(z_values) - \
        np.min(z_values)

    if H_G < H_MIN:

        s_H = 0.0

    elif H_G <= H_MAX:

        if H_MAX == H_MIN:

            s_H = 0.0

        else:

            s_H = (
                H_G - H_MIN
            ) / float(
                H_MAX - H_MIN
            )

    else:

        s_H = 0.0

    return np.clip(
        s_H,
        0.0,
        1.0
    ), H_G


# ============================================================
# 平均半径得分
# ============================================================

def calculate_radius_score(cluster_xy):

    """
    R_G采用XY平面包围盒等效半径：

    R_G =
    0.5 * sqrt(
        (Xmax-Xmin)^2 +
        (Ymax-Ymin)^2
    )
    """

    cluster_xy = np.asarray(
        cluster_xy
    )

    if len(cluster_xy) == 0:
        return 0.0, 0.0

    x_values = cluster_xy[:, 0]
    y_values = cluster_xy[:, 1]

    x_range = \
        np.max(x_values) - \
        np.min(x_values)

    y_range = \
        np.max(y_values) - \
        np.min(y_values)

    R_G = 0.5 * np.sqrt(
        x_range ** 2 +
        y_range ** 2
    )

    if R_G < R_MIN:

        s_R = 0.0

    elif R_G <= R_MAX:

        if R_MAX == R_MIN:

            s_R = 0.0

        else:

            s_R = (
                R_G - R_MIN
            ) / float(
                R_MAX - R_MIN
            )

    else:

        s_R = 0.0

    return np.clip(
        s_R,
        0.0,
        1.0
    ), R_G


# ============================================================
# 距离衰减得分
# ============================================================

def calculate_distance_score(cluster_xy):

    """
    d = sqrt(x_c^2 + y_c^2)

    s_d =
    clip(
        1-d/D_MAX,
        0,1
    )
    """

    cluster_xy = np.asarray(
        cluster_xy
    )

    if len(cluster_xy) == 0:
        return 0.0, 0.0

    # 聚类中心
    center_x = np.mean(
        cluster_xy[:, 0]
    )

    center_y = np.mean(
        cluster_xy[:, 1]
    )

    # 聚类中心距离雷达的距离
    d = np.sqrt(
        center_x ** 2 +
        center_y ** 2
    )

    if D_MAX <= 0:

        s_d = 0.0

    else:

        s_d = np.clip(
            1.0 -
            d / float(D_MAX),
            0.0,
            1.0
        )

    return s_d, d


# ============================================================
# LiDAR总置信度
# ============================================================

def calculate_lidar_confidence(
        N_G,
        cluster_xy,
        cluster_xyz):

    # 点数量得分
    s_N = \
        calculate_point_number_score(
            N_G
        )

    # 高度得分
    s_H, H_G = \
        calculate_height_score(
            cluster_xyz
        )

    # 平均半径得分
    s_R, R_G = \
        calculate_radius_score(
            cluster_xy
        )

    # 距离衰减得分
    s_d, d = \
        calculate_distance_score(
            cluster_xy
        )

    # ========================================================
    # 雷达总置信度加权融合公式
    #
    # C_lidar =
    # W_N*s_N +
    # W_H*s_H +
    # W_R*s_R +
    # W_D*s_d
    # ========================================================

    C_lidar = (
        W_N * s_N +
        W_H * s_H +
        W_R * s_R +
        W_D * s_d
    )

    C_lidar = np.clip(
        C_lidar,
        0.0,
        1.0
    )

    return {
        's_N': s_N,
        's_H': s_H,
        's_R': s_R,
        's_d': s_d,

        'N_G': N_G,
        'H_G': H_G,
        'R_G': R_G,
        'd': d,

        'C_lidar': C_lidar
    }


# ============================================================
# 发布PoseArray
# ============================================================

def publish_to_perception(pose_array):

    publisher_perception.publish(
        pose_array
    )


# ============================================================
# 主聚类函数
# ============================================================

def _clustering_frame(msg):

    increment_snapshot_counter()

    timestamp, \
    nparray_obstacle_xy, \
    starray_obstacle_xyz = \
        preprocessing(msg)

    # ========================================================
    # 防止空点云导致程序报错
    # ========================================================

    if len(nparray_obstacle_xy) == 0:

        rospy.logwarn(
            "Empty obstacle point cloud."
        )

        publish_to_perception(get_PoseArray({}, nparray_obstacle_xy,
                                           starray_obstacle_xyz, timestamp))
        publish_cones({}, {}, nparray_obstacle_xy, timestamp)
        return

    # ========================================================
    # 自适应DBSCAN
    #
    # 替代原来的：
    #
    # DBSCAN(
    #     eps=0.3,
    #     min_samples=5
    # )
    #
    # 现在每个点都有自己的epsilon_i
    # ========================================================

    labels, epsilon_values, \
    core_samples_mask = adaptive_dbscan(
        nparray_obstacle_xy,
        k=K_NEIGHBORS,
        min_pts=MIN_PTS
    )

    # ========================================================
    # 聚类数量
    # 忽略标签-1的噪声点
    # ========================================================

    n_clusters_ = len(
        set(labels)
    ) - (
        1 if -1 in labels else 0
    )

    # ========================================================
    # 根据标签存储每个聚类对应的点索引
    # ========================================================

    idxs_by_cluster = {
        i: []
        for i in range(n_clusters_)
    }

    for idx, label in enumerate(labels):

        if label != -1:

            idxs_by_cluster[
                label
            ].append(
                idx
            )

    clusters_keys = \
        idxs_by_cluster.keys()

    cone_idx_pseudopositions = []

    # ========================================================
    # 保留原来的伪位置获取方式
    # ========================================================

    for k in clusters_keys:

        cone_idx_pseudopositions.append(
            idxs_by_cluster[k][-1]
        )

    # ========================================================
    # 计算每个聚类的LiDAR置信度
    # ========================================================

    lidar_confidences = {}

    for cluster_id in \
            idxs_by_cluster.keys():

        cluster_indices = \
            idxs_by_cluster[
                cluster_id
            ]

        # 当前簇点数量
        N_G = len(
            cluster_indices
        )

        # 当前簇XY点
        cluster_xy = \
            nparray_obstacle_xy[
                cluster_indices
            ]

        # 当前簇XYZ点
        cluster_xyz = \
            starray_obstacle_xyz[
                cluster_indices
            ]

        # 计算LiDAR置信度
        confidence = \
            calculate_lidar_confidence(
                N_G,
                cluster_xy,
                cluster_xyz
            )

        lidar_confidences[
            cluster_id
        ] = confidence

    idxs_by_cluster = dict((key, indices) for key, indices in idxs_by_cluster.items()
                           if accept_cluster(lidar_confidences[key], cluster_limits))
    publish_cones(idxs_by_cluster, lidar_confidences, nparray_obstacle_xy, timestamp)

    # ========================================================
    # RVIZ显示
    # ========================================================

    publish_markers_to_RVIZ(
        idxs_by_cluster,
        nparray_obstacle_xy,
        starray_obstacle_xyz
    )

    # ========================================================
    # PoseArray发布
    # ========================================================

    publish_to_perception(
        get_PoseArray(
            idxs_by_cluster,
            nparray_obstacle_xy,
            starray_obstacle_xyz,
            timestamp
        )
    )


# ============================================================
# Snapshot Counter
# ============================================================

global_snapshot_counter = 0


def increment_snapshot_counter():

    global global_snapshot_counter

    global_snapshot_counter = \
        global_snapshot_counter + 1

    rospy.loginfo_throttle(1.0,
        global_snapshot_counter
    )


# ============================================================
# Main
# ============================================================

def accept_cluster(features, limits):
    values = [features[key] for key in ('N_G', 'H_G', 'R_G', 'd', 'C_lidar')]
    if not all(np.isfinite(value) for value in values):
        return False
    return (features['N_G'] >= limits['min_points'] and
            0.0 <= features['H_G'] <= limits['max_height'] and
            0.0 <= features['R_G'] <= limits['max_radius'] and
            0.0 <= features['d'] <= limits['max_distance'] and
            limits['min_confidence'] <= features['C_lidar'] <= 1.0)


def publish_cones(clusters, confidences, xy, timestamp):
    msg = ConeArray()
    msg.header.stamp = timestamp
    msg.header.frame_id = 'velodyne'
    for key in sorted(clusters):
        center = np.mean(xy[clusters[key]], axis=0)
        cone = Cone()
        cone.x, cone.y = float(center[0]), float(center[1])
        cone.color = 'unknown'
        cone.confidence = float(confidences[key]['C_lidar'])
        msg.cones.append(cone)
    publisher_cones.publish(msg)


def clustering(msg):
    started = getattr(time, 'monotonic', time.time)()
    try:
        _clustering_frame(msg)
    finally:
        elapsed = getattr(time, 'monotonic', time.time)() - started
        rospy.loginfo_throttle(1.0, 'Lidar clustering processing=%.1f ms', elapsed * 1000.0)
        if elapsed > processing_warn_seconds:
            rospy.logwarn_throttle(1.0, 'Lidar clustering exceeds budget: %.1f ms', elapsed * 1000.0)


if __name__ == '__main__':
    rospy.init_node('clustering')
    epsilon_min = float(rospy.get_param('~epsilon_min', 0.08))
    epsilon_max = float(rospy.get_param('~epsilon_max', 0.25))
    if not (np.isfinite(epsilon_min) and np.isfinite(epsilon_max) and
            0 < epsilon_min <= epsilon_max):
        raise ValueError('Invalid clustering radii')
    cluster_limits = dict(min_points=rospy.get_param('~cone_min_points', 5),
                          max_height=rospy.get_param('~cone_max_height', 0.60),
                          max_radius=rospy.get_param('~cone_max_radius', 0.30),
                          max_distance=rospy.get_param('~cone_max_distance', 5.0),
                          min_confidence=rospy.get_param('~cone_min_confidence', 0.25))
    if (not all(np.isfinite(v) for v in cluster_limits.values()) or
            cluster_limits['min_points'] < 1 or
            min(cluster_limits[k] for k in ('max_height', 'max_radius', 'max_distance')) <= 0 or
            not 0.0 <= cluster_limits['min_confidence'] <= 1.0):
        raise ValueError('Invalid cone screening parameters')
    D_MAX = cluster_limits['max_distance']
    processing_warn_seconds = float(rospy.get_param('~processing_warn_seconds', 0.10))
    if not np.isfinite(processing_warn_seconds) or processing_warn_seconds <= 0:
        raise ValueError('Invalid processing budget')
    publisher_rviz_markers = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=1)
    publisher_perception = rospy.Publisher('/clustered_points', PoseArray, queue_size=1)
    publisher_cones = rospy.Publisher('/lidar/cones', ConeArray, queue_size=1)
    subscriber_obstacle_cloud = rospy.Subscriber('/ground_segmentation/obstacle_cloud',
                                                PointCloud2, clustering, queue_size=1)
    rospy.spin()
