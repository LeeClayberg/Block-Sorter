from math import dist, cos, sin, pi

import numpy as np
import open3d as o3d


# Rotates a point around an origin by a given angle (in radians)
def rotate(origin, point, angle):
    ox, oy, _ = origin
    px, py, pz = point

    qx = ox + cos(angle) * (px - ox) - sin(angle) * (py - oy)
    qy = oy + sin(angle) * (px - ox) + cos(angle) * (py - oy)
    return [qx, qy, pz]


# *** Perception pipeline mocking ***

# Read .ply file containing one frame from the depth camera
input_file = "boxblock.ply"
pcd = o3d.io.read_point_cloud(input_file)

# Get clusters for box and block
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
max_label = labels.max()

# Divide points into cluster lists
pre_clusters = dict()
for point, label in zip(pcd.points, labels):
    if label not in pre_clusters.keys():
        pre_clusters[label] = []
    pre_clusters[label].append(list(point))

# Calculate clusters info
clusters = []
for cluster in pre_clusters.values():
    count = len(cluster)
    max_z = 0
    for point in cluster:
        max_z = max(max_z, list(point)[2])
    top_surface = []
    for point in cluster:
        if abs(point[2] - max_z) < 0.0075:
            top_surface.append([point[0], point[1], max_z])
    centroid = [sum(x) / len(x) for x in zip(*top_surface)]
    name = f"cluster_{len(clusters)}"
    clusters.append({"name": name, "location": centroid, "num_points": len(cluster), "surface_points": top_surface})

# *** Perception pipeline mocking ***

# Get box and blocks
box = clusters[0]
for cluster in clusters:
    # Box is assumed to have the most points because it is the largest
    if cluster["num_points"] > box["num_points"]:
        box = cluster
blocks = list(filter(lambda clstr: clstr["name"] != box["name"], clusters))
print("[BlockSorter DEBUG] Box and block separation")

# Get negative from box points
# Calculate grid
min_x, max_x = box["location"][0], box["location"][0]
min_y, max_y = box["location"][1], box["location"][1]
avg_z = 0
point_density = 100
for x, y, z in box["surface_points"]:
    min_x = min(min_x, x)
    max_x = max(max_x, x)
    min_y = min(min_y, y)
    max_y = max(max_y, y)
    avg_z += z / len(box["surface_points"])
change_x = (max_x - min_x) / point_density
change_y = (max_y - min_y) / point_density
grid = [[min_x + x * change_x, min_y + y * change_y, avg_z] for x in range(point_density + 1) for y in
        range(point_density + 1)]
# subset of surface points
surface_points = box["surface_points"][::5]
# Filter out the points in grid that are not near any surface points
negative = []
for grid_x, grid_y, grid_z in grid:
    check = True
    for x, y, _ in surface_points:
        if dist([grid_x, grid_y], [x, y]) < max(change_x, change_y):
            check = False
    if check:
        negative.append([grid_x, grid_y, grid_z])
print("[BlockSorter DEBUG] Box surface negative generation")

# Create clusters from negative
#
# The points in the queue are in the current cluster. They are used to find other potential
# cluster points. Once the queue is empty a new cluster is started. Clusters located on
# the outside of the box surface should be discarded.
h_clusters = []
queue = []
corners = [[min_x, min_y, avg_z], [min_x, max_y, avg_z], [max_x, min_y, avg_z], [max_x, max_y, avg_z]]
outer_cluster = False
while len(negative) > 0:
    # Use the points from the queue first (continue building current cluster)
    if len(queue) > 0:
        point = queue.pop(0)
        h_clusters[-1].append(point)
        # Check if point is a corner of the grid (tells whether cluster is outside the box surface)
        if len([corner for corner in corners if dist(corner, point) < max(change_x, change_y)]) > 0:
            outer_cluster = True
        i = 0
        # Add point's neighbors to queue
        while i < len(negative):
            if dist(point, negative[i]) < max(change_x, change_y) * 2:
                queue.append(negative[i])
                negative.pop(i)
            else:
                i += 1
    else:
        queue.append(negative.pop(0))
        if len(negative) > 0:
            if outer_cluster:
                # Discard cluster
                h_clusters[-1] = []
            else:
                # Create new cluster
                h_clusters.append([])
            outer_cluster = False

# Get cluster centroids and find the three holes on the top (clusters with the most points)
centroids = [[sum(x) / len(x) for x in zip(*cluster)] for cluster in h_clusters]
h_clusters = list(map(lambda cent, p: {"centroid": cent, "points": p}, centroids, h_clusters))
h_clusters = sorted(h_clusters, key=(lambda h: len(h["points"])), reverse=True)
h_clusters = h_clusters[:3]
print("[BlockSorter DEBUG] Negative clustering")

# Modified RANSAC for finding destination for each block
block_destinations = []
for block in blocks:
    best_position, best_angle, best_score = block["location"], 0, 0
    for cluster in h_clusters:
        p_c = [cluster["centroid"][0] - block["location"][0], cluster["centroid"][1] - block["location"][1],
               cluster["centroid"][2] - block["location"][2]]
        for i in range(180):
            angle = i * pi / 90
            # Rotate points around center of cluster
            rotated_points = [rotate(block["location"], point, angle) for point in block["surface_points"]]
            # Translate the points so the center of the block and hole are aligned
            translated_points = list(map(lambda p: [p[0] + p_c[0], p[1] + p_c[1], p[2] + p_c[2]], rotated_points))
            # Calculate score with inliers and outliers
            score = 0
            for b_point in translated_points[::2]:
                check = False
                for c_point in cluster["points"][::2]:
                    if dist(b_point, c_point) < max(change_x, change_y):
                        # This point on the block is close enough to a point on the hole cluster
                        check = True
                        break
                score += 1 if check else -1
            if score > best_score:
                best_score = score
                best_angle = angle
                best_position = cluster["centroid"]
    # Clip the angle change between -pi and pi (so joint limits will not be exceeded)
    best_angle = best_angle - 2 * pi if best_angle > pi else best_angle
    block_destinations.append({"position": best_position, "angle_change": best_angle})
print("[BlockSorter DEBUG] RANSAC")

# *** Displaying the results ***

# Block display destinations
new_block_points = []
for block, dest in zip(blocks, block_destinations):
    a, b, c = dest["position"]
    d, e, f = block["location"]
    n_x, n_y, n_z = [a - d, b - e, c - f]
    translated = [[p[0] + n_x, p[1] + n_y, p[2] + n_z] for p in block["surface_points"]]
    rotated_points = [rotate(dest["position"], p, dest["angle_change"]) for p in translated]
    new_block_points.append(rotated_points)

points = []
colors = []
# Block points
for idx, new_points in enumerate(new_block_points):
    points += new_points
    colors += [[1.0, 0.23, 0] for _ in new_points]
# Hole cluster points
for cluster in h_clusters:
    points += cluster["points"]
    colors += [[0, 0.69, 1.0] for _ in cluster["points"]]
# Box surface points
points += box["surface_points"]
colors += [[0.23, 0.77, 0.23] for _ in box["surface_points"]]

# Show points with open3d's visualization tool
points = np.array(points)
colors = np.array(colors)
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])

# *** Displaying the results ***
