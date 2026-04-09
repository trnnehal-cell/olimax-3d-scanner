#!/usr/bin/env python3
"""
Olimax Scanner Backend — SOR, ICP, SLAM via WebSocket

Install:
    pip install websockets open3d numpy

Run:
    python backend_server.py

The frontend connects to ws://localhost:8765
"""

import asyncio
import json
import time
import numpy as np

try:
    import open3d as o3d
    print("[OK] Open3D", o3d.__version__)
except ImportError:
    print("[ERROR] pip install open3d")
    exit(1)

try:
    import websockets
    print("[OK] websockets")
except ImportError:
    print("[ERROR] pip install websockets")
    exit(1)


# ============================================================
# HELPERS
# ============================================================

def pts_to_pcd(pts):
    """Nx6 [pan,tilt,range,x,y,z] → Open3D PointCloud (xyz only)"""
    xyz = pts[:, 3:6].astype(np.float64)
    # Remove NaN/Inf points
    valid = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[valid]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def estimate_params(pcd):
    """Auto-estimate processing params from point spacing."""
    pts = np.asarray(pcd.points)
    if len(pts) < 3:
        return {"normal_radius": 10.0, "normal_max_nn": 30, "icp_threshold": 50.0, "avg_spacing": 5.0}
    n = min(300, len(pts))
    idx = np.random.choice(len(pts), n, replace=False)
    tree = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    for i in idx:
        _, _, d = tree.search_knn_vector_3d(pts[i], 2)
        if len(d) >= 2 and d[1] > 0:
            dists.append(np.sqrt(d[1]))
    avg = np.mean(dists) if dists else 5.0
    return {
        "normal_radius": avg * 6,
        "normal_max_nn": 30,
        "icp_threshold": avg * 12,
        "avg_spacing": avg
    }


def prep_normals(pcd, params):
    """Estimate normals on a point cloud."""
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=params["normal_radius"], max_nn=params["normal_max_nn"]))
    pcd.orient_normals_consistent_tangent_plane(k=15)
    return pcd


# ============================================================
# SOR
# ============================================================

def handle_sor(msg):
    pts = np.array(msg["scan"], dtype=np.float64)
    k = msg.get("k", 8)
    std = msg.get("std", 1.0)

    pcd = pts_to_pcd(pts)
    _, idx = pcd.remove_statistical_outlier(nb_neighbors=k, std_ratio=std)

    removed = len(pts) - len(idx)
    print(f"  [SOR] {len(pts)} → {len(idx)} (removed {removed}, {100*removed/len(pts):.1f}%)")

    return {"cmd": "sor", "scan": pts[idx].tolist(), "removed": removed}


# ============================================================
# ICP — Align source scan onto target scan
# ============================================================

def handle_icp(msg):
    source = np.array(msg["source"], dtype=np.float64)
    target = np.array(msg["target"], dtype=np.float64)

    src_pcd = pts_to_pcd(source)
    tgt_pcd = pts_to_pcd(target)

    params = estimate_params(tgt_pcd)
    print(f"  [ICP] avg_spacing={params['avg_spacing']:.2f}mm, threshold={params['icp_threshold']:.2f}mm")

    prep_normals(src_pcd, params)
    prep_normals(tgt_pcd, params)

    result = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd,
        params["icp_threshold"],
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )

    T = result.transformation

    # Transform source xyz
    src_xyz = source[:, 3:6].copy()
    ones = np.ones((len(src_xyz), 1))
    homo = np.hstack([src_xyz, ones])  # Nx4
    aligned_homo = (T @ homo.T).T      # Nx4
    aligned_xyz = aligned_homo[:, :3]

    # Update source array
    aligned = source.copy()
    aligned[:, 3:6] = aligned_xyz

    print(f"  [ICP] RMSE={result.inlier_rmse:.4f}, fitness={result.fitness:.4f}")

    return {
        "cmd": "icp",
        "aligned": aligned.tolist(),
        "transform": T.tolist(),
        "rmse": float(result.inlier_rmse),
        "fitness": float(result.fitness)
    }


# ============================================================
# SLAM — Sequential ICP + Loop Closure
# ============================================================

def handle_slam(msg):
    scans_raw = [np.array(s, dtype=np.float64) for s in msg["scans"]]
    num = len(scans_raw)

    if num < 2:
        return {"cmd": "slam", "error": "Need at least 2 scans"}

    print(f"  [SLAM] {num} scans")

    # SOR each scan first
    clean_scans = []
    for i, pts in enumerate(scans_raw):
        pcd = pts_to_pcd(pts)
        _, idx = pcd.remove_statistical_outlier(nb_neighbors=8, std_ratio=1.0)
        clean_scans.append(pts[idx])
        print(f"    SOR scan {i}: {len(pts)} → {len(idx)}")

    # Sequential ICP
    poses = [np.eye(4)]
    pairwise_rmse = []

    for i in range(1, num):
        src_pcd = pts_to_pcd(clean_scans[i])
        tgt_pcd = pts_to_pcd(clean_scans[i - 1])

        params = estimate_params(tgt_pcd)
        prep_normals(src_pcd, params)
        prep_normals(tgt_pcd, params)

        result = o3d.pipelines.registration.registration_icp(
            src_pcd, tgt_pcd,
            params["icp_threshold"],
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )

        T_rel = result.transformation
        T_global = poses[i - 1] @ T_rel
        poses.append(T_global)
        pairwise_rmse.append(float(result.inlier_rmse))
        print(f"    ICP {i}→{i-1}: RMSE={result.inlier_rmse:.4f}, fitness={result.fitness:.4f}")

    # Loop closure check (last scan vs first)
    loop_closed = False
    lc_rmse = None
    if num >= 4:
        src_pcd = pts_to_pcd(clean_scans[-1])
        tgt_pcd = pts_to_pcd(clean_scans[0])
        params = estimate_params(tgt_pcd)
        prep_normals(src_pcd, params)
        prep_normals(tgt_pcd, params)

        # Use the global poses to give ICP an initial guess
        T_init = np.linalg.inv(poses[-1]) @ poses[0]

        lc_result = o3d.pipelines.registration.registration_icp(
            src_pcd, tgt_pcd,
            params["icp_threshold"],
            T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )

        lc_rmse = float(lc_result.inlier_rmse)
        print(f"    LC check: RMSE={lc_rmse:.4f}, fitness={lc_result.fitness:.4f}")

        if lc_result.fitness > 0.3:
            loop_closed = True
            # Simple loop closure correction: distribute translation error
            T_lc = lc_result.transformation
            # Error = gap between where last scan ends up vs where first scan is
            src_final = pts_to_pcd(clean_scans[-1])
            src_final.transform(poses[-1])
            tgt_origin = pts_to_pcd(clean_scans[0])

            # Compute the residual transform
            error_T = T_lc
            error_t = error_T[:3, 3]

            # Distribute error linearly across all poses
            for i in range(1, num):
                frac = float(i) / float(num - 1)
                correction = np.eye(4)
                correction[:3, 3] = -error_t * frac
                poses[i] = correction @ poses[i]

            print(f"    LC applied — distributed error across {num} poses")

    # Transform all scans into global frame
    aligned_scans = []
    for i in range(num):
        pts = clean_scans[i].copy()
        xyz = pts[:, 3:6]
        ones = np.ones((len(xyz), 1))
        homo = np.hstack([xyz, ones])
        transformed = (poses[i] @ homo.T).T[:, :3]
        pts[:, 3:6] = transformed
        aligned_scans.append(pts)

    # Build response
    poses_out = []
    for T in poses:
        poses_out.append({
            "R": T[:3, :3].tolist(),
            "t": T[:3, 3].tolist()
        })

    print(f"  [SLAM] Done — loop_closed={loop_closed}")

    return {
        "cmd": "slam",
        "scans": [s.tolist() for s in aligned_scans],
        "poses": poses_out,
        "pairwise_rmse": pairwise_rmse,
        "loop_closed": loop_closed,
        "lc_rmse": lc_rmse
    }


# ============================================================
# EDGE PIPELINE — Step-by-step processing with server-side state
# ============================================================

# Pipeline state (persists between calls for step-by-step flow)
pipe = {
    "scan": None,           # Nx6 array [pan,tilt,range,x,y,z]
    "pcd": None,            # Open3D PointCloud
    "normals": None,        # Nx3 normals
    "curvatures": None,     # N curvatures
    "planes": None,         # list of [a,b,c,d] plane equations
    "assignments": None,    # N-length array of plane index (-1=unassigned)
    "edges": None,          # list of edge dicts
}


def handle_normals(msg):
    """Step 1: Estimate normals + curvature for every point."""
    global pipe
    pts = np.array(msg["scan"], dtype=np.float64)
    k = msg.get("k", 15)

    # Filter out NaN/Inf points
    valid = np.all(np.isfinite(pts[:, 3:6]), axis=1)
    if not np.all(valid):
        print(f"  [NORMALS] Filtered {np.sum(~valid)} NaN/Inf points")
        pts = pts[valid]

    pcd = pts_to_pcd(pts)
    params = estimate_params(pcd)

    print(f"  [NORMALS] Processing {len(pts)} points with K={k}...")

    # Estimate normals (runs in C++)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=params["normal_radius"], max_nn=k))

    # Orient normals toward viewpoint (fast and safe)
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))

    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)

    # Curvature via batch KD-tree query (no Python loop)
    from scipy.spatial import cKDTree
    batch_k = min(k, 20)
    # Extra safety: ensure no NaN in points
    finite_mask = np.all(np.isfinite(points), axis=1)
    if not np.all(finite_mask):
        print(f"  [NORMALS] Warning: {np.sum(~finite_mask)} non-finite points in cloud, zeroing curvature")
        curvatures = np.zeros(len(points))
    else:
        tree = cKDTree(points)
        _, indices = tree.query(points, k=batch_k)
        # Gather all neighbor normals: NxKx3
        nbr_normals = normals[indices]
        # Mean normal per point: Nx3
        mean_normals = nbr_normals.mean(axis=1, keepdims=True)
        # Variance of normals = curvature
        diff = nbr_normals - mean_normals
        curvatures = np.mean(np.sum(diff**2, axis=2), axis=1)
    print(f"  [NORMALS] Curvature computed (vectorized)")

    # Store state for next steps
    pipe["scan"] = pts
    pipe["pcd"] = pcd
    pipe["normals"] = normals
    pipe["curvatures"] = curvatures

    print(f"  [NORMALS] Done, curvature range: {curvatures.min():.6f} to {curvatures.max():.6f}")

    return {"cmd": "normals", "curvatures": curvatures.tolist()}



def handle_ransac(msg):
    """Step 2: RANSAC plane segmentation — find flat faces."""
    global pipe
    if pipe["pcd"] is None:
        return {"cmd": "ransac", "error": "Run normals first"}

    min_pts = msg.get("min_points", 100)
    max_planes = msg.get("max_planes", 20)

    pcd = pipe["pcd"]
    points = np.asarray(pcd.points)
    params = estimate_params(pcd)
    dist_thresh = params["avg_spacing"] * 2

    assignments = np.full(len(points), -1, dtype=int)
    planes = []
    remaining = pcd
    remaining_idx = np.arange(len(points))

    for p in range(max_planes):
        if len(remaining.points) < min_pts:
            break

        plane_model, inliers = remaining.segment_plane(
            distance_threshold=dist_thresh, ransac_n=3, num_iterations=1000)

        if len(inliers) < min_pts:
            break

        # Normalize plane equation
        a, b, c, d = plane_model
        norm = np.sqrt(a*a + b*b + c*c)
        plane_eq = [a/norm, b/norm, c/norm, d/norm]

        global_inliers = remaining_idx[inliers]
        assignments[global_inliers] = p
        planes.append(plane_eq)

        n1 = np.array(plane_eq[:3])
        print(f"    Plane {p}: {len(inliers)} pts, normal=({n1[0]:.3f},{n1[1]:.3f},{n1[2]:.3f})")

        mask = np.ones(len(remaining.points), dtype=bool)
        mask[inliers] = False
        remaining_idx = remaining_idx[mask]
        remaining = remaining.select_by_index(inliers, invert=True)

    pipe["planes"] = planes
    pipe["assignments"] = assignments

    unassigned = int(np.sum(assignments < 0))
    print(f"  [RANSAC] {len(planes)} planes, {unassigned} unassigned points")

    return {
        "cmd": "ransac",
        "assignments": assignments.tolist(),
        "num_planes": len(planes),
        "plane_equations": planes
    }


def handle_edges(msg):
    """Step 3: Intersect adjacent planes to compute true edge lines."""
    global pipe
    if pipe["planes"] is None or len(pipe["planes"]) < 2:
        return {"cmd": "edges", "error": "Run RANSAC first (need 2+ planes)"}

    from itertools import combinations
    angle_thresh = msg.get("angle_thresh", 10.0)

    planes = pipe["planes"]
    points = np.asarray(pipe["pcd"].points)
    assignments = pipe["assignments"]
    
    # Bounding box for clipping
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_diag = np.linalg.norm(bbox_max - bbox_min)
    
    # Estimate point spacing for proximity threshold
    params = estimate_params(pipe["pcd"])
    proximity = params["avg_spacing"] * 8
    
    edges = []

    for (i, eq1), (j, eq2) in combinations(enumerate(planes), 2):
        n1 = np.array(eq1[:3])
        n2 = np.array(eq2[:3])

        cos_angle = abs(np.dot(n1, n2))
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, 0, 1)))
        if angle_deg < angle_thresh:
            continue

        direction = np.cross(n1, n2)
        dlen = np.linalg.norm(direction)
        if dlen < 1e-8:
            continue
        direction /= dlen

        abs_dir = np.abs(direction)
        free_axis = np.argmax(abs_dir)
        axes = [a for a in range(3) if a != free_axis]
        A = np.array([[n1[axes[0]], n1[axes[1]]],
                       [n2[axes[0]], n2[axes[1]]]])
        b_vec = np.array([-eq1[3], -eq2[3]])
        det = np.linalg.det(A)
        if abs(det) < 1e-10:
            continue
        sol = np.linalg.solve(A, b_vec)
        line_pt = np.zeros(3)
        line_pt[axes[0]] = sol[0]
        line_pt[axes[1]] = sol[1]

        # Only use points from BOTH planes that are actually CLOSE to the edge line
        mask_i = (assignments == i)
        mask_j = (assignments == j)
        pts_i = points[mask_i]
        pts_j = points[mask_j]
        if len(pts_i) < 5 or len(pts_j) < 5:
            continue

        # Adjacency check: are these planes actually neighbors?
        # Find minimum distance between any point in plane i and plane j
        from scipy.spatial import cKDTree as ckdt
        tree_j = ckdt(pts_j)
        min_dists, _ = tree_j.query(pts_i, k=1)
        if np.min(min_dists) > proximity * 3:
            continue  # planes are far apart, not adjacent

        mask = mask_i | mask_j
        candidate_pts = points[mask]

        # Compute distance from each candidate to the line
        vecs = candidate_pts - line_pt
        t_vals = vecs @ direction
        projections = line_pt + np.outer(t_vals, direction)
        dist_to_line = np.linalg.norm(candidate_pts - projections, axis=1)

        # Only keep points within proximity of the line
        close_mask = dist_to_line < proximity
        if np.sum(close_mask) < 5:
            continue

        close_t = t_vals[close_mask]
        
        # Tight percentile on only the close points
        t_min = np.percentile(close_t, 5)
        t_max = np.percentile(close_t, 95)
        
        # Skip if the edge is tiny
        edge_length = abs(t_max - t_min)
        if edge_length < bbox_diag * 0.05:
            continue

        start = line_pt + t_min * direction
        end = line_pt + t_max * direction
        
        # Clip to bounding box with small margin
        margin = bbox_diag * 0.02
        for axis in range(3):
            for pt in [start, end]:
                pt[axis] = np.clip(pt[axis], bbox_min[axis] - margin, bbox_max[axis] + margin)

        edges.append({
            "start": start.tolist(),
            "end": end.tolist(),
            "plane_i": i,
            "plane_j": j,
            "angle_deg": float(angle_deg),
            "direction": direction.tolist(),
            "line_point": line_pt.tolist()
        })

        print(f"    Edge: plane {i}↔{j}, dihedral={angle_deg:.1f}°, len={edge_length:.1f}mm, {np.sum(close_mask)} nearby pts")

    pipe["edges"] = edges
    print(f"  [EDGES] {len(edges)} edge lines computed")

    return {
        "cmd": "edges",
        "lines": [{"start": e["start"], "end": e["end"], "angle": e["angle_deg"]} for e in edges],
        "num_edges": len(edges)
    }


def handle_reproject(msg):
    """Step 4: Snap boundary points onto computed edge lines."""
    global pipe
    if pipe["edges"] is None or len(pipe["edges"]) == 0:
        return {"cmd": "reproject", "error": "Run edges first"}

    points = np.asarray(pipe["pcd"].points).copy()
    assignments = pipe["assignments"]
    params = estimate_params(pipe["pcd"])
    proximity = params["avg_spacing"] * 4
    boundary_band = params["avg_spacing"] * 6

    snapped_mask = np.zeros(len(points), dtype=bool)
    total_snapped = 0

    for edge in pipe["edges"]:
        line_pt = np.array(edge["line_point"])
        line_dir = np.array(edge["direction"])
        i, j = edge["plane_i"], edge["plane_j"]

        eq1 = pipe["planes"][i]
        eq2 = pipe["planes"][j]
        n1 = np.array(eq1[:3])
        n2 = np.array(eq2[:3])

        # Distance of every point to both planes
        d1 = np.abs(points @ n1 + eq1[3])
        d2 = np.abs(points @ n2 + eq2[3])

        # Points near the boundary: close to both planes
        near_boundary = (d1 < boundary_band) & (d2 < boundary_band)
        if not np.any(near_boundary):
            continue

        candidate_idx = np.where(near_boundary)[0]
        candidate_pts = points[candidate_idx]

        # Project onto edge line
        vecs = candidate_pts - line_pt
        t = vecs @ line_dir
        projections = line_pt + np.outer(t, line_dir)

        # Only snap points close enough to the line
        dists = np.linalg.norm(candidate_pts - projections, axis=1)
        snap_mask = dists < proximity

        snap_idx = candidate_idx[snap_mask]
        if len(snap_idx) > 0:
            points[snap_idx] = projections[snap_mask]
            snapped_mask[snap_idx] = True
            total_snapped += len(snap_idx)

    # Update scan with new positions
    updated_scan = pipe["scan"].copy()
    updated_scan[:, 3:6] = points

    print(f"  [REPROJECT] {total_snapped} points snapped to edge lines")

    return {
        "cmd": "reproject",
        "scan": updated_scan.tolist(),
        "snapped_mask": snapped_mask.tolist(),
        "reprojected": total_snapped
    }


# ============================================================
# WEBSOCKET HANDLER
# ============================================================

async def handler(ws):
    remote = ws.remote_address
    print(f"\n[+] Client connected from {remote}")
    try:
        async for raw in ws:
            msg = json.loads(raw)
            cmd = msg.get("cmd")
            t0 = time.time()

            print(f"\n[CMD] {cmd}")

            if cmd == "ping":
                await ws.send(json.dumps({"cmd": "pong"}))

            elif cmd == "sor":
                result = await asyncio.to_thread(handle_sor, msg)
                result["ms"] = int((time.time() - t0) * 1000)
                await ws.send(json.dumps(result))

            elif cmd == "icp":
                result = await asyncio.to_thread(handle_icp, msg)
                result["ms"] = int((time.time() - t0) * 1000)
                await ws.send(json.dumps(result))

            elif cmd == "slam":
                result = await asyncio.to_thread(handle_slam, msg)
                result["ms"] = int((time.time() - t0) * 1000)
                await ws.send(json.dumps(result))

            elif cmd == "normals":
                result = await asyncio.to_thread(handle_normals, msg)
                result["ms"] = int((time.time() - t0) * 1000)
                await ws.send(json.dumps(result))

            elif cmd == "ransac":
                result = await asyncio.to_thread(handle_ransac, msg)
                result["ms"] = int((time.time() - t0) * 1000)
                await ws.send(json.dumps(result))

            elif cmd == "edges":
                result = await asyncio.to_thread(handle_edges, msg)
                result["ms"] = int((time.time() - t0) * 1000)
                await ws.send(json.dumps(result))

            elif cmd == "reproject":
                result = await asyncio.to_thread(handle_reproject, msg)
                result["ms"] = int((time.time() - t0) * 1000)
                await ws.send(json.dumps(result))

            else:
                await ws.send(json.dumps({"cmd": "error", "msg": f"Unknown command: {cmd}"}))

    except websockets.ConnectionClosed:
        print(f"[-] Client disconnected")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


async def main():
    print("=" * 50)
    print("  Olimax Scanner Backend")
    print("  ws://localhost:8765")
    print("=" * 50)
    print()
    async with websockets.serve(handler, "localhost", 8765, max_size=50 * 1024 * 1024, ping_timeout=300, ping_interval=30):
        print("[READY] Waiting for connections...\n")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())