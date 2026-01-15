import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import cv2 as cv
    from pathlib import Path
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from dataclasses import dataclass


@app.cell
def _():
    mo.md(r"""
    ## Camera Calibration

    Estimate the intrinsic parameters using checkerboard pattern.

    TODO: principles how the method actually works
    - the checkerboard corners are equidistantly spaced grid of points in the world frame and we know all their coordinates because we choose frame (origin+axes orientation) to be the same as the checkerboard grid. Thus they have trivial coordinates (0,0,0), (0, 1, 0)... (one coordinate is always 0 because checkerboard is a plane; two principal axes of the world frame are aligned with it)
    - these corners need to be first detected in the calibration images in order to determine the respective pixel coordinates of the checkerboard points
    - from the pairs of points, ie. a 3D world points (corners) and their corresponding 2D pixel coordinates, one can determine the homography transformation relating these. The camera intrinsic matrix can be derived from this.
    """)
    return


@app.cell
def _():
    def calibrate_camera(img_dir: Path = Path("data/calibration")):
        """Compute camera intrinsics given a sample of checkerboard photos."""

        # Checkerboard parameters
        CHECKERBOARD = (8, 6)        # inner corners (width, height)
        SQUARE_SIZE = 0.025          # meters (example)

        # Prepare object points (0,0,0), (1,0,0), ...
        objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        objp *= SQUARE_SIZE

        objpoints = []  # 3D points
        imgpoints = []  # 2D points

        images = list(img_dir.glob("*.jpg"))

        for fname in images:
            img = cv.imread(str(fname))
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ret, corners = cv.findChessboardCorners(
                gray, CHECKERBOARD,
                flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
            )

            if ret:
                corners_refined = cv.cornerSubPix(
                    gray, corners,
                    winSize=(11, 11),
                    zeroZone=(-1, -1),
                    criteria=(
                        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                        30, 0.001
                    )
                )

                objpoints.append(objp)
                imgpoints.append(corners_refined)

        # Camera calibration
        ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        return K, dist

    K, dist = calibrate_camera()

    print("Camera matrix K:\n", K)
    print("Distortion coefficients:\n", dist)
    return K, dist


@app.cell
def _():
    mo.md(r"""
    ## Step 0: Keypoint Extraction (SIFT)

    Several possible feature detectors/descriptors: SIFT, ORB etc.
    """)
    return


@app.cell
def _():
    def extract_sift(img_path: Path):
        img = cv.imread(img_path)
        # img = cv.resize(img, (800, 600))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)

        return kp, des, img


    # for path in img_list:
    #     kp, des, img = extract_sift(path)

    # img = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(img)
    # cv.imwrite('sift_keypoints.jpg', img)
    return (extract_sift,)


@app.cell
def _():
    mo.md(r"""
    ## Step 1: 2-view SfM


    ### Matching Keypoint Descriptors
    They are **not** symmetrical by default. This is a very common point of confusion because, intuitively, if Point A looks most like Point B, we assume Point B must look most like Point A.

    In the world of high-dimensional feature matching (like SIFT's 128-D descriptors), this isn't always true.

    #### 1. The Nearest Neighbor "Love Triangle"

    Matching is a **Nearest Neighbor** search. The logic goes: *"For this specific point in Image 1, which point in Image 2 has the smallest mathematical distance?"*

    Consider three points:

    * **Point** $A_1$ (Image 1)
    * **Point** $B_1$ (Image 2)
    * **Point** $A_2$ (Image 1)

    It is entirely possible that:

    1. $A_1$’s closest neighbor in Image 2 is $B_1$.
    2. But $B_1$’s closest neighbor in Image 1 is $A_2$ (perhaps because $A_2$ is even more similar to $B_1$ than $A_1$ was).

    In this case, a "Forward Match" $(A \to B)$  gives you $(A_1, B_1)$ , but a "Backward Match" $(B\to A)$ gives you $(B_1, A_2)$. The match is **asymmetrical**.

    ---

    #### 2. How OpenCV Handles This

    OpenCV’s `BFMatcher` (Brute Force Matcher) has a specific flag for this called **`crossCheck`**.

    * **`crossCheck = False` (Your current code):** It only does a one-way search (Query  Train). If  likes , it’s a match, even if  prefers someone else. This is faster but noisier.
    * **`crossCheck = True`:** It performs the search in both directions  and . It only returns the match if they both "pick each other." This is a powerful way to remove outliers.

    #### 3. The Ratio Test vs. Symmetry

    You’ve been using **Lowe's Ratio Test** (`knnMatch` with `k=2`).

    There is a catch: **You cannot use `crossCheck=True` and `knnMatch` together** in standard OpenCV.

    * `crossCheck` requires finding the *single* best match.
    * Lowe's Ratio Test requires finding the *two* best matches to compare their distances.

    **Which is better?**
    For Structure from Motion (SfM), Lowe's Ratio Test is generally considered superior to simple cross-checking because it filters out **ambiguous** points (like the identical holes in your LEGO or the rings on your vase) rather than just "unrequited" ones.

    #### 4. Why Symmetry Matters for SfM

    If you have asymmetrical matches, you might end up with "Many-to-One" mappings (multiple points in Image 1 trying to triangulate to the same single point in Image 2).

    This creates "noise spikes" in your 3D point cloud. RANSAC usually cleans this up, but it makes the Essential Matrix calculation much harder.

    ---

    #### Summary Table

    | Feature | Symmetrical? | Purpose |
    | --- | --- | --- |
    | **Simple Match** | No | Basic "best guess" |
    | **Cross-Check Match** | **Yes** | Forces mutual agreement; very robust |
    | **kNN Match (Ratio Test)** | No | Discards ambiguous/repetitive textures |

    ### Query Set vs. Train Set
    In OpenCV terminology, the "Query" and "Train" sets are simply the labels for the "Searcher" and the "Database."
    In keypoint matching, the names describe the direction of the search:

    - **Query Set** (des0): These are the "active" features. For every single feature in this set, the algorithm asks: "Where is my best match in the other image?"

    - **Train Set** (des1): This is the "reference" library. The matcher "trains" (indexes) these descriptors so it can search through them efficiently (especially when using high-speed matchers like FLANN).

    Analogy: If you are looking for a specific face in a crowd, the photo of the face in your hand is the Query, and the crowd of people is the Train set.

    Each match is a `DMatch` object.
    When you use `knnMatch(k=2)`, OpenCV returns a list of lists. Each inner list contains two `DMatch` objects: `m` (the best) and `n` (the runner-up).

    **DMatch** attributes:
    - `queryIdx`: The row index of the keypoint in your first image (kp0).
    - `trainIdx`: The row index of the keypoint in your second image (kp1).
    - `distance`: The "cost" of the match. Lower is better (0.0 would be a perfect pixel-for-pixel match).
    - `imgIdx`: Used only if you are matching one image against a large folder of images. In your case, it's always 0.

    **Lowe's ratio test**:
    accepts only the best matches that significantly surpass their second-best competitor. If the best one is only marginally better, it is rejected alltogether.
    """)
    return


@app.cell
def _():
    img_dir = Path("data") / "raw" / "statue"
    img_list = sorted(list(img_dir.glob("*.jpg")))
    return img_dir, img_list


@app.cell
def _(extract_sift, img_list):
    kp0, des0, img0 = extract_sift(img_list[1])
    kp1, des1, img1 = extract_sift(img_list[2])

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)  # replace with FlannBasedMatcher?
    matches = bf.knnMatch(des0, des1, k=2)

    print(f"# Matches before filtering {len(matches)}")
    # Lowe's ratio test: the best match (m), second-best match (n)
    matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    print(f"# Matches after filtering {len(matches)}")
    return img0, img1, kp0, kp1, matches


@app.cell
def _(img0, img1, kp0, kp1, matches):
    img_matches = cv.drawMatches(
        img0, kp0,
        img1, kp1,
        matches, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    img_matches = cv.cvtColor(img_matches, cv.COLOR_BGR2RGB)

    fig0 = plt.figure(figsize=(12,5))
    ax0 = fig0.add_subplot()
    ax0.axis("off")
    ax0.imshow(img_matches)
    return


@app.cell
def _(K, kp0, kp1, matches):
    # extract corresponding pixel coordinates
    pts1 = np.float32([kp0[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp1[m.trainIdx].pt for m in matches])

    # compute Essential matrix using camera intrinsics
    # TODO: what is the mask for?
    E, mask = cv.findEssentialMat(
        pts1, pts2,
        K,
        method=cv.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    # camera pose
    retval, R, t, mask, points_4d = cv.recoverPose(
        E=E,
        points1=pts1,
        points2=pts2,
        cameraMatrix=K,
        distanceThresh=50.0, # mandatory for triangulation
    )

    # ret, R, t, mask_pose = cv.recoverPose(E, pts1, pts2, K)
    # # Projection matrices: from 3D world to each camera 2D image plane
    # P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    # P1 = K @ np.hstack((R, t))

    # # Inliers only
    # inliers = mask_pose.ravel() > 0
    # if np.all(inliers==False):
    #     print("No inliers detected! Can't triangulate 3D points.")
    # pts1_in = pts1[inliers]
    # pts2_in = pts2[inliers]
    # print(f"# inliers: {len(inliers)}")

    # # Triangulate
    # points_4d = cv.triangulatePoints(P0, P1, pts1_in.T, pts2_in.T)

    # Homogeneous → Euclidean
    points_3d = (points_4d[:3] / points_4d[3]).T
    return (points_3d,)


@app.cell
def _(points_3d):
    import plotly.express as px
    import pandas as pd

    # Convert to DataFrame
    df = pd.DataFrame(points_3d, columns=['x', 'y', 'z'])

    # --- OUTLIER REMOVAL ---
    # Calculate the distance from the median to find the "main cluster"
    median = df.median()
    distance = np.sqrt(((df - median)**2).sum(axis=1))

    # Keep only points within the 95th percentile of distance
    # This removes the "points at infinity" that squash your visualization
    mask_pts = distance < distance.quantile(0.95)
    df_filtered = df[mask_pts]
    return df_filtered, px


@app.cell
def _(df_filtered, px):
    # --- PLOTTING ---
    fig = px.scatter_3d(
        df_filtered, x='x', y='y', z='z',
        color='z',  # Map color to depth
        color_continuous_scale='Viridis',
        opacity=0.7,
        title="3D Statue Reconstruction",
    )
    # Customize markers
    fig.update_traces(
        marker=dict(
            size=1.5,           # Small size for dense clouds
            symbol='circle',    # Options: 'circle', 'square', 'diamond', 'cross'
            opacity=0.8,        # Slight transparency helps see depth
            line=dict(width=0)  # Remove outlines to make points look cleaner
        )
    )

    # Fix the aspect ratio so the statue isn't stretched
    fig.update_layout(scene_aspectmode='data')
    fig.show()
    return (fig,)


@app.cell
def _(fig):
    # 'eye' coordinates define where the camera is located
    # 'up' defines which direction is 'up' (usually the Z-axis)
    camera = dict(
        up=dict(x=0, y=-1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=2, y=-0.25, z=0.5) # Adjust these values to rotate the view
    )

    fig.update_layout(scene_camera=camera)
    return


@app.cell
def _(df_filtered, img0, pts1_in, px):
    # 1. Get colors from the first image at the inlier locations
    # pts1_in are (x, y) coordinates. We need to cast to int to index the image.
    colors = []
    for pt in pts1_in:
        x, y = int(pt[0]), int(pt[1])
        # OpenCV uses BGR, Plotly/Matplotlib use RGB
        b, g, r = img0[y, x] 
        colors.append(f'rgb({r},{g},{b})')

    # 2. Plot with real colors
    fig_color = px.scatter_3d(
        df_filtered, x='x', y='y', z='z',
    )

    fig_color.update_traces(
        marker=dict(
            size=1.5,
            color=colors  # Pass the list of RGB strings
        )
    )
    fig_color.show()
    return


@app.cell
def _():
    mo.md(r"""
    ## Step 2: n-view SfM

    - extract KPs and descriptors from each image
    - build view graph to know which images overlap each other; matches stored for each overlapping image pair
    - create initial 2-view 3D point estimates (essential mat estimation, pose recovery via decomp, triangulation, project)
        - create tracks for 3D points: "this 3D point (track #1) has KP_344 from image_0 and KP_12 from image_1"
    - examine view graph to determine which image pair to process next
        - some of the matching KPs between image_1 and the new image_2 will have already been used to triangulate points; ignore these; use only the KPs from image_2 (matched to image_1 KPs, that have not yet been used (ie. don't yet have a track)
    - `cv.solvePnP` to estimate the next image's camera pose + combine with `K` to get proj mats
    - `cv.triangulate` using the pair of proj mats and KPs

    **Bundle adjustment**: given N images of a scene, it solves a non-linear least squares optimization to refine camera poses and 3D point positions to minimize reprojection error.


    **Implementation of Track Management**
    I need the data structure to answer a question: _"Given a 2D point (kp_u, kp_v), is there a track for it?"_

    - `(image_id, kp_u, kp_v) --> track_id`
    - `track_id --> (x, y, z)`

    S.D. Prince: with n-images intrinsics can be estimated (auto-calibration)
    """)
    return


@app.cell
def _(extract_sift, img_dir, self):
    @dataclass
    class ImageData:
        idx: int
        path: Path
        img: np.ndarray
        kp: list[cv.KeyPoint]
        des: np.ndarray
        R: np.ndarray | None = None
        t: np.ndarray | None = None

        # @property
        # def stamped_keypoints():
        #     return [(idx, p.)]
    
    class FeatureStore:

        def __init__(self, img_dir: Path):
            self._store: list[ImageData] = []
            self._load_dir(img_dir)

        def _load_dir(self, img_dir: Path, ext: str = "jpg"):
            img_paths = sorted(list(img_dir.glob(f"*.{ext}")))
        
            if not img_paths:
                raise ValueError(f"No *.{ext} images found in {img_dir}")
            
            for idx, filepath in enumerate(img_paths):
                kp, des, img = extract_sift(filepath)
                self._store.append(ImageData(idx, filepath, img, kp, des))

        @property
        def size():
            return len(self._store)

        def __getitem__(self, img_idx: int):
            return self._store[img_idx]

        def set_pose(img_idx: int, R, t):
            self._store[img_idx].R = R
            self._store[img_idx].t = t
        
    
        def keypoints(self):
            yield from (item.kp for item in self._store)

        def descriptors(self):
            yield from (item.des for item in self._store)
        
    # load all images & extract features
    store = FeatureStore(img_dir)
    return ImageData, store


@app.cell
def _(K, store):
    def has_overlap(kp1, des1, kp2, des2, K, min_inliers=50):
        bf = cv.BFMatcher(cv.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)

        # Lowe ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < min_inliers:
            return False, None, None

        # geometric validation: rejects matches that cannot arise from a rigid 3D scene
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        E, mask = cv.findEssentialMat(
            pts1, pts2, K, method=cv.RANSAC, threshold=1.0
        )

        if E is None:
            return False, None, None

        inliers = int(mask.sum())
        if inliers < min_inliers:
            return False, None, None

        return True, inliers, good


    @dataclass
    class ViewEdge:
        i: int
        j: int
        inliers_ij: int  # matches i -> j
        inliers_ji: int  # matches j -> i

        @property
        def weight(self):
            # symmetric weight used for ranking
            return min(self.inliers_ij, self.inliers_ji)

    class ViewGraph:
        """
        Undirected weighted view graph with asymmetric match support.
        """

        def __init__(self):
            self.adj = defaultdict(dict)  # adj[i][j] = ViewEdge
            self.edges = []  # list of ViewEdge (global access)

        def add_edge(self, i, j, inliers_ij, inliers_ji):
            """
            Add or update an undirected edge between image i and j.
            """
            if i == j:
                return

            edge = ViewEdge(i, j, inliers_ij, inliers_ji)

            self.adj[i][j] = edge
            self.adj[j][i] = edge
            self.edges.append(edge)

        def neighbors(self, i) -> dict[int, ViewEdge]:
            """
            Return neighbors of image i with weights.
            """
            return self.adj[i]

        def best_edge(self) -> ViewEdge:
            """
            Return the edge with maximum symmetric weight.
            """
            return max(self.edges, key=lambda e: e.weight, default=None)


    def construct_view_graph(kp: list, des: list):
        view_graph = ViewGraph()
        assert len(kp) == len(des)
        N = len(kp)
        for i in range(N):
            for j in range(i + 1, N):
                ok_ij, inliers_ij, _ = has_overlap(kp[i], des[i], kp[j], des[j], K)
                ok_ji, inliers_ji, _ = has_overlap(kp[j], des[j], kp[i], des[i], K)
                # ASK: why the matches should not be preserved ???
            if ok_ij and ok_ji:
                view_graph.add_edge(i, j, inliers_ij, inliers_ji)

        return view_graph

    # keypoints, descriptors, images = [], [], []
    # for path in img_list:
    #     kp, des, img = extract_sift(path)
    #     keypoints.append(kp)
    #     descriptors.append(des)
    #     images.append(img)

    # TODO: change store to SOA layout? need materialized lists for view graph anyway
    kp_list, des_list = list(store.keypoints()), list(store.descriptors())
    view_graph = construct_view_graph(kp_list, des_list)

    return (view_graph,)


@app.cell
def _(ImageData, K, dist, images, mask_pose, store, track_manager, view_graph):
    def compute_baseline_estimate(
        img_0: ImageData, img_1: ImageData, track_manager
    ):
        """Computes two-view baseline estimate of 3D points and poses

        First image is at the origin.
        """

        # Match key points
        # TODO: extract into function
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(img_0.des, img_1.des, k=2)
        # Lowe's ratio filtering
        matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        # extract corresponding pixel coordinates
        pts0 = np.float32([img_0.kp[m.queryIdx].pt for m in matches])
        pts1 = np.float32([img_1.kp[m.trainIdx].pt for m in matches])

        # compute Essential matrix using camera intrinsics
        E, mask = cv.findEssentialMat(
            pts0, pts1, K, method=cv.RANSAC, prob=0.999, threshold=1.0
        )

        # estimate camera pose & triangulate 3D points
        # TODO: mask tells us which pairs of 2D points were successfully triangulated to 3D?
        retval, R, t, mask, points_4d = cv.recoverPose(
            E=E,
            points1=pts0,
            points2=pts1,
            cameraMatrix=K,
            distanceThresh=50.0,  # mandatory for triangulation
        )

        # Homogeneous --> Euclidean
        points_3d = (points_4d[:3] / points_4d[3]).T

        # TODO: TrackManager records tracked object points

        # Update image data structs w/ new estimates
        img_0.R, img_0.t = np.eye(3), np.zeros(3)
        img_1.R, img_1.t = R, t

        return R, t, points_3d


    def add_view(img_new: ImageData, img_ref: ImageData, track_manager):
        """Adds 3D points from new view using PnP and triangulation.

        img_ref is reference image for which we already have 2D-3D pt correspondence in track_manager
        """
        # Compute KP matches from ref image to new image
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
        # TODO: match in new -> ref or ref -> new
        # Matching from new to ref image: Where does tracked ref img KP match to in new img?
        matches = bf.knnMatch(img_ref.des, img_new.des, k=2)
        matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
        # pts_ref[i] matched to pts_new[i]
        pts_ref = np.float32([img_ref.kp[m.queryIdx].pt for m in matches])
        pts_new = np.float32([img_new.kp[m.trainIdx].pt for m in matches])

        # add new img KPs, that are matched to by tracked ref img KPs, to current tracks (3D pts)
        # TODO: what args do I need here? 
        tracked_kps, untracked_kps, tracks = track_manager.update_tracks(img_new, img_ref, matches)
        # Only use tracked KPs for reconstruction; tracked == "have 3D point"
        # TODO: as if TM should store the 3D pts; thiking PointCloud should be managed separately, TM should index into it; PointCloud.export(),plot_colored(),plot_depth()?
        object_points = track_manager.get_object_points(tracks)
    
        # Estimate pose of new image camera
        # PnP needs tracked KPs from new image (2D) and matching 3D pts
        pnp_ok, R, t, inliers = cv.solvePnPRansac(object_points, tracked_kps, K, dist)
        if not pnp_ok:
            raise ValueError("solvePnPRansac failed to estimate pose.")
        img_new.R, img_new.t = R, t
    
        # Inliers only
        inliers = mask_pose.ravel() > 0
        if np.all(inliers==False):
            print("No inliers detected! Can't triangulate 3D points.")
        pts_ref = pts_ref[inliers]
        pts_new = pts_new[inliers]
        print(f"# inliers: {len(inliers)}")
    
        # Projection matrices: from 3D world to each camera 2D image plane
        P_ref = K @ np.hstack((img_ref.R, img_ref.t))
        P_new = K @ np.hstack((img_new.R, img_new.t))

        # Triangulate
        points_4d = cv.triangulatePoints(P_ref, P_new, pts_ref.T, pts_new.T)

        # TODO: TrackManager add tracks for newly triangulated points

        return R, t, points_4d[:3] / points_4d[3]


    # Pick strongest baseline:
    # - The edge of the view graph with greatest weight (ie. # kp matches) determines the two images
    best_edge = view_graph.best_edge()
    img_0, img_1 = store[best_edge.i], store[best_edge.j]
    # TODO: managed PointCloud; .add_points()?
    # matches -> E -> pose -> triangulation
    _, _, points_3d = compute_baseline_estimate(img_0, img_1)

    R = set((best_edge.i, best_edge.j))
    U = set(range(len(images)))
    U.difference_update(R)

    while True:
        # find unregistered images connected to the registered ones
        # "connected" == "sharing matched keypoints"
        candidate_edges = [
            e
            for e in view_graph.edges
            if (e.i in R and e.j in U) or (e.j in R and e.i in U)
        ]

        if not candidate_edges:
            # TODO: U could still be non-empty (disconnected graph)
            break

        best_edge = max(candidate_edges, key=lambda e: e.weight)
        idx_ref, idx_new = (
            (best_edge.i, best_edge.j)
            if best_edge.i in R
            else (best_edge.j, best_edge.i)
        )
        img_ref, img_new = store[idx_ref], store[idx_new]

        # matches --> 2D-3D pairs --PnP--> pose -> triangulate untracked
        _, _, points_3d = add_view(img_new, img_ref, track_manager)

        # move currently processed image/node index from U to R
        if best_edge.i in R:
            R.add(best_edge.j)
            U.remove(best_edge.j)
        else:
            R.add(best_edge.i)
            U.remove(best_edge.i)


    # - given baseline estimate (two poses + 3d points)
    # - for each edge in view graph sorted by weight from best to worst
    #   - compute KP matches
    #   - if neither node has poses: estimate E -> R, t, triang. 3d pts
    #   - else if either node has a pose:
    #     - solvePnP to estimate R,t using KPs that have tracks (3d pts)
    #     - triangulate remaining new trackless KPs, create tracks for them

    # WHATIF: edge list ordering doesn't respect graph connectivity; ie. edge[0] connects nodes 1 and 2, edge[1] connects 3 and 4; WHATIF graph has several components
    # NEED: make sure the next edge being processed shares a node with the previously processed edge, ensures growing the

    # Compute initial reconstruction using the baseline images
    # - create tracks for 3d pts (ie. note involved kps)
    # Determine next image to process by examining the view graph
    return (points_3d,)


if __name__ == "__main__":
    app.run()
