import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import cv2 as cv
    from pathlib import Path
    import matplotlib.pyplot as plt


@app.cell
def _():
    mo.md(r"""
    ## Camera Calibration

    Estimate the intrinsic parameters using checkerboard pattern.

    TODO: principles how the method actually works
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

    return (K,)


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
    """)
    return


@app.cell
def _():
    img_dir = Path("data") / "raw" / "statue"
    img_list = sorted(list(img_dir.glob("*.jpg")))
    return (img_list,)


@app.cell
def _(extract_sift, img_list):
    kp0, des0, img0 = extract_sift(img_list[1])
    kp1, des1, img1 = extract_sift(img_list[2])

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)  # replace with FlannBasedMatcher?
    matches = bf.knnMatch(des0, des1, k=2)

    print(f"# Matches before filtering {len(matches)}")
    # TODO: understand the match filtering logic: Lowe's ratio test 
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
    # TODO: what is mask_pose?
    ret, R, t, mask_pose = cv.recoverPose(E, pts1, pts2, K)
    # ret, R, t, points_3d = cv.recoverPose(E, pts1, pts2, K, mask=mask)
    # print(f"recoverPose: {ret= }\n{R= }\n{t= }\n{mask_pose= }")

    # Projection matrices: from 3D world to each camera 2D image plane
    P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K @ np.hstack((R, t))

    # Inliers only
    inliers = mask_pose.ravel() > 0
    if np.all(inliers==False):
        print("No inliers detected! Can't triangulate 3D points.")
    pts1_in = pts1[inliers].T
    pts2_in = pts2[inliers].T
    print(f"# inliers: {len(inliers)}")

    # Triangulate
    points_4d = cv.triangulatePoints(P0, P1, pts1.T, pts2.T)

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
def _():
    # 1. Get colors from the first image at the inlier locations
    # pts1_in are (x, y) coordinates. We need to cast to int to index the image.
    # colors = []
    # for pt in pts1_in:
    #     x, y = int(pt[0]), int(pt[1])
    #     # OpenCV uses BGR, Plotly/Matplotlib use RGB
    #     b, g, r = img0[y, x] 
    #     colors.append(f'rgb({r},{g},{b})')

    # # 2. Plot with real colors
    # fig = px.scatter_3d(
    #     df_filtered, x='x', y='y', z='z',
    # )

    # fig.update_traces(
    #     marker=dict(
    #         size=1.5,
    #         color=colors  # Pass the list of RGB strings
    #     )
    # )
    # fig.show()
    return


@app.cell
def _():
    mo.md(r"""
    ## Step 2: n-view SfM

    - build view graph to know which images overlap each other
    - S.D. Prince: with n-images intrinsics can be estimated (auto-calibration)
    """)
    return


@app.cell
def _(K, des, images, kp):
    def has_overlap(kp1, des1, kp2, des2, K,
                    min_inliers=50):

        bf = cv.BFMatcher(cv.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)

        # Lowe ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < min_inliers:
            return False, None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        E, mask = cv.findEssentialMat(
            pts1, pts2, K,
            method=cv.RANSAC,
            threshold=1.0
        )

        if E is None:
            return False, None

        inliers = int(mask.sum())
        if inliers < min_inliers:
            return False, None

        return True, inliers

    def construct_view_graph():
        # View graph construction
        view_graph = {}
        N = len(images)
        for i in range(N):
            for j in range(i + 1, N):
                ok, inliers = has_overlap(
                    kp[i], des[i],
                    kp[j], des[j],
                    K
                )
                if ok:
                    view_graph.setdefault(i, []).append(j)
                    view_graph.setdefault(j, []).append(i)


    return


if __name__ == "__main__":
    app.run()
