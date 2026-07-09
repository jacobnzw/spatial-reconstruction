import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium", layout_file="layouts/slides.slides.json")


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    import pandas as pd
    import numpy as np

    return mo, np, pd, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bring Something Awesome

    TODO: needs work! 🙂
    ## How I scratched my itch by building a simple SfM pipeline?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    intro_md = mo.md(r"""
    ## Introduce the idea

    TODO: needs work! 🙂

    ### *Why did you bring this?*
    - I had an itch: I hear it's possible to triangulate but something in me couldn't believe it could be that easy
    - I want to understand the logic in detail - not to outclass SOTA SfM pipelines!
    - I found the epipolarity a bit mysterious at first glance
    - most familiar with sensor fusion using Kalman filters where the inputs are pre-processed features from the preceding stages in the pipeline (bounding boxes, ), but wanted to understand what's happening earlier in the pipeline

    Motivation: understand principles of 3D computer vision as a prerequisite for understanding SLAM, VIO

    3D CV indentified as the corner stone behind all of these

    ### What is SfM?

    """)

    sfm_demo = mo.video(src="https://lpanaf.github.io/assets/img/eccv24_glomap/facade.webm", controls=True, muted=True, autoplay=True, loop=True)
    sfm_demo_caption = mo.md(r"""Credit: [Pan, Linfei et al., Global Structure-from-Motion Revisited, ECCV, 2024](https://lpanaf.github.io/eccv24_glomap/)""")
    sfm_demo_pane = mo.vstack((sfm_demo, sfm_demo_caption))

    mo.hstack((intro_md, sfm_demo_pane), widths=[1, 1], justify="space-around")
    return


@app.cell(hide_code=True)
def _(mo):
    getting_data_text = mo.md(r"""
    ## Getting Data

    I want this to work on my data!

    How cool it would be to snap pictures of an object and turn those into a 3D model?

    Idea: turn some object from my shelf into a sparse 3D point cloud

    - First attempt: Lego Technic snowplow
      - trouble with feature matching
      - Lesson: don't use something with repetitive texture, leads to similar descriptors, hard to disambiguate keypoints across views, bad matches

    - Second attempt: Terracotta Archer Statue
      - Challenges
        - Inconsistent exposure
        - Head slightly out of focus

    """)
    statue_gif = mo.image(src="public/statue_orbit.gif", height=400)
    # statue_gif = mo.video(src="public/statue_orbit.gif")

    mo.hstack((getting_data_text, statue_gif), widths=[1, 1])
    return


@app.cell
def _(mo):
    proj_md = mo.md(r"""
    ## Understanding the Basics

    ### Projective Pinhole Camera Model

    *How does a 3D point in the world get mapped to the image plane?*
    $$
    \lambda
    \begin{bmatrix}
    x\\
    y\\
    1
    \end{bmatrix} = 
    \begin{bmatrix}
       \phi_x & \gamma & \delta_x & 0 \\
       0 & \phi_y & \delta_y & 0 \\
       0 & 0 & 1 & 0
    \end{bmatrix}
    \begin{bmatrix}
       r_{11} & r_{12} & r_{13} & t_x \\
       r_{21} & r_{22} & r_{23} & t_y \\
       r_{31} & r_{32} & r_{33} & t \\
       0 & 0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    u \\
    v \\
    w \\
    1
    \end{bmatrix}
    $$

    $$
    \lambda
    \tilde{\mathbf{x}} = 
    \begin{bmatrix}
       \mathbf{K} & \mathbf{0}
    \end{bmatrix}
    \begin{bmatrix}
       \mathbf{R} & \mathbf{t} \\
       \mathbf{0}^T & 1
    \end{bmatrix}
    \tilde{\mathbf{w}} = 
    \mathbf{K}
    \begin{bmatrix}
       \mathbf{R} & \mathbf{t}
    \end{bmatrix}
    \tilde{\mathbf{w}}
    $$

    - $\tilde{\mathbf{x}}$ is a projection of the world point $\tilde{\mathbf{w}}$ onto the image place
    - $\mathbf{K}$ is the camera intrinsic matrix
      - *How do 3D world points map to 2D image pixels?*
    - Rotation matrix $\mathbf{R}$ and translation vector $\mathbf{t}$ are the camera extrinsics
      - *Where is the camera in the world frame?*

    """)
    proj_md
    return


@app.cell(hide_code=True)
def _(mo):
    epicon_md = mo.md(r"""
    ## Understanding the Basics

    ### Essential matrix $\mathbf{E}$

    - *Epipolar constraint*: $\mathbf{x}^{\top}\mathbf{E}\mathbf{x} = 0$
    - Decomposition $\mathbf{E}=[\mathbf{t}]^{\times} \mathbf{R}$
      - Recovers relative camera pose $[\mathbf{R}, \mathbf{t}]$
    - Estimation: 5-point algorithm
    - RANSAC
    - OpenCV: `findEssentialMat(pts0, pts1, K, method=cv.RANSAC, ...)`

    """)

    epicon_img = mo.image(src="https://datahacker.rs/wp-content/uploads/2019/07/45.png", 
                          alt="Epipolar constraint",
                          caption="Epipolar constraint illustrated. [Credit: datahacker.rs]",
                          width=400)

    mo.hstack((epicon_md, epicon_img), widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    basics_kp_matching = mo.md(r"""
    ## Understanding the Basics

    ### Keypoint Detection and Matching

    - SIFT
    - Brute-Force kNN (`k=2`) matcher
    - Lowe's ratio match filter

    Important: Don't get lost in the weeds!

    """)
    statue_matches = mo.image(src="public/statue_matches.png", width=600, caption="Keypoint matches visualized: Criss-crossing lines (bottom checkerboard) indicate bad matches.")
    # mo.hstack((basics_kp_matching, ))
    mo.hstack((basics_kp_matching, statue_matches), widths=(1, 1))
    return


@app.cell(hide_code=True)
def _(mo, np, pd, px):
    left = mo.md(r"""
    ## Basics in Practice

    Key-point detection & matching

    OpenCV primitives for 3D reconstruction

    - `findEssentialMat()`
    - `recoverPose()`
    - `triangulatePoints()`

    First reconstruction! :tada:

    I was in business!

    *"OK, this is really cool!"*

    *"How can I incorporate more views, though?"* :thinking:

    """)
    # right = mo.image(src="public/first-good-statue-two-view-sparse-3d-point-cloud.png")

    # Load my first statue reconstruction
    df = pd.read_csv('notes/public/statue_two-view.csv')
    # Calculate the distance from the median to find the "main cluster"
    distance = np.sqrt(((df - df.median())**2).sum(axis=1))
    # Keep only points within the 95th percentile of distance
    df_filtered = df[distance < distance.quantile(0.95)]


    fig = px.scatter_3d(
        df_filtered, 
        x='x', y='y', z='z',
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
    # fig.show()


    mo.hstack((left, fig), widths=[1, 2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Enter View Graph

    After few LLM prompts, I understood I need a view graph to keep track of which image pairs should be used for triangulation of new points.


    I had another problem...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Enter Track Manager

    How to keep track of which views observe which 3D points?

    Important when adding a new view!

    Two kinds of keypoints in the new image that we wanna add
    - Tracked
      - that match to reference image keypoints
    - Untracked
      -
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    neural_features_md = mo.md(r"""
    ## Neural Features and Matcher

    I had to try some SOTA neural features and matchers.

    I used Kornia library.
    """)
    disk_md = mo.md(r"""
    ### DISK Features

    - finds keypoints
    - descriptor extraction

    """)
    pdf_opts = {"width": "75%", "height": "50vh"}
    disk_paper = mo.pdf(src="https://arxiv.org/pdf/2006.13566", initial_page=2, **pdf_opts)
    disk_pane = mo.vstack((disk_md, disk_paper))
    lightglue_md = mo.md(r"""
    ### LightGlue Matcher

    - bi-directional matches
    - confidence score
    """)
    lightglue_paper = mo.pdf(src="https://arxiv.org/pdf/2306.13643", initial_page=10, **pdf_opts)
    lightglue_pane = mo.vstack((lightglue_md, lightglue_paper))
    # paper_hstack = mo.hstack((disk_paper, lightglue_paper), widths=[1, 1])
    paper_panes = mo.hstack((disk_pane, lightglue_pane), widths=[1, 1])
    mo.vstack((neural_features_md, paper_panes))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Full Reconstruction Results

    TODO: Use fragments (sub-slides) to show recons (SIFT+BF, DISK+LightGlue w/ 1000 and 2000) different models
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hard Lessons

    - highly sensitive to intrinsics
      - TODO: compare recons w/ & w/o perturbed K
    - logging or experiment tracking
      - tip: set up as early as first plausible results show up,
      - log inputs, params, outputs
      - considering experiment tracking via W&B (as in my RL project)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Detours & Dead-Ends

    - Gaussian splatting volumetric rendering
    - Factor graphs via GTSAM
      - Cool framework but I went in too fast...
      - Tried coding basic SLAM on TUM-VI `corridor4` sequence re-using the some SfM primitives
      - Stuck on finding the right keyframe selection logic: finicky!
      - Future: Revert to easier task: SfM via Factor Graphs
      - Future: Try on easier datsets: KITTI, Colmap test data
        - TUM-VI challenging even for SOTA VIO frameworks (OKVIS, etc.)
    """)
    return


if __name__ == "__main__":
    app.run()
