CODA is the world's first real-world self-driving corner case dataset of 1500 scenes (frames) containing nearly 6K corner cases.
This is the full release of the 1500 scenes of CODA.
Corner case annotations are stored in `base-val-1500/corner_case.json` in COCO-compatible format.
Out of the 1500 scenes, 1057 are taken from ONCE, 134 are taken from nuScenes, and 309 are taken from KITTI.
Due to license issues, for nuScenes and KITTI, only corner case annotations and the correponding sample indices/tokens of the original datasets are provided (`base-val-1500/kitti_indices.json` and `base-val-1500/nuscenes_sample_tokens.json`).
For ONCE, in addition to corner case annotations, we also provide the front-view images captured by the camera named `cam03`.
The images taken from ONCE are named in the format of `[sequence_id]_[frame_id].jpg`, for example, `000001_1616005007200.jpg`.
The two identifiers (`sequence_id` and `frame_id`) can be used to extract other data (e.g., lidar point clouds) from the ONCE dataset if needed.