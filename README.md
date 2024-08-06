# Interactive 3D Medical Image Segmentation with SAM 2

Interactive medical image segmentation (IMIS) has shown significant potential in enhancing segmentation accuracy by integrating iterative feedback from medical professionals. However, the limited availability of enough 3D medical data restricts the generalization and robustness of most IMIS methods. The Segment Anything Model (SAM), though effective for 2D images, requires expensive semi-auto slice-by-slice annotations for 3D medical images. 

In this paper, we explore the zero-shot capabilities of SAM 2, the next-generation Meta SAM model trained on videos, for 3D medical image segmentation. By treating sequential 2D slices of 3D images as video frames, SAM 2 can fully automatically propagate annotations from a single frame to the entire 3D volume. We propose a practical pipeline for using SAM 2 in 3D medical image segmentation and present key findings highlighting its efficiency and potential for further optimization. Numerical experiments on the BraTS2020 and the medical segmentation decathlon datasets demonstrate that SAM 2 still has a gap with supervised methods but can narrow the gap in specific settings and organ types, significantly reducing the annotation burden on medical professionals. 

Our paper is now available on [arXiv](https://arxiv.org/abs/2408.02635).

![SAM 2 architecture](assets/med_sam2.png?raw=true)

## Installation

Install SAM 2 on a GPU machine using:

```bash
git clone https://github.com/Chuyun-Shen/SAM_2_Medical_3D.git
cd SAM_2_Medical_3D
pip install -e .
```

To use the SAM 2 predictor and run the example notebooks, `jupyter` and `matplotlib` are required:

```bash
pip install -e ".[demo]"
```

## Getting Started

### Download Checkpoints

First, download a model checkpoint. You can download all the model checkpoints by running:

```bash
cd checkpoints
./download_ckpts.sh
```

Or individually from:

- [sam2_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt)
- [sam2_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt)
- [sam2_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt)
- [sam2_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt)

### 3D Image Prediction

For promptable segmentation and tracking in videos, we provide a video predictor with APIs, for example, to add prompts and propagate masklets throughout a video. SAM 2 supports video inference on multiple objects and uses an inference state to keep track of the interactions in each video.

```python
import torch
from sam2.build_sam import build_sam2_video_predictor

checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(<your_video>)

    # add new prompts and instantly get the output on the same frame
    frame_idx, object_ids, masks = predictor.add_new_points(state, <your_prompts>)

    # propagate the prompts to get masklets throughout the video
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        ...
```

Refer to [video_predictor_example.ipynb](./notebooks/video_predictor_example.ipynb) for details on how to add prompts, make refinements, and track multiple objects in videos.

### Example Notebooks

- For inference with mask: [video_predictor_example_with_mask.ipynb](./notebooks/video_predictor_example_with_mask.ipynb)
- For inference with interactive clicks: [video_predictor_example_with_clicks.ipynb](./notebooks/video_predictor_example_with_clicks.ipynb)

### Usage

First, convert nii.gz or .dicom files to jpgs as shown in ./notebooks/videos/brats001. A script for nifti to jpg conversion is provided in `./nii_2_video.py`.

Next, run `./segment_any_image.py` or `./segment_any_image_with_clicks.py`. For more details, refer to our 
example notebooks.

## Citing Us

```bibtex
@misc{shen2024interactive3dmedicalimage,
      title={Interactive 3D Medical Image Segmentation with SAM 2}, 
      author={Chuyun Shen and Wenhao Li and Yuhang Shi and Xiangfeng Wang},
      year={2024},
      eprint={2408.02635},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.02635}, 
}
```

---
