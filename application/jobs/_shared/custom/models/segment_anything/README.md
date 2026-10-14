# segment_anything (SAM-Med2D, trimmed)

Model code for the SAM-Med2D image encoder used by the `MST_SAMMed2D` backbone
(`models/mst.py`, `backbone_type="sammed2d"`). Contributed by Zhongying Deng (University of
Cambridge) on branch `custom_cam_ZD`; ported onto `main` in September 2026.

Origin: [OpenGVLab/SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D), itself derived from
[facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything).
Both are Apache-2.0; the copyright headers are kept as received.

What is here: `build_sam.py` (model registry and checkpoint loading, with the SAM-Med2D
adapter layers and 256-pixel position embeddings) and `modeling/` (ViT image encoder with
`Adapter_Layer`, prompt encoder, mask decoder, two-way transformer). Only the image encoder is
used for classification; the prompt encoder and mask decoder are built so that the published
checkpoint loads without key surgery, then discarded.

What was left out: `predictor.py`, `predictor_sammed.py`, `automatic_mask_generator.py` and
`utils/` (they import cv2, pycocotools and albumentations, none of which is in the ODELIA
image, and none is needed to run the encoder).

Weights: `sam-med2d_b.pth` from the SAM-Med2D release (Google Drive / Baidu links in their
README). It is not committed. Put it at `/MediSwarm/pretrained_weights/sam-med2d_b.pth` in the
image (the build's `_cacheAndCopyPretrainedModelWeights.sh` copies it from the build cache if
present) or point `SAM_MED2D_CHECKPOINT` at it. The release file has sha256
`921d8c130cc4bcc4b7220c064e60f207f15ad59f17ab36de340e41c67a490d18`.

`MST_SAMMed2D` refuses to start without it: `models/mst.py` raises `RuntimeError` (naming the
paths it searched and the fix) when no file is found, when the file cannot be read, or when its
`image_encoder.*` tensors do not match the encoder exactly (missing, unexpected or reshaped; the
loader in `build_sam.py` accepts partial loads, so `mst.py` loads the encoder itself). On success
it logs `SAM-Med2D backbone weights: <path> (<bytes>, sha256 <digest>); loaded 249/249 encoder
tensors`. `SAM_MED2D_ALLOW_RANDOM_INIT=1` is the only override: the encoder then starts from
random weights, with a loud warning. It exists for unit tests and CI images built without the
file; a model trained like that is not the pretrained SAM-Med2D model.
