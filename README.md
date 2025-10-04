# Vision Transformer on CIFAR-10

I implemented a **Vision Transformer (ViT)** from scratch on CIFAR-10 using PyTorch. My model includes modern training strategies like label smoothing, AdamW, cosine learning rate schedule with warmup, and mixed-precision training.  

---

## Running in Google Colab

1. Open [Colab](https://colab.research.google.com/).
2. Change runtime type to T4.
3. Download the .ipynb and upload it on colab.  
4. Run each cell sequentially. 


The script automatically downloads CIFAR-10, splits it into **train, validation, and test sets**, trains for 200 epochs, and reports test accuracy.  

---

## Hyperparameters 

- **Image size**: 32×32  
- **Patch size**: 4  
- **Embedding dim**: 256  
- **Depth**: 6 Transformer blocks  
- **Heads**: 4  
- **MLP ratio**: 4  
- **DropPath**: 0.1  
- **Label smoothing**: 0.1  
- **Optimizer**: AdamW (`lr=5e-4`, `wd=0.05`)  
- **Scheduler**: Warmup (10 epochs) + CosineAnnealing (min lr=1e-5)  
- **Augmentations**: RandAugment, random crop + flip, CutMix/Mixup  
- **Batch size**: 256 (train), 2048 (val/test)  
- **Epochs**: 200  
- **Data split**: 90% train, 10% validation, plus CIFAR-10 test set  

The best checkpoint is saved as **`best_vit.pth`**.  

---

## Results

| Dataset | Accuracy (%) |
|---------|--------------|
| Val     |    86.40     | 
| Test    |    87.58     |  
| Train   |    97.31     |   

---

## Analysis

- **Train/Val/Test split**:  
  Instead of only using train and test, I created a validation set (10% of train) to tune schedules and checkpoint the best model.
- **Patch size trade-off**:  
  I chose small patches (4×4) since CIFAR-10 is low-res. Bigger patches (8×8) sped things up but hurt accuracy.  
- **Depth vs width**:  
  A depth of 6 layers with 4 heads was the best balance in my runs. Using say 12 layers gave small gains and each epoch took much longer, making it inefficient..
- **Augmentation effects**:  
  RandAugment, CutMix, and Mixup improved generalization. Without them, accuracy was really low in the initial epochs.  
- **Optimizer/scheduler choices**:  
  AdamW with cosine decay proved to be the most stable and efficient. I tried SGD, but it converged slower and needed more tuning of momentum and learning rates.I also added EMA updates, but they didn’t help since the shadow weights updated every step, lagging behind instead of stabilizing the model.
- **Overlapping patches (not used)**:  
  Overlaps could have helped squeeze out a bit more performance but they added compute and complexity.To enable them, the patch embedding stride would need to be smaller than the patch size (e.g., patch=4 but stride=2).I used non-overlapping patches for speed and simplicity.
- **Overall gain**:
  The best results came after data augmentation and scheduling tricks. Architectural tweaks like depth changes made smaller differences. This showed training strategy mattered more than brute force scaling in cases like this.
  
  
# Text-Driven Image & Video Segmentation Pipeline

I built a **text-to-segmentation pipeline** that uses **GroundingDINO** for object detection and **Segment Anything 2 (SAM2)** for segmentation.  
The pipeline works for both **images** and **videos**, where I provide query as a text prompt and the pipeline segments the exact objects.

---

## Pipeline

1. **Object Detection**  
   - I used GroundingDINO to detect bounding boxes given an image and a text prompt.  
   - The detections are filtered using a confidence threshold and only the top results are kept.

2. **Image Segmentation**  
   - With the detected boxes, I run SAM2’s image predictor to generate segmentation masks.  
   - I overlay these masks on the original image with colors and bounding boxes for clarity.

3. **Video Segmentation**  
   - I picked a reference frame and detect objects in it.  
   - Then I use SAM2’s video predictor to propagate these masks across all frames.  
   - Finally, saved the segmented video as an MP4 with overlays.

---

## Limitations

- **Multiple models required**: Since SAM2 is not open-vocabulary, I use Grounding DINO for text-driven detection and then pass the results to SAM2 for segmentation. The pipeline therefore always needs at least two models working together.  
- **Heavy dependencies**: The setup requires PyTorch, HuggingFace Transformers, GroundingDINO, and SAM2, which increases installation time and complexity.  
- **GPU dependency**: Performance is reasonable only on GPU. Running on CPU is possible but very slow for both image and video segmentation.  
- **Prompt sensitivity**: The quality of results depends strongly on the wording of the text prompt and the confidence threshold used.  
- **Performance on crowded scenes**: The system works best with a small number of objects. Segmentation becomes inconsistent in cluttered or crowded frames.  
- **Resolution constraints**: Frames are padded to multiples of 16 for codec compatibility, which can slightly distort the original aspect ratio.  
- **Inference only**: The current implementation only supports inference with pre-trained models. There is no training or fine-tuning loop included.
- **Not suitable for long videos**: The pipeline is not designed for very long videos (e.g., 10 minutes or more). Since every frame is processed and masks are propagated sequentially, both memory usage and runtime increases with video length.  

---
