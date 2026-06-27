## @ **TripletShift: Evaluating Vision-Language Model Vulnerability via Embedding Space Manipulation** 

**TripletShift (TS)** is a transfer-based adversarial attack framework for evaluating Vision-Language Models (VLMs) with an adaptive distance control mechanism that dynamically balances semantic alignment and visual imperceptibility. 

**Paper** : _TripletShift: Evaluating Vision-Language Model Vulnerability via Embedding Space Manipulation_ (PRICAI 2026) 

## **Table of Contents** 

- Framework Overview 

- Key Features 

- Installation 

- Supported Models 

- Quick Start 

- Usage 

- Methodology 

- Results 

- Citation 

## **Framework Overview** 

The figure below illustrates the complete TripletShift attack framework: 

Figure 1: TripletShift Framework 

## **Key Components:** 

- **Source Image** ($\mathcal{X}_c$): Clean image with original caption (e.g., “brown and white dog”) 

- **Target Caption** ($\mathcal{C}_t$): Desired target caption sampled from MS-COCO (e.g., “giraffe walking through trees”) 

- **Target Image** ($\mathcal{X}_t$): Reference image synthesized from target caption using Stable Diffusion 

- **Surrogate Encoder** ($\Phi$): Frozen CLIP image encoder (ViT-B/16) that maps images to embedding space 

- **Triplet Formation** : Anchor ($ \mathbf{a}$), Positive ($ \mathbf{p}_t$), Negative ($ \mathbf{p}_s$) 

- **Adaptive Distance Control** : $\omega(d_{\text{perc}}) = \alpha \cdot \exp(-\beta \cdot d_{\text{perc}})$ based on perceptual distance 



- **Victim VLM** : Black-box models (BLIP, BLIP-2, InstructBLIP, LLaVA, SmallCap, ViECap, etc.) 

## **Mathematical Formulation** 

|Component|Equation|
|---|---|
|Adversarial Image|𝒳𝑎= 𝒳𝑐+ 𝜂|
|Anchor Embedding|**a**= Φ(𝒳𝑎)|
|Target Prototype|**p**𝑡= Φ(𝒳𝑡)|
|Source Prototype|**p**𝑠= Φ(𝒳𝑐)|
|Adaptive Distance Control|𝜔(𝑑perc) = 𝛼exp(−𝛽⋅𝑑perc(𝒳𝑎, 𝒳𝑐))|
|TripletShift Loss|ℒTS =max(0, 𝜔−[cos(**a**,**p**𝑡) −cos(**a**,**p**𝑠)])|
|Total Loss|ℒtotal = ℒTS+ 𝜆⋅𝑑perc(𝒳𝑎, 𝒳𝑐) + 𝜇‖𝜂‖2<br>2|



## **Key Features** 

- **Adaptive Distance Control** : Automatically adjusts the alignment constraint based on perceptual distance without fixed hyperparameters 

- **Triplet Alignment** : Explicitly shifts adversarial embeddings toward target semantics while repelling from source semantics 

- **Black-Box Transfer** : No access to victim model internals required — uses frozen CLIP surrogate encoder 

- **Visual Imperceptibility** : LPIPS-based perceptual distance + L2 regularization ensures minimal visual distortion 

- **Multi-Model Evaluation** : Tested on BLIP, BLIP-2, InstructBLIP, LLaVA-7B, SmallCap, and ViECap 

## **Installation** 

# Clone repository git clone https://github.com/yourusername/TripletShift.git cd TripletShift 

# Create conda environment (recommended) conda create -n tripletshift python=3.10 conda activate tripletshift 

# Install dependencies pip install -r requirements.txt 

## **Requirements** 

torch>=1.13.0 torchvision>=0.14.0 transformers>=4.30.0 clip @ git+https://github.com/openai/CLIP.git 

  

lpips>=0.1.4 Pillow>=9.0.0 numpy>=1.21.0 nltk>=3.7 rouge-score>=0.1.2 pycocoevalcap sentencepiece accelerate tqdm 

## **Additional Setup** 

# Download NLTK data for evaluation metrics python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')" 

# Install pycocoevalcap (SPICE, CIDEr metrics) pip install git+https://github.com/tylin/coco-caption.git@master#subdirectory=pycocoevalcap 

## **Supported Models** 

|Model|Type|Access|||Script|
|---|---|---|---|---|---|
|BLIP|Victim|HuggingFace|||tripletshift_blip.py|
|BLIP-2|Victim|HuggingFace|||tripletshift_blip2.py|
|InstructBLIP|Victim|HuggingFace|/|ModelScope|tripletshift_instructblip.py|
|LLaVA-7B|Victim|HuggingFace|||tripletshift_llava.py|
|SmallCap|Victim|GitHub|||tripletshift_smallcap.py|
|ViECap|Victim|GitHub|||tripletshift_viecap.py|



## **Data Preparation** 

## **Directory Structure** 

D:\dmta_data\ ├──clean_images\ # Source images (ImageNet validation) │ ├──ILSVRC2012_val_00000001.JPEG │ ├──ILSVRC2012_val_00000002.JPEG │ └──... ├──clean_captions.txt # Original captions (one per line) ├──target_captions.txt # Target captions from MS-COCO (one per line) └──target_images\ # Synthesized target images from Stable Diffusion ├──ILSVRC2012_val_00000001.JPEG ├──ILSVRC2012_val_00000002.JPEG └──... 

  

## **Generate Target Images (Optional)** 

If you have target captions but not target images, synthesize them using Stable Diffusion: 

from diffusers import StableDiffusionPipeline import torch 

pipe = StableDiffusionPipeline.from_pretrained( "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 ).to("cuda") 

with open("target_captions.txt", "r") as f: captions = [line.strip() for line in f] 

for i, caption in enumerate(captions): image = pipe(caption, num_inference_steps=50).images[0] image.save(f"target_images/{i:08d}.jpg") 

## **Quick Start** 

## **1. BLIP (Simplest Setup)** 

python tripletshift_blip.py 

## **2. BLIP-2 (Requires ~15GB GPU)** 

python tripletshift_blip2.py 

## **3. LLaVA-7B (Requires ~14GB GPU)** 

python tripletshift_llava.py 

## **4. InstructBLIP (Vicuna-7B, China-compatible via ModelScope)** 

python tripletshift_instructblip.py 

## **5. SmallCap (Requires cloning smallcap repo)** 

# First: clone and setup SmallCap 

git clone https://github.com/RitaRamo/smallcap.git D:\SmallCap 

# Update checkpoint path in script, then run python tripletshift_smallcap.py 

## **6. ViECap (Requires cloning ViECap repo)** 

# First: clone and setup ViECap git clone https://github.com/FeiElysia/ViECap.git D:\ViECap 

  

# Update checkpoint path in script, then run python tripletshift_viecap.py 

## **Usage** 

## **Custom Hyperparameters** 

All scripts support configurable hyperparameters matching the paper: 

from tripletshift_blip import TripletShift 

attack = TripletShift( epsilon=8/255, # Perturbation budget (ε) — default: 8/255 gamma_step=1/255, # PGD step size (γ) — default: 1/255 K=100, # Optimization iterations — default: 100 alpha_dc=1.0, # Distance control magnitude (α) — default: 1.0 beta_dc=5.0, # Distance control decay (β) — default: 5.0 lambda_p=0.1, # Perceptual loss weight (λ) — default: 0.1 mu=0.001, # L2 regularization weight (μ) — default: 0.001 device="cuda" ) 

## **Recommended Settings (from Ablation Studies)** 

|Parameter|Value|Description|
|---|---|---|
|𝜀|8/255|Best balance of attack strength vs. visual|
|||quality|
|𝛾|1/255|Stable PGD convergence|
|𝐾|100|Sufficient for convergence|
|𝛼|1.0|Optimal distance control magnitude|
|𝛽|5.0|Optimal decay rate|
|𝜆|0.1|Perceptual weight for MiniGPT-4|
|𝜇|0.001|L2 regularization weight|



## **Generate Adversarial Image** 

import torch from PIL import Image 

# Load clean image 

img = Image.open("clean_images/example.jpg").convert("RGB") clean_image = attack.transform(img).unsqueeze(0).to("cuda") 

# Generate adversarial image 

target_caption = "a small dog wearing a blue bandana sits on the hood of a car" adv_image, metrics = attack.generate_adversarial_image( clean_image, target_caption, 

  

target_image=None # Optional: provide target_image for visual guidance ) # Generate caption with victim model caption = attack.generate_caption(adv_image) print(f"Generated: {caption}") 

## **Methodology** 

## **TripletShift Attack Pipeline** 

Clean Image (X_c) ──→Φ(・) ──→p_s (source prototype) │ Target Caption (C_t) ──→Stable Diffusion ──→X_t ──→Φ(・) ──→p_t (target prototype) │ │ └──────────────────────────────────────────────┘ │ PGD Optimization │ ┌─────────────────────┴─────────────────────┐ │ │ eta ←gradient of L_total L_TS + λ・d_perc + μ・||eta||² │ │ └─────────────────────┬─────────────────────┘ │ X_a = X_c + eta (adversarial image) │ Victim VLM ──→C* (targeted caption) 

## **Algorithm 1: TripletShift Optimization** 

Require: Clean image X_c, target caption C_t, perturbation bound ε, step size γ, iterations K Ensure: Adversarial image X_a 

1: Generate target reference image X_t from C_t using Stable Diffusion 2: Initialize perturbation η = 0 3: Compute source prototype p_s = Φ(X_c) 4: Compute target prototype p_t = Φ(X_t) 5: for k = 1 to K do 6: X_a ←X_c + η 7: Compute adversarial embedding a = Φ(X_a) 8: Compute perceptual distance d_perc(X_a, X_c) 9: Compute adaptive distance control ω ←α・exp(−β・d_perc) 10: Compute L_TS = max(0, ω −[cos(a,p_t) −cos(a,p_s)]) 11: Compute L_total = L_TS + λ・d_perc + μ・||η||² 12: Update perturbation η ←η −γ・∇_η L_total 13: Project perturbation η ←clip(η, −ε, ε) 14: end for 

  

15: X_a ←X_c + η 16: return X_a 

## **Results** 

## **Attack Performance on MS-COCO (Image Captioning)** 

|Victim Model|Method|SPICE↑|BLEU-1↑|BLEU-4↑|METEOR↑|
|---|---|---|---|---|---|
|**BLIP**|Self-|2.6|43.0|6.5|10.1|
||Universality|||||
||SASD|3.3|43.8|6.9|10.7|
||Any-Attack|3.4|44.4|7.1|11.1|
||MF-II|1.3|39.8|5.0|8.8|
||**TripletShift**|**4.2**|**47.9**|**8.5**|**13.5**|
||**(Ours)**|||||
|**BLIP-2**|Self-|1.6|40.9|5.6|9.2|
||Universality|||||
||SASD|2.6|43.0|6.3|10.2|
||Any-Attack|3.3|44.2|6.0|11.0|
||MF-II|1.2|39.6|5.3|8.7|
||**TripletShift**|**3.8**|**45.4**|**6.5**|**13.2**|
||**(Ours)**|||||
|**InstructBLIP**|Self-|1.9|40.7|6.0|9.3|
||Universality|||||
||SASD|3.4|43.9|7.2|10.5|
||Any-Attack|4.7|46.5|7.5|12.2|
||MF-II|1.4|38.9|5.4|8.7|
||**TripletShift**|**5.5**|**50.0**|**9.0**|**14.5**|
||**(Ours)**|||||
|**MiniGPT-4**|Self-|2.0|29.5|2.9|9.9|
||Universality|||||
||SASD|2.8|30.5|2.4|10.9|
||Any-Attack|4.6|32.5|4.0|12.4|
||MF-II|1.6|29.5|2.3|9.3|
||**TripletShift**|**5.3**|**35.5**|**5.0**|**14.8**|
||**(Ours)**|||||



## **CLIP Score Comparison (Embedding Space Alignment)** 

|Victim VLM|Method|ViT-B/16↑|ViT-L/14↑|
|---|---|---|---|
|**UniDiffuser**|Clean Image|42.9|30.5|
||ChainofAttack|77.2|69.8|
||**TripletShift (Ours)**|**78.5**|**71.2**|
|**LLaVA-7B**|Clean Image|48.1|33.7|



Table 5 – continued 

|Victim VLM|Method|ViT-B/16↑|ViT-L/14↑|
|---|---|---|---|
||ChainofAttack|52.0|35.8|
||**TripletShift (Ours)**|**53.2**|**36.9**|
|**SmallCap**|Clean Image|51.1|37.5|
||ChainofAttack|70.0|60.4|
||**TripletShift (Ours)**|**71.5**|**62.0**|
|**ViECap**|Clean Image|47.7|35.2|
||ChainofAttack|83.8|78.2|
||**TripletShift (Ours)**|**85.2**|**79.5**|



## **Ablation Studies** 

## **Perturbation Budget (￿)** 

|SPICE|SPICE|BLEU-4|CLIP|Score|LPIPS↓|
|---|---|---|---|---|---|
|2|2.9|2.4|58.4||0.098|
|4|4.1|3.7|68.7||0.069|
|**8**|**5.3**|**5.0**|**77.3**||**0.008**|
|12|5.2|4.9|77.1||0.019|
|16|5.1|4.8|76.8||0.041|



## **Distance Control Parameters (￿, ￿)** 

|SPICE|SPICE|SPICE|CLIP|Score|LPIPS↓|
|---|---|---|---|---|---|
|0.5|5|3.8|68.2||0.045|
|0.75|5|4.5|73.0||0.042|
|**1.0**|**5**|**5.3**|**77.3**||**0.030**|
|1.25|5|5.1|76.5||0.038|
|1.0|2|4.4|72.4||0.038|
|1.0|10|5.0|76.8||0.030|
|1.0|20|4.8|76.0||0.041|



## **Repository Structure** 

|TripletShift/|||
|---|---|---|
|├──tripletshift_blip.py|#|BLIP victim evaluation|
|├──tripletshift_blip2.py|#|BLIP-2 victim evaluation|
|├──tripletshift_instructblip.py|#|InstructBLIP victim evaluation|
|├──tripletshift_llava.py|#|LLaVA-7B victim evaluation|
|├──tripletshift_smallcap.py|#|SmallCap victim evaluation|
|├──tripletshift_viecap.py|#|ViECap victim evaluation|



├──requirements.txt # Python dependencies ├──README.md # This file ├──images/ # Framework figures │ └──triplet_attack_paper_1.pdf └──sample_data/ # Example data structure ├──clean_images/ ├──target_images/ ├──clean_captions.txt └──target_captions.txt 


## **License** 

This project is licensed under the MIT License - see the LICENSE file for details. 

## **Acknowledgments** 

- OpenAI CLIP for the surrogate encoder 

- HuggingFace Transformers for VLM implementations 

- LPIPS for perceptual distance 

- pycocoevalcap for caption evaluation metrics 

- Stable Diffusion for target image synthesis 

  

