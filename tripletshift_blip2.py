import torch
import sys
sys.path.insert(0, r"D:\CLIP")
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
from torchvision import transforms
import lpips
import clip
from transformers import (
    Blip2Processor,           # For BLIP-2
    Blip2ForConditionalGeneration,  # For BLIP-2
    AutoProcessor,
    AutoModelForVision2Seq
)
from tqdm import tqdm
import json
import os
import warnings
warnings.filterwarnings('ignore')

# NLTK imports for evaluation
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


class SemanticEncoder(nn.Module):
    """
    Semantic encoder for TripletShift attack.
    Uses CLIP ViT-B/16 as the surrogate encoder (frozen).
    Maps images to a shared embedding space for triplet alignment.
    """

    def __init__(self, device="cuda", model_name="ViT-B/16"):
        super().__init__()
        self.device = device
        self.model_name = model_name

        # Load CLIP as surrogate encoder (frozen)
        self.clip_model, _ = clip.load(model_name, device=device)
        self.clip_model.eval()

        # Freeze all parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.embedding_dim = 512  # ViT-B/16 output dimension

        print(f"SemanticEncoder initialized with {model_name} on {device}")

    def encode_image(self, images):
        """Encode images to embedding space"""
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
            # L2 normalize (as per Eq. 8 in paper)
            features = F.normalize(features, dim=-1)
        return features

    def encode_text(self, text):
        """Encode text to embedding space"""
        with torch.no_grad():
            text_tokens = clip.tokenize(text, truncate=True).to(self.device)
            features = self.clip_model.encode_text(text_tokens)
            features = F.normalize(features, dim=-1)
        return features

    def compute_clip_score(self, image_tensor, text):
        """Compute CLIP similarity score between image and text using the same encoder"""
        with torch.no_grad():
            img_features = self.encode_image(image_tensor)
            text_features = self.encode_text([text])
            similarity = (img_features * text_features).sum().item() * 100  # Scale to 0-100
        return similarity

    def forward(self, images):
        """Forward pass for images"""
        return self.encode_image(images)


class TripletShift:
    """
    TripletShift (TS) Attack with BLIP-2 Victim Model

    Implements the attack described in Section 3.2 of the paper:
    - Triplet formulation with adaptive distance control
    - Total loss: L_total = L_TS + lambda * d_perc + mu * ||eta||_2^2
    - L_infinity constraint on perturbation
    """

    def __init__(self,
                 epsilon=8/255,           # L_inf perturbation budget (epsilon in paper)
                 gamma_step=1/255,        # Step size for PGD (gamma in Algorithm 1)
                 K=100,                   # Number of PGD iterations
                 alpha_dc=1.0,            # Distance control magnitude (alpha in Eq. 4)
                 beta_dc=5.0,             # Distance control decay rate (beta in Eq. 4)
                 lambda_p=0.1,            # Perceptual loss weight (lambda in Eq. 6)
                 mu=0.001,                # L2 regularization weight (mu in Eq. 6)
                 blip2_model_name="Salesforce/blip2-opt-2.7b",  # BLIP-2 model variant
                 device="cuda"):
        """
        Initialize TripletShift attack.

        Args:
            epsilon: Maximum L_inf perturbation (default: 8/255)
            gamma_step: Step size for gradient updates (gamma in Algorithm 1)
            K: Number of PGD iterations
            alpha_dc: Distance control magnitude (Eq. 4)
            beta_dc: Distance control decay rate (Eq. 4)
            lambda_p: Weight for perceptual loss (Eq. 6)
            mu: Weight for L2 regularization (Eq. 6)
            blip2_model_name: BLIP-2 model variant to use
            device: Device to run on
        """
        self.epsilon = epsilon
        self.gamma_step = gamma_step
        self.K = K
        self.alpha_dc = alpha_dc
        self.beta_dc = beta_dc
        self.lambda_p = lambda_p
        self.mu = mu
        self.blip2_model_name = blip2_model_name
        self.device = device

        # Initialize LPIPS for perceptual distance (Section 3.2)
        self.lpips = lpips.LPIPS(net='alex').to(device).eval()

        # Initialize surrogate semantic encoder (CLIP)
        self.encoder = SemanticEncoder(device=device)

        # Initialize victim BLIP-2 model
        self._load_victim_model()

        # Image transforms (CLIP normalization)
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        # Unnormalize for visualization
        self.unnormalize = transforms.Normalize(
            mean=[-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711],
            std=[1/0.26862954, 1/0.26130258, 1/0.27577711]
        )

        print(f"\n{'='*60}")
        print(f"TripletShift Attack with BLIP-2 Victim Model")
        print(f"{'='*60}")
        print(f"BLIP-2 Model: {blip2_model_name}")
        print(f"Perturbation budget epsilon: {epsilon*255:.1f}/255")
        print(f"Step size gamma: {gamma_step*255:.1f}/255")
        print(f"Iterations K: {K}")
        print(f"Distance control: alpha={alpha_dc}, beta={beta_dc}")
        print(f"Loss weights: lambda={lambda_p}, mu={mu}")
        print(f"{'='*60}\n")

    def _load_victim_model(self):
        """Load BLIP-2 victim model from HuggingFace with compatibility fixes"""
        print(f"Loading BLIP-2 victim model: {self.blip2_model_name}...")

        # Method 1: Try direct loading with specific configuration
        try:
            # Use AutoProcessor and AutoModelForVision2Seq for better compatibility
            self.victim_processor = AutoProcessor.from_pretrained(self.blip2_model_name)
            self.victim_model = AutoModelForVision2Seq.from_pretrained(
                self.blip2_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            print("✓ BLIP-2 model loaded successfully via AutoModel!")
            self.victim_model = self.victim_model.to(self.device)
            self.victim_model.eval()
            return

        except Exception as e:
            print(f"AutoModel loading failed: {e}")

        # Method 2: Try with trust_remote_code
        try:
            self.victim_processor = Blip2Processor.from_pretrained(
                self.blip2_model_name,
                trust_remote_code=True
            )
            self.victim_model = Blip2ForConditionalGeneration.from_pretrained(
                self.blip2_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print("✓ BLIP-2 model loaded successfully with trust_remote_code!")
            self.victim_model = self.victim_model.to(self.device)
            self.victim_model.eval()
            return

        except Exception as e:
            print(f"Loading with trust_remote_code failed: {e}")

        # Method 3: Try different BLIP-2 variant (OPT 2.7b is having issues, try the Flan-T5 variant)
        alternative_models = [
            "Salesforce/blip2-flan-t5-xl",
            "Salesforce/blip2-flan-t5-xxl", 
            "Salesforce/blip2-opt-2.7b-coco"
        ]

        for alt_model in alternative_models:
            try:
                print(f"Trying alternative model: {alt_model}")
                self.victim_processor = AutoProcessor.from_pretrained(alt_model)
                self.victim_model = AutoModelForVision2Seq.from_pretrained(
                    alt_model,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True
                )
                self.blip2_model_name = alt_model
                print(f"✓ BLIP-2 model loaded successfully with {alt_model}!")
                self.victim_model = self.victim_model.to(self.device)
                self.victim_model.eval()
                return
            except Exception as e:
                print(f"Failed to load {alt_model}: {e}")
                continue

        # Method 4: Final fallback - use local cache or re-download with specific revision
        try:
            # Use a specific revision that is known to work
            self.victim_processor = Blip2Processor.from_pretrained(
                "Salesforce/blip2-flan-t5-xl",
                revision="main"
            )
            self.victim_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-flan-t5-xl",
                revision="main",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.blip2_model_name = "Salesforce/blip2-flan-t5-xl"
            print("✓ BLIP-2 model loaded with Flan-T5 XL variant!")
            self.victim_model = self.victim_model.to(self.device)
            self.victim_model.eval()
            return

        except Exception as e:
            print(f"All loading attempts failed: {e}")
            raise RuntimeError("Could not load BLIP-2 model. Please check your transformers version or network connection.")

    def compute_perceptual_distance(self, img1, img2):
        """
        Compute perceptual distance using LPIPS.
        d_perc(X_a, X_c) from Eq. 4
        Returns scalar value
        """
        # LPIPS expects inputs in [-1, 1] range
        img1_norm = img1 * 2 - 1
        img2_norm = img2 * 2 - 1
        with torch.no_grad():
            distance = self.lpips(img1_norm, img2_norm)
        return distance.item()

    def adaptive_distance_control(self, perceptual_distance):
        """
        Compute adaptive distance control based on perceptual distance.
        Eq. 4: omega(d_perc) = alpha * exp(-beta * d_perc)
        """
        return self.alpha_dc * torch.exp(-self.beta_dc * perceptual_distance)

    def ts_loss(self, anchor, p_t, p_s, omega):
        """
        Compute TripletShift loss.
        Eq. 5: L_TS = max(0, omega - [cos(a, p_t) - cos(a, p_s)])

        Args:
            anchor: Embedding of adversarial image (a)
            p_t: Embedding of target prototype (positive)
            p_s: Embedding of source prototype (negative)
            omega: Adaptive distance control value
        """
        # Cosine similarities (embeddings are already L2 normalized)
        sim_target = (anchor * p_t).sum(dim=-1)
        sim_source = (anchor * p_s).sum(dim=-1)
        sim_gap = sim_target - sim_source

        # Triplet loss with adaptive distance control
        loss = torch.clamp(omega - sim_gap, min=0.0)

        return loss.mean(), sim_target.mean(), sim_source.mean()

    def total_loss(self, adv_image, clean_image, p_t, p_s):
        """
        Compute total loss.
        Eq. 6: L_total = L_TS + lambda * d_perc + mu * ||eta||_2^2
        """
        # Get adversarial embedding (anchor)
        anchor = self.encoder.encode_image(adv_image)

        # Compute perceptual distance
        d_perc = self.compute_perceptual_distance(adv_image, clean_image)
        d_perc_tensor = torch.tensor(d_perc, device=self.device)

        # Compute adaptive distance control
        omega = self.adaptive_distance_control(d_perc)

        # TS loss
        loss_triplet, sim_target, sim_source = self.ts_loss(
            anchor, p_t, p_s, omega
        )

        # Perceptual loss (lambda * d_perc)
        loss_perceptual = self.lambda_p * d_perc_tensor

        # L2 regularization (mu * ||eta||_2^2)
        eta = adv_image - clean_image
        loss_l2 = self.mu * torch.norm(eta, p=2) ** 2

        # Total loss
        total_loss = loss_triplet + loss_perceptual + loss_l2

        return total_loss, loss_triplet, loss_perceptual, loss_l2, sim_target, sim_source, d_perc

    def generate_adversarial_image(self, clean_image, target_caption, target_image=None):
        """
        Generate adversarial image using PGD optimization.

        Args:
            clean_image: Clean source image tensor [1, 3, 224, 224]
            target_caption: Target caption string
            target_image: Optional target image tensor for visual guidance

        Returns:
            adversarial_image: Generated adversarial image
            metrics: Dictionary containing intermediate metrics
        """
        # Initialize perturbation
        eta = torch.zeros_like(clean_image, device=self.device, requires_grad=True)

        # Get source prototype (Eq. 3)
        with torch.no_grad():
            p_s = self.encoder.encode_image(clean_image)

        # Get target prototype (Eq. 2)
        with torch.no_grad():
            if target_image is not None:
                p_t = self.encoder.encode_image(target_image)
            else:
                p_t = self.encoder.encode_text([target_caption])

        # Track best result
        best_sim_target = -1.0
        best_eta = None
        best_lpips = float('inf')

        print(f"Initial similarity to target: {self._compute_similarity(clean_image, p_t):.4f}")

        # PGD optimization loop (Algorithm 1)
        for iteration in range(self.K):
            # Zero gradients
            if eta.grad is not None:
                eta.grad.zero_()

            # Create adversarial image (Eq. 1)
            adv_image = torch.clamp(clean_image + eta, 0, 1)

            # Compute total loss
            total_loss, loss_triplet, loss_percep, loss_l2, sim_target, sim_source, d_perc = \
                self.total_loss(adv_image, clean_image, p_t, p_s)

            # Backward pass
            total_loss.backward()

            # Update perturbation with sign gradient (Algorithm 1, line 12)
            if eta.grad is not None:
                eta.data = eta.data - self.gamma_step * eta.grad.sign()

            # Project onto L_inf ball (Algorithm 1, line 13)
            eta.data = torch.clamp(eta.data, -self.epsilon, self.epsilon)
            eta.data = torch.clamp(clean_image + eta.data, 0, 1) - clean_image

            # Track best similarity
            current_sim = sim_target.item() if torch.is_tensor(sim_target) else sim_target
            if current_sim > best_sim_target:
                best_sim_target = current_sim
                best_eta = eta.data.clone()
                best_lpips = d_perc

            # Logging
            if (iteration + 1) % 20 == 0 or iteration == 0:
                print(f"Iter {iteration+1:3d}/{self.K}: "
                      f"L_total={total_loss.item():.4f}, "
                      f"L_TS={loss_triplet.item():.4f}, "
                      f"Sim_tgt={current_sim:.4f}, "
                      f"Sim_src={sim_source.item():.4f}, "
                      f"LPIPS={d_perc:.4f}")

            # Early stopping if target similarity is high enough
            if best_sim_target > 0.85:
                print(f"✓ Early stopping at iteration {iteration+1} (sim_target={best_sim_target:.4f})")
                break

        # Use best eta found
        if best_eta is not None:
            adversarial_image = torch.clamp(clean_image + best_eta, 0, 1)
        else:
            adversarial_image = torch.clamp(clean_image + eta, 0, 1)

        print(f"\nFinal similarity to target: {best_sim_target:.4f}")
        print(f"Final LPIPS distance: {best_lpips:.4f}")

        return adversarial_image, {'sim_target': best_sim_target, 'lpips': best_lpips}

    def _compute_similarity(self, image, target_embedding):
        """Helper to compute cosine similarity between image and target"""
        with torch.no_grad():
            img_emb = self.encoder.encode_image(image)
            similarity = (img_emb * target_embedding).sum().item()
        return similarity

    def generate_caption(self, image_tensor):
        """
        Generate caption using BLIP-2 victim model.
        """
        # Denormalize image
        image_denorm = self.unnormalize(image_tensor)
        image_denorm = torch.clamp(image_denorm, 0, 1)

        # Convert to PIL
        image_pil = transforms.ToPILImage()(image_denorm.squeeze().cpu())

        # Prepare inputs
        inputs = self.victim_processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Handle dtype conversion for fp16
        if hasattr(self.victim_model, 'dtype') and self.victim_model.dtype == torch.float16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.victim_model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=5,
                do_sample=False,
                temperature=1.0
            )

        caption = self.victim_processor.decode(outputs[0], skip_special_tokens=True)
        return caption


class EvaluationMetrics:
    """Evaluation metrics for image captioning"""

    def __init__(self):
        self.cider_scorer = Cider()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothie = SmoothingFunction().method4

    def compute_bleu(self, reference, candidate):
        """Compute BLEU-1 and BLEU-4"""
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()

        bleu1 = sentence_bleu([ref_tokens], cand_tokens, 
                            weights=(1, 0, 0, 0), 
                            smoothing_function=self.smoothie)
        bleu4 = sentence_bleu([ref_tokens], cand_tokens,
                            weights=(0.25, 0.25, 0.25, 0.25),
                            smoothing_function=self.smoothie)
        return bleu1, bleu4

    def compute_meteor(self, reference, candidate):
        """Compute METEOR"""
        try:
            return meteor_score([reference.split()], candidate.split())
        except:
            return 0.0

    def compute_rouge_l(self, reference, candidate):
        """Compute ROUGE-L"""
        scores = self.rouge_scorer.score(reference, candidate)
        return scores['rougeL'].fmeasure

    def compute_cider(self, references, candidates):
        """Compute CIDEr"""
        if isinstance(references, str):
            references = [references]
        if isinstance(candidates, str):
            candidates = [candidates]

        refs_dict = {i: [ref] for i, ref in enumerate(references)}
        cands_dict = {i: [cand] for i, cand in enumerate(candidates)}

        try:
            score, _ = self.cider_scorer.compute_score(refs_dict, cands_dict)
            return score if not isinstance(score, list) else score[0]
        except:
            return 0.0

    def compute_spice(self, references, candidates):
        """Compute SPICE"""
        if isinstance(references, str):
            references = [references]
        if isinstance(candidates, str):
            candidates = [candidates]

        refs_dict = {i: [ref] for i, ref in enumerate(references)}
        cands_dict = {i: [cand] for i, cand in enumerate(candidates)}

        try:
            scorer = Spice()
            score, _ = scorer.compute_score(refs_dict, cands_dict)
            return score if not isinstance(score, list) else score[0]
        except:
            return 0.0


def load_data(clean_images_path, clean_captions_path, target_captions_path):
    """Load images and captions"""
    with open(clean_captions_path, 'r', encoding='utf-8') as f:
        clean_captions = [line.strip() for line in f.readlines()]
    with open(target_captions_path, 'r', encoding='utf-8') as f:
        target_captions = [line.strip() for line in f.readlines()]

    image_files = sorted([f for f in os.listdir(clean_images_path)
                         if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))])

    return image_files, clean_captions, target_captions


def main():
    # Path configuration
    clean_images_path = r"D:\dmta_data\clean_images"
    clean_captions_path = r"D:\dmta_data\clean_captions.txt"
    target_captions_path = r"D:\dmta_data\target_captions.txt"
    target_images_path = r"D:\dmta_data\target_images"
    output_dir = r"D:\ts_output_blip2"

    os.makedirs(output_dir, exist_ok=True)

    # BLIP-2 model variants (try these if one fails)
    # - "Salesforce/blip2-flan-t5-xl" (Recommended - more stable)
    # - "Salesforce/blip2-flan-t5-xxl" (Largest, best quality)
    # - "Salesforce/blip2-opt-2.7b-coco" (COCO fine-tuned)

    # Initialize TripletShift attack with BLIP-2 (using Flan-T5 XL which is more stable)
    attack = TripletShift(
        epsilon=8/255,      # epsilon = 8 as in Section 4.2
        gamma_step=1/255,   # gamma step size
        K=100,              # K = 100 as in Section 4.2
        alpha_dc=1.0,       # alpha = 1.0 (optimal from ablation)
        beta_dc=5.0,        # beta = 5.0 (optimal from ablation)
        lambda_p=0.1,       # lambda = 0.1 (optimal from ablation)
        mu=0.001,           # mu = 0.001 (optimal from ablation)
        blip2_model_name="Salesforce/blip2-flan-t5-xl",  # Changed to Flan-T5 XL (more stable)
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    evaluator = EvaluationMetrics()

    # Load data
    image_files, clean_captions, target_captions = load_data(
        clean_images_path, clean_captions_path, target_captions_path
    )

    num_samples = min(len(image_files), len(clean_captions), len(target_captions))
    print(f"\nProcessing {num_samples} samples\n")

    image_files = image_files[:num_samples]
    clean_captions = clean_captions[:num_samples]
    target_captions = target_captions[:num_samples]

    # Storage for results
    all_gen_captions = []
    all_target_captions = []
    results = {
        'bleu1': [], 'bleu4': [], 'meteor': [], 'rouge_l': [],
        'sim_target': [], 'sim_source': [], 'sim_gap': [],
        'lpips': [],           # Perceptual distance
        'clip_score': [],      # CLIP Score using the same encoder
    }

    for idx, (img_file, clean_cap, target_cap) in enumerate(
        tqdm(zip(image_files, clean_captions, target_captions), total=num_samples, desc="TripletShift Attack with BLIP-2")):

        try:
            print(f"\n{'='*60}")
            print(f"[{idx+1}/{num_samples}] Processing: {img_file}")
            print(f"Target caption: {target_cap[:80]}...")

            # Load clean image
            img_path = os.path.join(clean_images_path, img_file)
            clean_pil = Image.open(img_path).convert('RGB')
            clean_image = attack.transform(clean_pil).unsqueeze(0).to(attack.device)

            # Load target image if available
            target_image = None
            target_img_path = os.path.join(target_images_path, img_file)
            if os.path.exists(target_img_path):
                target_pil = Image.open(target_img_path).convert('RGB')
                target_image = attack.transform(target_pil).unsqueeze(0).to(attack.device)

            # Generate adversarial image
            print("\nGenerating adversarial image with TripletShift...")
            adv_image, gen_metrics = attack.generate_adversarial_image(clean_image, target_cap, target_image)

            # Compute similarities
            with torch.no_grad():
                adv_emb = attack.encoder.encode_image(adv_image)
                src_emb = attack.encoder.encode_image(clean_image)

                if target_image is not None:
                    tgt_emb = attack.encoder.encode_image(target_image)
                else:
                    tgt_emb = attack.encoder.encode_text([target_cap])

                sim_target = (adv_emb * tgt_emb).sum().item()
                sim_source = (adv_emb * src_emb).sum().item()
                sim_gap = sim_target - sim_source

            # Compute LPIPS (perceptual distance)
            lpips_score = attack.compute_perceptual_distance(adv_image, clean_image)

            # Compute CLIP Score using the same encoder
            clip_score = attack.encoder.compute_clip_score(adv_image, target_cap)

            # Generate caption using BLIP-2
            generated_caption = attack.generate_caption(adv_image)
            all_gen_captions.append(generated_caption)
            all_target_captions.append(target_cap)

            # Compute metrics
            bleu1, bleu4 = evaluator.compute_bleu(target_cap, generated_caption)
            meteor = evaluator.compute_meteor(target_cap, generated_caption)
            rouge_l = evaluator.compute_rouge_l(target_cap, generated_caption)

            # Store results
            results['bleu1'].append(bleu1)
            results['bleu4'].append(bleu4)
            results['meteor'].append(meteor)
            results['rouge_l'].append(rouge_l)
            results['sim_target'].append(sim_target)
            results['sim_source'].append(sim_source)
            results['sim_gap'].append(sim_gap)
            results['lpips'].append(lpips_score)
            results['clip_score'].append(clip_score)

            # Save adversarial image
            adv_denorm = attack.unnormalize(adv_image)
            adv_denorm = torch.clamp(adv_denorm, 0, 1)
            adv_pil = transforms.ToPILImage()(adv_denorm.squeeze().cpu())
            adv_pil.save(os.path.join(output_dir, f"adv_{idx:05d}_{img_file}"))

            # Print results
            print(f"\n{'='*50}")
            print(f"TARGET:    {target_cap[:80]}...")
            print(f"GENERATED (BLIP-2): {generated_caption[:80]}...")
            print(f"{'='*50}")
            print(f"CAPTION METRICS:")
            print(f"  BLEU-1:   {bleu1*100:.2f}%")
            print(f"  BLEU-4:   {bleu4*100:.2f}%")
            print(f"  METEOR:   {meteor*100:.2f}%")
            print(f"  ROUGE-L:  {rouge_l*100:.2f}%")
            print(f"\nEMBEDDING METRICS:")
            print(f"  Sim Target: {sim_target:.4f} | Sim Source: {sim_source:.4f} | Gap: {sim_gap:+.4f}")
            print(f"\nPERCEPTUAL METRIC:")
            print(f"  LPIPS:     {lpips_score:.4f}")
            print(f"\nCLIP SCORE (ViT-B/16):")
            print(f"  {clip_score:.2f}")

            if sim_gap > 0:
                print("\n✓ SUCCESS: Adversarial embedding is closer to target than source")

        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compute corpus-level metrics
    if all_gen_captions:
        print("\nComputing CIDEr and SPICE...")
        cider = evaluator.compute_cider(all_target_captions, all_gen_captions)
        spice = evaluator.compute_spice(all_target_captions, all_gen_captions)

    # Print final results
    if results['bleu1']:
        print("\n" + "="*70)
        print("TripletShift ATTACK - FINAL RESULTS (BLIP-2 Victim)")
        print("="*70)

        print(f"\n{'='*50}")
        print("CAPTION QUALITY METRICS:")
        print(f"{'='*50}")
        print(f"BLEU-1:    {np.mean(results['bleu1'])*100:.2f}%")
        print(f"BLEU-4:    {np.mean(results['bleu4'])*100:.2f}%")
        print(f"METEOR:    {np.mean(results['meteor'])*100:.2f}%")
        print(f"ROUGE-L:   {np.mean(results['rouge_l'])*100:.2f}%")
        print(f"CIDEr:     {cider:.2f}")
        print(f"SPICE:     {spice:.2f}")

        print(f"\n{'='*50}")
        print("EMBEDDING SIMILARITY METRICS:")
        print(f"{'='*50}")
        print(f"Avg Sim Target: {np.mean(results['sim_target']):.4f}")
        print(f"Avg Sim Source: {np.mean(results['sim_source']):.4f}")
        print(f"Avg Sim Gap:    {np.mean(results['sim_gap']):+.4f}")

        print(f"\n{'='*50}")
        print("PERCEPTUAL METRIC (LPIPS):")
        print(f"{'='*50}")
        print(f"Avg LPIPS:   {np.mean(results['lpips']):.4f}")

        print(f"\n{'='*50}")
        print("CLIP SCORE (ViT-B/16):")
        print(f"{'='*50}")
        print(f"Avg CLIP Score: {np.mean(results['clip_score']):.2f}")

        # Success rate
        success_count = sum(1 for g in results['sim_gap'] if g > 0)
        print(f"\n{'='*50}")
        print(f"Success Rate (gap > 0): {success_count}/{num_samples} ({success_count/num_samples*100:.1f}%)")

        # Save results
        final_results = {
            'model': 'BLIP-2',
            'blip2_model_variant': attack.blip2_model_name,
            'metrics': {
                'bleu1': float(np.mean(results['bleu1'])),
                'bleu4': float(np.mean(results['bleu4'])),
                'meteor': float(np.mean(results['meteor'])),
                'rouge_l': float(np.mean(results['rouge_l'])),
                'cider': float(cider),
                'spice': float(spice),
                'sim_target': float(np.mean(results['sim_target'])),
                'sim_source': float(np.mean(results['sim_source'])),
                'sim_gap': float(np.mean(results['sim_gap'])),
                'lpips': float(np.mean(results['lpips'])),
                'clip_score': float(np.mean(results['clip_score'])),
            },
            'success_rate': success_count / num_samples,
            'detailed': results
        }

        with open(os.path.join(output_dir, "ts_results_blip2.json"), 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"\n✓ Results saved to {output_dir}")


if __name__ == "__main__":
    import numpy as np
    main()
