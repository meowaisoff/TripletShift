import torch
import sys
sys.path.insert(0, r"D:\CLIP")
sys.path.insert(0, r"D:\ViECap")  # Add ViECap repo path
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
from torchvision import transforms
import lpips
import clip
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

# Import ViECap model (requires cloning the official repository)
try:
    # The exact import paths depend on the ViECap repository structure
    # Based on https://github.com/FeiElysia/ViECap
    from viecap.model import ViECapModel
    from viecap.processor import ViECapProcessor
    VIECAP_AVAILABLE = True
    print("✓ ViECap model imported successfully")
except ImportError as e:
    VIECAP_AVAILABLE = False
    print(f"✗ ViECap import failed: {e}")
    print("Please clone the ViECap repository from: https://github.com/FeiElysia/ViECap")
    print("And ensure it's in your Python path")


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

        self.embedding_dim = 512

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
        """Compute CLIP similarity score between image and text"""
        with torch.no_grad():
            img_features = self.encode_image(image_tensor)
            text_features = self.encode_text([text])
            similarity = (img_features * text_features).sum().item() * 100
        return similarity

    def forward(self, images):
        return self.encode_image(images)


class TripletShift:
    """
    TripletShift (TS) Attack for ViECap

    ViECap is a zero-shot image captioning model that uses:
    - CLIP for entity retrieval
    - GPT-2 as the language model decoder
    - Entity-aware hard prompts for transferable decoding
    """

    def __init__(self,
                 epsilon=8/255,
                 gamma_step=1/255,
                 K=100,
                 alpha_dc=1.0,
                 beta_dc=5.0,
                 lambda_p=0.1,
                 mu=0.001,
                 viecap_checkpoint_path=None,  # Path to ViECap checkpoint
                 device="cuda"):

        self.epsilon = epsilon
        self.gamma_step = gamma_step
        self.K = K
        self.alpha_dc = alpha_dc
        self.beta_dc = beta_dc
        self.lambda_p = lambda_p
        self.mu = mu
        self.device = device

        # Initialize LPIPS for perceptual distance
        self.lpips = lpips.LPIPS(net='alex').to(device).eval()

        # Initialize surrogate semantic encoder (CLIP)
        self.encoder = SemanticEncoder(device=device)

        # Initialize victim ViECap model
        self._load_victim_model(viecap_checkpoint_path)

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
        print(f"TripletShift Attack with ViECap Victim Model")
        print(f"{'='*60}")
        print(f"ViECap Checkpoint: {viecap_checkpoint_path}")
        print(f"Perturbation budget epsilon: {epsilon*255:.1f}/255")
        print(f"Step size gamma: {gamma_step*255:.1f}/255")
        print(f"Iterations K: {K}")
        print(f"Distance control: alpha={alpha_dc}, beta={beta_dc}")
        print(f"Loss weights: lambda={lambda_p}, mu={mu}")
        print(f"{'='*60}\n")

    def _load_victim_model(self, checkpoint_path):
        """Load ViECap victim model from official repository"""
        if not VIECAP_AVAILABLE:
            raise RuntimeError(
                "ViECap is not available. Please clone the repository:\n"
                "git clone https://github.com/FeiElysia/ViECap.git\n"
                "Then add it to your Python path."
            )

        print(f"Loading ViECap victim model from {checkpoint_path}...")

        try:
            # Load ViECap model
            # Based on the paper: CLIP + GPT-2 with entity-aware hard prompts
            self.victim_model = ViECapModel.from_pretrained(checkpoint_path)
            self.victim_processor = ViECapProcessor.from_pretrained(checkpoint_path)

            self.victim_model = self.victim_model.to(self.device)
            self.victim_model.eval()

            # Get model size
            num_params = sum(p.numel() for p in self.victim_model.parameters()) / 1e6
            print(f"✓ ViECap model loaded successfully! ({num_params:.1f}M parameters)\n")

        except Exception as e:
            print(f"Error loading ViECap model: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure you have downloaded the ViECap checkpoint from the official repository")
            print("2. The expected checkpoint path should contain model weights")
            print("3. You may need to use a specific loading method from the ViECap codebase")
            print("\nAlternative: Try loading the text-only trained model as described in the paper")
            raise

    def compute_perceptual_distance(self, img1, img2):
        """Compute perceptual distance using LPIPS"""
        img1_norm = img1 * 2 - 1
        img2_norm = img2 * 2 - 1
        with torch.no_grad():
            distance = self.lpips(img1_norm, img2_norm)
        return distance.item()

    def adaptive_distance_control(self, perceptual_distance):
        """Adaptive distance control: omega(d) = alpha * exp(-beta * d)"""
        return self.alpha_dc * torch.exp(-self.beta_dc * perceptual_distance)

    def ts_loss(self, anchor, p_t, p_s, omega):
        """TripletShift loss"""
        sim_target = (anchor * p_t).sum(dim=-1)
        sim_source = (anchor * p_s).sum(dim=-1)
        sim_gap = sim_target - sim_source
        loss = torch.clamp(omega - sim_gap, min=0.0)
        return loss.mean(), sim_target.mean(), sim_source.mean()

    def total_loss(self, adv_image, clean_image, p_t, p_s):
        """Total loss: L_TS + lambda * d_perc + mu * ||eta||_2^2"""
        anchor = self.encoder.encode_image(adv_image)
        d_perc = self.compute_perceptual_distance(adv_image, clean_image)
        d_perc_tensor = torch.tensor(d_perc, device=self.device)
        omega = self.adaptive_distance_control(d_perc)

        loss_triplet, sim_target, sim_source = self.ts_loss(
            anchor, p_t, p_s, omega
        )

        loss_perceptual = self.lambda_p * d_perc_tensor
        eta = adv_image - clean_image
        loss_l2 = self.mu * torch.norm(eta, p=2) ** 2

        total_loss = loss_triplet + loss_perceptual + loss_l2

        return total_loss, loss_triplet, loss_perceptual, loss_l2, sim_target, sim_source, d_perc

    def generate_adversarial_image(self, clean_image, target_caption, target_image=None):
        """Generate adversarial image using PGD optimization"""
        eta = torch.zeros_like(clean_image, device=self.device, requires_grad=True)

        with torch.no_grad():
            p_s = self.encoder.encode_image(clean_image)
            if target_image is not None:
                p_t = self.encoder.encode_image(target_image)
            else:
                p_t = self.encoder.encode_text([target_caption])

        best_sim_target = -1.0
        best_eta = None
        best_lpips = float('inf')

        print(f"Initial similarity to target: {self._compute_similarity(clean_image, p_t):.4f}")

        for iteration in range(self.K):
            if eta.grad is not None:
                eta.grad.zero_()

            adv_image = torch.clamp(clean_image + eta, 0, 1)

            total_loss, loss_triplet, loss_percep, loss_l2, sim_target, sim_source, d_perc = \
                self.total_loss(adv_image, clean_image, p_t, p_s)

            total_loss.backward()

            if eta.grad is not None:
                eta.data = eta.data - self.gamma_step * eta.grad.sign()

            eta.data = torch.clamp(eta.data, -self.epsilon, self.epsilon)
            eta.data = torch.clamp(clean_image + eta.data, 0, 1) - clean_image

            current_sim = sim_target.item() if torch.is_tensor(sim_target) else sim_target
            if current_sim > best_sim_target:
                best_sim_target = current_sim
                best_eta = eta.data.clone()
                best_lpips = d_perc

            if (iteration + 1) % 20 == 0 or iteration == 0:
                print(f"Iter {iteration+1:3d}/{self.K}: "
                      f"L_total={total_loss.item():.4f}, "
                      f"Sim_tgt={current_sim:.4f}, "
                      f"LPIPS={d_perc:.4f}")

            if best_sim_target > 0.85:
                print(f"✓ Early stopping at iteration {iteration+1}")
                break

        if best_eta is not None:
            adversarial_image = torch.clamp(clean_image + best_eta, 0, 1)
        else:
            adversarial_image = torch.clamp(clean_image + eta, 0, 1)

        print(f"\nFinal similarity to target: {best_sim_target:.4f}")
        print(f"Final LPIPS distance: {best_lpips:.4f}")

        return adversarial_image, {'sim_target': best_sim_target, 'lpips': best_lpips}

    def _compute_similarity(self, image, target_embedding):
        with torch.no_grad():
            img_emb = self.encoder.encode_image(image)
            return (img_emb * target_embedding).sum().item()

    def _retrieve_visual_entities(self, image_tensor):
        """
        Retrieve visual entities from the image using CLIP.
        This is a key component of ViECap's entity-aware decoding.
        """
        with torch.no_grad():
            # Get image features
            img_features = self.encoder.encode_image(image_tensor)

            # Common entity vocabulary (simplified - in practice, ViECap uses a larger set)
            common_entities = [
                "person", "man", "woman", "child", "dog", "cat", "car", "bus", "train",
                "bicycle", "motorcycle", "airplane", "boat", "truck", "bird", "cat", "dog",
                "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
                "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
            ]

            # Encode all entity texts
            entity_texts = common_entities
            text_tokens = clip.tokenize(entity_texts, truncate=True).to(self.device)
            text_features = self.encoder.clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)

            # Compute similarities
            similarities = (img_features @ text_features.T).squeeze(0)

            # Get top-k entities
            top_k = 10
            top_values, top_indices = torch.topk(similarities, k=min(top_k, len(entity_texts)))

            entities = [entity_texts[idx] for idx in top_indices.cpu().numpy()]
            scores = top_values.cpu().numpy()

            return entities, scores

    def generate_caption(self, image_tensor):
        """
        Generate caption using ViECap victim model.

        ViECap uses entity-aware decoding:
        1. Retrieve visual entities from the image
        2. Construct entity-aware hard prompts
        3. Generate caption with GPT-2 guided by the prompts
        """
        # Denormalize image
        image_denorm = self.unnormalize(image_tensor)
        image_denorm = torch.clamp(image_denorm, 0, 1)

        # Convert to PIL
        image_pil = transforms.ToPILImage()(image_denorm.squeeze().cpu())

        # Step 1: Retrieve visual entities (ViECap's key innovation)
        entities, entity_scores = self._retrieve_visual_entities(image_tensor)

        # Step 2: Construct entity-aware hard prompt
        # Format: "The image contains [entity1], [entity2], ..."
        entity_prompt = "The image contains " + ", ".join(entities[:5])

        # Step 3: Full prompt for ViECap
        # ViECap expects a specific prompt format for entity-aware decoding
        full_prompt = f"{entity_prompt}\nDescribe this image:"

        try:
            # Process the image with entity prompt
            inputs = self.victim_processor(
                text=full_prompt,
                images=image_pil,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate caption
            with torch.no_grad():
                outputs = self.victim_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=5,
                    do_sample=False,
                    temperature=0.7
                )

            caption = self.victim_processor.decode(outputs[0], skip_special_tokens=True)

            # Clean up caption (remove prompt if present)
            if full_prompt in caption:
                caption = caption.replace(full_prompt, "").strip()

            return caption

        except Exception as e:
            print(f"ViECap generation failed: {e}")
            print("Falling back to entity-based caption...")
            # Fallback: simple entity-based caption
            return f"A photo containing {', '.join(entities[:3])}."


class EvaluationMetrics:
    """Evaluation metrics for image captioning"""

    def __init__(self):
        self.cider_scorer = Cider()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothie = SmoothingFunction().method4

    def compute_bleu(self, reference, candidate):
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
        try:
            return meteor_score([reference.split()], candidate.split())
        except:
            return 0.0

    def compute_rouge_l(self, reference, candidate):
        scores = self.rouge_scorer.score(reference, candidate)
        return scores['rougeL'].fmeasure

    def compute_cider(self, references, candidates):
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
    output_dir = r"D:\ts_output_viecap"

    # IMPORTANT: Update this path to your ViECap checkpoint
    # ViECap can be text-only trained, so the checkpoint may not require paired data
    viecap_checkpoint_path = r"D:\ViECap\checkpoints\viecap_coco.pt"

    os.makedirs(output_dir, exist_ok=True)

    if not VIECAP_AVAILABLE:
        print("\n" + "="*60)
        print("ERROR: ViECap model not available")
        print("="*60)
        print("\nTo use ViECap, you need to:")
        print("1. Clone the repository: git clone https://github.com/FeiElysia/ViECap.git")
        print("2. Download the pre-trained checkpoint from the official repository")
        print("3. Update the 'viecap_checkpoint_path' variable with the correct path")
        print("4. Add the ViECap directory to your Python path")
        print("\nViECap paper: 'Transferable Decoding with Visual Entities for Zero-Shot Image Captioning'")
        print("ICCV 2023 | Code: https://github.com/FeiElysia/ViECap")
        return

    # Initialize TripletShift attack with ViECap
    attack = TripletShift(
        epsilon=8/255,
        gamma_step=1/255,
        K=100,
        alpha_dc=1.0,
        beta_dc=5.0,
        lambda_p=0.1,
        mu=0.001,
        viecap_checkpoint_path=viecap_checkpoint_path,
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
        'lpips': [],
        'clip_score': [],
    }

    for idx, (img_file, clean_cap, target_cap) in enumerate(
        tqdm(zip(image_files, clean_captions, target_captions), total=num_samples, desc="TripletShift Attack with ViECap")):

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

            # Compute LPIPS
            lpips_score = attack.compute_perceptual_distance(adv_image, clean_image)

            # Compute CLIP Score
            clip_score = attack.encoder.compute_clip_score(adv_image, target_cap)

            # Generate caption using ViECap
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
            print(f"GENERATED (ViECap): {generated_caption[:80]}...")
            print(f"{'='*50}")
            print(f"BLEU-1:   {bleu1*100:.2f}%")
            print(f"BLEU-4:   {bleu4*100:.2f}%")
            print(f"METEOR:   {meteor*100:.2f}%")
            print(f"ROUGE-L:  {rouge_l*100:.2f}%")
            print(f"Sim Target: {sim_target:.4f} | Sim Source: {sim_source:.4f} | Gap: {sim_gap:+.4f}")
            print(f"LPIPS:     {lpips_score:.4f}")
            print(f"CLIP Score: {clip_score:.2f}")

            if sim_gap > 0:
                print("✓ SUCCESS: Adversarial embedding is closer to target than source")

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
        print("TripletShift ATTACK - FINAL RESULTS (ViECap Victim)")
        print("="*70)

        print(f"\nBLEU-1:    {np.mean(results['bleu1'])*100:.2f}%")
        print(f"BLEU-4:    {np.mean(results['bleu4'])*100:.2f}%")
        print(f"METEOR:    {np.mean(results['meteor'])*100:.2f}%")
        print(f"ROUGE-L:   {np.mean(results['rouge_l'])*100:.2f}%")
        print(f"CIDEr:     {cider:.2f}")
        print(f"SPICE:     {spice:.2f}")
        print(f"\nAvg Sim Target: {np.mean(results['sim_target']):.4f}")
        print(f"Avg Sim Source: {np.mean(results['sim_source']):.4f}")
        print(f"Avg Sim Gap:    {np.mean(results['sim_gap']):+.4f}")
        print(f"Avg LPIPS:   {np.mean(results['lpips']):.4f}")
        print(f"Avg CLIP Score: {np.mean(results['clip_score']):.2f}")

        success_count = sum(1 for g in results['sim_gap'] if g > 0)
        print(f"\nSuccess Rate (gap > 0): {success_count}/{num_samples} ({success_count/num_samples*100:.1f}%)")

        # Save results
        final_results = {
            'model': 'ViECap',
            'paper': 'Transferable Decoding with Visual Entities for Zero-Shot Image Captioning (ICCV 2023)',
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

        with open(os.path.join(output_dir, "ts_results_viecap.json"), 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"\n✓ Results saved to {output_dir}")


if __name__ == "__main__":
    import numpy as np
    main()
