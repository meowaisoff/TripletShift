"""
AnyAttack-Inspired TripletShift with InstructBLIP (China-Compatible)
=============================================================
Using AI-ModelScope/instructblip-vicuna-7b from ModelScope
"""

import os
os.environ['MODELSCOPE_CACHE'] = os.path.expanduser("~/.cache/modelscope/hub")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

import numpy as np
from PIL import Image
from torchvision import transforms
import lpips
import os
sys.path.insert(0, r"D:\CLIP")
import clip
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from tqdm import tqdm
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import warnings
warnings.filterwarnings('ignore')


class AnyAttackDecoder(nn.Module):
    """
    AnyAttack-style Decoder: Maps CLIP embeddings to adversarial noise
    """
    def __init__(self, embed_dim=512, noise_dim=128):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, noise_dim * 8 * 8),
            nn.ReLU(inplace=True)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

        self.noise_dim = noise_dim

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.noise_dim, 8, 8)
        noise = self.deconv(x)
        if noise.shape[-1] != 224:
            noise = F.interpolate(noise, size=(224, 224), mode='bilinear', align_corners=False)
        return noise


class AnyAttackInspiredTripletShift:
    """
    AnyAttack-Inspired TripletShift with InstructBLIP (Vicuna-7B via ModelScope)
    """

    def __init__(self,
                 epsilon=16/255,
                 gamma_step=1.0/255,
                 K=300,
                 alpha_dc=0.3,
                 beta_dc=3.0,
                 lambda_p=2.0,
                 mu=0.001,
                 momentum=0.9,
                 decoder_lr=0.01,
                 use_pretrained_decoder=True,
                 device="cuda"):

        self.epsilon = epsilon
        self.gamma_step = gamma_step
        self.K = K
        self.alpha_dc = alpha_dc
        self.beta_dc = beta_dc
        self.lambda_p = lambda_p
        self.mu = mu
        self.momentum = momentum
        self.decoder_lr = decoder_lr
        self.use_pretrained_decoder = use_pretrained_decoder

        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"\n{'='*70}")
        print(f"ANYATTACK-INSPIRED TripletShift with InstructBLIP-Vicuna (ModelScope)")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Epsilon: {epsilon*255:.1f}/255 | Gamma: {gamma_step*255:.1f}/255 | K: {K}")

        # LPIPS
        print("\nSetting up LPIPS...")
        self.lpips_model = lpips.LPIPS(net='alex').to(self.device).eval()

        # CLIP (surrogate encoder) - FROZEN
        print("Loading CLIP encoder (frozen)...")
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Initialize decoder
        self.decoder = self._initialize_decoder()

        # InstructBLIP victim model - ModelScope version
        self._load_victim_model()

        # Transforms
        self.clip_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])

        # InstructBLIP uses different normalization
        self.instructblip_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])

        self.unnormalize = transforms.Normalize(
            mean=[-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711],
            std=[1/0.26862954, 1/0.26130258, 1/0.27577711]
        )

        print("\n✓ AnyAttack-Inspired TripletShift initialized\n")

    def _initialize_decoder(self):
        """Initialize AnyAttack-style decoder"""
        print("Initializing AnyAttack-style decoder...")
        decoder = AnyAttackDecoder(embed_dim=512, noise_dim=128).to(self.device)

        if self.use_pretrained_decoder:
            decoder_path = os.path.join(os.path.dirname(__file__), "anyattack_decoder_pretrained.pth")
            if os.path.exists(decoder_path):
                try:
                    state_dict = torch.load(decoder_path, map_location=self.device)
                    decoder.load_state_dict(state_dict)
                    print(f"✓ Loaded pre-trained decoder")
                except Exception as e:
                    print(f"Could not load pre-trained decoder: {e}")
                    print("Using random initialization + fine-tuning")
            else:
                print("No pre-trained decoder found. Will use random init + fine-tuning.")

        return decoder

    def _load_victim_model(self):
        """
        Load InstructBLIP Vicuna-7B from ModelScope
        CORRECTED MODEL ID: AI-ModelScope/instructblip-vicuna-7b
        """
        print("Loading InstructBLIP-Vicuna-7B from ModelScope...")

        # CORRECT ModelScope model ID for InstructBLIP
        modelscope_model_id = "AI-ModelScope/instructblip-vicuna-7b"

        # Local cache paths
        local_cache_paths = [
            r"D:\models\AI-ModelScope\instructblip-vicuna-7b",
            r"D:\models\instructblip-vicuna-7b",
            os.path.expanduser("~/.cache/modelscope/hub/AI-ModelScope/instructblip-vicuna-7b"),
            "./models/AI-ModelScope/instructblip-vicuna-7b",
            "./models/instructblip-vicuna-7b",
        ]

        # Try local cache first
        for cache_path in local_cache_paths:
            if os.path.exists(cache_path):
                try:
                    print(f"Found local cache: {cache_path}")
                    self.victim_processor = InstructBlipProcessor.from_pretrained(
                        cache_path, local_files_only=True
                    )
                    self.victim_model = InstructBlipForConditionalGeneration.from_pretrained(
                        cache_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        local_files_only=True
                    )
                    self.victim_model = self.victim_model.to(self.device)
                    self.victim_model.eval()
                    print(f"✓ InstructBLIP-Vicuna-7B loaded from local cache\n")
                    return
                except Exception as e:
                    print(f"Local cache failed: {e}, trying next...")
                    continue

        # Try ModelScope download
        try:
            print(f"Downloading from ModelScope: {modelscope_model_id}")
            from modelscope import snapshot_download

            cache_dir = os.environ.get('MODELSCOPE_CACHE', './models')
            model_dir = snapshot_download(modelscope_model_id, cache_dir=cache_dir)

            self.victim_processor = InstructBlipProcessor.from_pretrained(model_dir)
            self.victim_model = InstructBlipForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.victim_model = self.victim_model.to(self.device)
            self.victim_model.eval()
            print(f"✓ InstructBLIP-Vicuna-7B loaded from ModelScope\n")

        except Exception as e:
            print(f"ModelScope download failed: {e}")
            print("\n" + "="*70)
            print("MANUAL DOWNLOAD REQUIRED")
            print("="*70)
            print("Please download the model manually:")
            print(f"Method 1 - ModelScope CLI:")
            print(f"  modelscope download --model {modelscope_model_id} --local_dir ./models/instructblip")
            print(f"\nMethod 2 - Git Clone:")
            print(f"  git lfs install")
            print(f"  git clone https://www.modelscope.cn/{modelscope_model_id}.git ./models/instructblip")
            print("="*70 + "\n")
            raise RuntimeError("Failed to load InstructBLIP model")

    def get_image_embedding(self, images):
        """Get CLIP image embedding - FROZEN encoder"""
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            return features

    def get_text_embedding(self, text):
        """Get CLIP text embedding"""
        with torch.no_grad():
            text_tokens = clip.tokenize([text], truncate=True).to(self.device)
            features = self.clip_model.encode_text(text_tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            return features

    def compute_lpips(self, img1, img2):
        """Compute LPIPS perceptual distance"""
        img1_norm = img1 * 2 - 1
        img2_norm = img2 * 2 - 1
        with torch.no_grad():
            return self.lpips_model(img1_norm, img2_norm)

    def adaptive_distance_control(self, d_perc):
        """Adaptive distance control: omega(d) = alpha * exp(-beta * d)"""
        return self.alpha_dc * torch.exp(-self.beta_dc * d_perc)

    def anyattack_generate_noise(self, target_embedding):
        """Generate initial noise using AnyAttack-style decoder"""
        self.decoder.eval()
        with torch.no_grad():
            noise = self.decoder(target_embedding)
            noise = noise * self.epsilon
        return noise

    def fine_tune_decoder(self, target_emb, clean_img, p_s, omega, steps=10):
        """Fine-tune decoder on specific target (AnyAttack-style adaptation)"""
        self.decoder.train()
        optimizer = optim.Adam(self.decoder.parameters(), lr=self.decoder_lr)

        for _ in range(steps):
            optimizer.zero_grad()
            noise = self.decoder(target_emb)
            noise = torch.clamp(noise * self.epsilon, -self.epsilon, self.epsilon)
            adv_img = torch.clamp(clean_img + noise, 0, 1)
            adv_emb = self.get_image_embedding(adv_img)

            sim_target = (adv_emb * target_emb).sum(dim=-1).mean()
            sim_source = (adv_emb * p_s).sum(dim=-1).mean()

            loss = torch.clamp(omega - (sim_target - sim_source), min=0.0).mean() - sim_target * 0.1
            loss.backward()
            optimizer.step()

        self.decoder.eval()

    def generate_adversarial_image(self, clean_image, target_caption, target_image_tensor=None):
        """
        Generate adversarial image using AnyAttack-inspired approach
        """
        # Get target embedding
        if target_image_tensor is not None:
            print("Using target image embedding via decoder")
            with torch.no_grad():
                p_t = self.get_image_embedding(target_image_tensor)
        else:
            print("Using target text embedding")
            p_t = self.get_text_embedding(target_caption)

        with torch.no_grad():
            p_s = self.get_image_embedding(clean_image)

        # STEP 1: Generate initial noise using decoder
        print("Generating initial noise via AnyAttack-style decoder...")
        eta = self.anyattack_generate_noise(p_t)
        eta.requires_grad = True

        # STEP 2: Optional fine-tuning of decoder
        if self.use_pretrained_decoder and self.decoder_lr > 0:
            print("Fine-tuning decoder for target adaptation...")
            with torch.no_grad():
                adv_temp = torch.clamp(clean_image + eta, 0, 1)
                perceptual_dist = self.compute_lpips(adv_temp, clean_image)
                omega = self.adaptive_distance_control(perceptual_dist)

            self.fine_tune_decoder(p_t, clean_image, p_s, omega, steps=5)

            with torch.no_grad():
                eta = self.anyattack_generate_noise(p_t)
            eta.requires_grad = True

        # STEP 3: TripletShift refinement
        print("TripletShift refinement...")
        optimizer = optim.Adam([eta], lr=self.gamma_step, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.K, eta_min=self.gamma_step/10)

        velocity = torch.zeros_like(eta)
        best_sim_target = -1.0
        best_eta = None

        for iteration in range(self.K):
            optimizer.zero_grad()
            adv_image = torch.clamp(clean_image + eta, 0, 1)
            a = self.get_image_embedding(adv_image)

            with torch.no_grad():
                perceptual_dist = self.compute_lpips(adv_image, clean_image)
                omega = self.adaptive_distance_control(perceptual_dist)

            sim_target = (a * p_t).sum(dim=-1).mean()
            sim_source = (a * p_s).sum(dim=-1).mean()
            sim_gap = sim_target - sim_source

            loss_triplet = torch.clamp(omega - sim_gap, min=0.0).mean()
            loss_perceptual = perceptual_dist.mean()
            loss_l2 = torch.norm(eta, p=2)

            total_loss = loss_triplet + self.lambda_p * loss_perceptual + self.mu * loss_l2
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_([eta], max_norm=2.0)

            if eta.grad is not None:
                velocity = self.momentum * velocity + (1 - self.momentum) * eta.grad
                eta.data = eta.data - self.gamma_step * velocity.sign()

            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                eta.data = torch.clamp(eta.data, -self.epsilon, self.epsilon)
                eta.data = torch.clamp(clean_image + eta.data, 0, 1) - clean_image

            current_sim = sim_target.item()
            if current_sim > best_sim_target:
                best_sim_target = current_sim
                best_eta = eta.detach().clone()

            if (iteration + 1) % 50 == 0 or iteration == 0:
                print(f"Iter {iteration+1}/{self.K}: "
                      f"Sim_Tgt={current_sim:.4f} (best={best_sim_target:.4f})")

            if best_sim_target > 0.9:
                print(f"\n✓ Early stop at {iteration}, sim={best_sim_target:.4f}")
                break

        if best_eta is not None:
            final_eta = best_eta
        else:
            final_eta = eta.detach()

        adv_image = torch.clamp(clean_image + final_eta, 0, 1)
        return adv_image

    def generate_caption(self, image_tensor):
        """Generate caption with InstructBLIP-Vicuna"""
        # Unnormalize and convert to PIL
        image_denorm = self.unnormalize(image_tensor)
        image_denorm = torch.clamp(image_denorm, 0, 1)
        image_pil = transforms.ToPILImage()(image_denorm.squeeze().cpu())

        # Process with InstructBLIP - Vicuna version uses different prompt format
        # For Vicuna-based models, use a simple instruction
        prompt = "Describe this image briefly."
        inputs = self.victim_processor(images=image_pil, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.victim_model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=5,
                do_sample=False,
                early_stopping=True
            )

        caption = self.victim_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return caption


class EvaluationMetrics:
    """Standard metrics for image captioning evaluation"""

    def __init__(self, device="cpu"):
        self.device = device
        self.scorer_cider = Cider()
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
            score, _ = self.scorer_cider.compute_score(refs_dict, cands_dict)
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

    def compute_clip_score(self, image_tensor, text, clip_model, device):
        with torch.no_grad():
            img_feat = clip_model.encode_image(image_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            text_tokens = clip.tokenize([text], truncate=True).to(device)
            text_feat = clip_model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ text_feat.T).item() * 100
        return max(0, sim)


def get_all_image_files(folder_path):
    if not os.path.exists(folder_path):
        return []
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp', '.JPEG')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
    return sorted(files)


def load_target_image_by_filename(filename, target_images_path, transform, device):
    filepath = os.path.join(target_images_path, filename)
    if os.path.exists(filepath):
        try:
            img = Image.open(filepath).convert('RGB')
            return transform(img).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    return None


def load_data(clean_images_path, clean_captions_path, target_captions_path):
    with open(clean_captions_path, 'r', encoding='utf-8') as f:
        clean_captions = [line.strip() for line in f.readlines()]
    with open(target_captions_path, 'r', encoding='utf-8') as f:
        target_captions = [line.strip() for line in f.readlines()]
    image_files = sorted([f for f in os.listdir(clean_images_path)
                         if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.jpeg', '.JPEG'))])
    return image_files, clean_captions, target_captions


def main():
    # Paths
    clean_images_path = r"D:\dmta_data\clean_images"
    clean_captions_path = r"D:\dmta_data\clean_captions.txt"
    target_captions_path = r"D:\dmta_data\target_captions.txt"
    target_images_path = r"D:\dmta_data\target_images"
    output_dir = r"D:\adversarial_images_anyattack_instructblip"

    os.makedirs(output_dir, exist_ok=True)

    # Initialize with CORRECTED ModelScope-compatible settings
    attack = AnyAttackInspiredTripletShift(
        epsilon=16/255,
        gamma_step=2.0/255,
        K=300,
        alpha_dc=0.3,
        beta_dc=3.0,
        lambda_p=1.0,
        mu=0.0001,
        decoder_lr=0.001,
        use_pretrained_decoder=False,  # Set True if you have pre-trained weights
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    evaluator = EvaluationMetrics(device=attack.device)

    # Load data
    image_files, clean_captions, target_captions = load_data(
        clean_images_path, clean_captions_path, target_captions_path
    )

    target_image_files = get_all_image_files(target_images_path)
    num_targets = len(target_image_files)
    print(f"\nFound {num_targets} target images")

    num_samples = min(num_targets if num_targets > 0 else len(image_files), 
                     len(image_files), len(clean_captions), len(target_captions))
    print(f"Processing {num_samples} samples\n")

    image_files = image_files[:num_samples]
    clean_captions = clean_captions[:num_samples]
    target_captions = target_captions[:num_samples]
    target_image_files = target_image_files[:num_samples] if num_targets > 0 else [None] * num_samples

    # Storage
    all_gen_captions = []
    all_tgt_captions = []

    results = {
        'bleu1': [], 'bleu4': [], 'meteor': [], 'rouge_l': [],
        'cider': [], 'spice': [], 'clip_score': [], 'lpips': [],
        'sim_target': [], 'sim_source': [], 'sim_gap': []
    }

    for idx, (img_file, clean_cap, tgt_cap, tgt_img_file) in enumerate(
        tqdm(zip(image_files, clean_captions, target_captions, target_image_files),
             total=num_samples, desc="Attacking")):

        try:
            print(f"\n{'='*70}")
            print(f"[{idx+1}/{num_samples}] {img_file}")
            print(f"Target: {tgt_cap[:60]}...")

            # Load images
            img_path = os.path.join(clean_images_path, img_file)
            clean_pil = Image.open(img_path).convert('RGB')
            clean_img = attack.clip_transform(clean_pil).unsqueeze(0).to(attack.device)

            tgt_img_tensor = None
            if tgt_img_file is not None:
                tgt_img_tensor = load_target_image_by_filename(
                    tgt_img_file, target_images_path, attack.clip_transform, attack.device
                )

            # Generate adversarial image
            print("Generating with AnyAttack-Inspired TripletShift...")
            adv_img = attack.generate_adversarial_image(clean_img, tgt_cap, tgt_img_tensor)

            # Get similarities
            with torch.no_grad():
                adv_emb = attack.get_image_embedding(adv_img)
                src_emb = attack.get_image_embedding(clean_img)

                if tgt_img_tensor is not None:
                    tgt_emb = attack.get_image_embedding(tgt_img_tensor)
                else:
                    tgt_emb = attack.get_text_embedding(tgt_cap)

                sim_tgt = (adv_emb * tgt_emb).sum().item()
                sim_src = (adv_emb * src_emb).sum().item()
                gap = sim_tgt - sim_src

            # Generate caption with InstructBLIP
            gen_cap = attack.generate_caption(adv_img)

            # Store
            all_gen_captions.append(gen_cap)
            all_tgt_captions.append(tgt_cap)

            # Compute metrics
            bleu1, bleu4 = evaluator.compute_bleu(tgt_cap, gen_cap)
            meteor = evaluator.compute_meteor(tgt_cap, gen_cap)
            rouge_l = evaluator.compute_rouge_l(tgt_cap, gen_cap)
            clip_score = evaluator.compute_clip_score(
                adv_img, tgt_cap, attack.clip_model, attack.device
            )
            lpips_score = attack.compute_lpips(adv_img, clean_img).item()

            # Store
            results['bleu1'].append(bleu1)
            results['bleu4'].append(bleu4)
            results['meteor'].append(meteor)
            results['rouge_l'].append(rouge_l)
            results['clip_score'].append(clip_score)
            results['lpips'].append(lpips_score)
            results['sim_target'].append(sim_tgt)
            results['sim_source'].append(sim_src)
            results['sim_gap'].append(gap)

            # Save image
            adv_denorm = attack.unnormalize(adv_img)
            adv_denorm = torch.clamp(adv_denorm, 0, 1)
            adv_pil = transforms.ToPILImage()(adv_denorm.squeeze().cpu())
            adv_pil.save(os.path.join(output_dir, f"adv_{idx:05d}_{img_file}"))

            # Print
            print(f"\nTARGET:    {tgt_cap[:65]}...")
            print(f"GENERATED: {gen_cap[:65]}...")
            print(f"BLEU-1: {bleu1*100:.2f}% | BLEU-4: {bleu4*100:.2f}% | METEOR: {meteor*100:.2f}%")
            print(f"ROUGE-L: {rouge_l*100:.2f}% | CLIP: {clip_score:.2f} | LPIPS: {lpips_score:.4f}")
            print(f"Sim: Tgt={sim_tgt:.4f} Src={sim_src:.4f} Gap={gap:+.4f}")

            if gap > 0:
                print("✓ SUCCESS: Closer to target than source")
            if sim_tgt > 0.7:
                print("✓✓ EXCELLENT: High target similarity")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compute corpus-level metrics
    if all_gen_captions:
        print("\nComputing CIDEr and SPICE...")
        cider = evaluator.compute_cider(all_tgt_captions, all_gen_captions)
        spice = evaluator.compute_spice(all_tgt_captions, all_gen_captions)

        results['cider'] = [cider] * len(results['bleu1'])
        results['spice'] = [spice] * len(results['bleu1'])

    # Final results
    if results['bleu1']:
        print("\n" + "="*70)
        print("ANYATTACK-INSPIRED TripletShift with InstructBLIP-Vicuna - FINAL RESULTS")
        print("="*70)

        avgs = {}
        for m in ['bleu1', 'bleu4', 'meteor', 'rouge_l', 'cider', 'spice', 'clip_score', 'lpips']:
            vals = [v for v in results[m] if isinstance(v, (int, float)) and not (np.isnan(v) or np.isinf(v))]
            avgs[m] = np.mean(vals) if vals else 0.0

        print(f"BLEU-1:  {avgs['bleu1']*100:.2f}%")
        print(f"BLEU-4:  {avgs['bleu4']*100:.2f}%")
        print(f"METEOR:  {avgs['meteor']*100:.2f}%")
        print(f"ROUGE-L: {avgs['rouge_l']*100:.2f}%")
        print(f"CIDEr:   {avgs['cider']:.2f}")
        print(f"SPICE:   {avgs['spice']:.2f}")
        print(f"CLIP:    {avgs['clip_score']:.2f}")
        print(f"LPIPS:   {avgs['lpips']:.4f}")

        success = sum(1 for g in results['sim_gap'] if g > 0)
        excellent = sum(1 for t in results['sim_target'] if t > 0.7)
        print(f"\nSuccess Rate (gap>0): {success}/{num_samples} ({success/num_samples*100:.1f}%)")
        print(f"Excellent Rate (sim>0.7): {excellent}/{num_samples} ({excellent/num_samples*100:.1f}%)")

        with open(os.path.join(output_dir, "results_instructblip_vicuna.json"), 'w') as f:
            json.dump({'average': avgs, 'detailed': results}, f, indent=2)

        print(f"\n✓ Results saved to {output_dir}")


if __name__ == "__main__":
    main()
