import os
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import StableDiffusion3Pipeline
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoModel,
    AutoImageProcessor,
)
from accelerate import Accelerator


# ------------------------------
#   CLIP embedding utils (UNCHANGED)
# ------------------------------
@torch.no_grad()
def clip_cls_embeddings(
    images: List[Image.Image],
    vision_model: CLIPVisionModel,
    processor: CLIPImageProcessor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Return normalized CLS embeddings: [B, D]
    """
    embs = []
    n = len(images)
    for i in range(0, n, batch_size):
        batch_imgs = images[i : i + batch_size]
        inputs = processor(images=batch_imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=vision_model.dtype)

        out = vision_model(pixel_values=pixel_values, output_hidden_states=False)
        # CLS token
        cls = out.last_hidden_state[:, 0, :].to(dtype=dtype)
        cls = F.normalize(cls, dim=-1)
        embs.append(cls)

    return torch.cat(embs, dim=0)


def mean_pairwise_cosine(embs_norm: torch.Tensor) -> torch.Tensor:
    """
    embs_norm: [N, D], already L2-normalized
    Return mean cosine over all pairs (i<j), scalar tensor.
    """
    n = embs_norm.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=embs_norm.device, dtype=embs_norm.dtype)

    sim = embs_norm @ embs_norm.T  # [N, N]
    tri = torch.triu(sim, diagonal=1)
    denom = n * (n - 1) / 2
    return tri.sum() / denom


# ------------------------------
#   DINOv3 embedding utils (UPDATED)
# ------------------------------
@torch.no_grad()
def dino_embeddings(
    images: List[Image.Image],
    dino_model: torch.nn.Module,
    dino_processor: AutoImageProcessor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Prefer outputs.pooler_output if present: [B, C] (your example: [1,1024]).
    Otherwise fallback to avg pooling over tokens: last_hidden_state.mean(dim=1).

    Returns: normalized embeddings [B, C]
    """
    embs = []
    n = len(images)
    for i in range(0, n, batch_size):
        batch_imgs = images[i : i + batch_size]
        inputs = dino_processor(images=batch_imgs, return_tensors="pt")

        pixel_values = inputs["pixel_values"].to(device=device, dtype=dino_model.dtype)

        out = dino_model(pixel_values=pixel_values, return_dict=True)

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output  # [B, C]
        else:
            # fallback: [B, N, C] -> [B, C]
            feat = out.last_hidden_state.mean(dim=1)

        feat = feat.to(dtype=dtype)
        feat = F.normalize(feat, dim=-1)
        embs.append(feat)

    return torch.cat(embs, dim=0)


# ------------------------------
#   SD3.5 sampling (fast + deterministic)
# ------------------------------
@torch.no_grad()
def generate_n_images_for_prompt(
    pipe: StableDiffusion3Pipeline,
    prompt: str,
    seeds: List[int],
    device: torch.device,
    resolution: int,
    guidance_scale: float,
    num_inference_steps: int,
    gen_batch_size: int = 4,
    max_sequence_length: int = 256,
) -> List[Image.Image]:
    do_cfg = guidance_scale > 1.0
    all_images: List[Image.Image] = []

    if len(seeds) == 0:
        return all_images
    if gen_batch_size <= 0:
        raise ValueError(f"gen_batch_size must be > 0, got {gen_batch_size}")

    enc_bsz = min(gen_batch_size, len(seeds))
    prompts = [prompt] * enc_bsz

    emb, neg_emb, pooled, neg_pooled = pipe.encode_prompt(
        prompt=prompts,
        prompt_2=prompts,
        prompt_3=prompts,
        negative_prompt="",
        negative_prompt_2="",
        negative_prompt_3="",
        do_classifier_free_guidance=do_cfg,
        device=device,
        max_sequence_length=max_sequence_length,
    )

    dtype = pipe.transformer.dtype
    num_channels_latents = pipe.transformer.config.in_channels
    scheduler = pipe.scheduler

    for i in range(0, len(seeds), gen_batch_size):
        batch_seeds = seeds[i : i + gen_batch_size]
        bsz = len(batch_seeds)

        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps

        emb_b = emb[:bsz]
        pooled_b = pooled[:bsz]

        if do_cfg:
            neg_emb_b = neg_emb[:bsz]
            neg_pooled_b = neg_pooled[:bsz]
            emb_cat = torch.cat([neg_emb_b, emb_b], dim=0)
            pooled_cat = torch.cat([neg_pooled_b, pooled_b], dim=0)
        else:
            emb_cat = emb_b
            pooled_cat = pooled_b

        generators = [torch.Generator(device=device).manual_seed(int(s)) for s in batch_seeds]

        latents = pipe.prepare_latents(
            bsz,
            num_channels_latents,
            resolution,
            resolution,
            dtype,
            device,
            generators,
        )

        x = latents
        for t in timesteps:
            if do_cfg:
                latent_in = torch.cat([x] * 2, dim=0)
                t_b = t.expand(latent_in.shape[0])
                v_all = pipe.transformer(
                    hidden_states=latent_in,
                    timestep=t_b,
                    encoder_hidden_states=emb_cat,
                    pooled_projections=pooled_cat,
                    return_dict=False,
                )[0]
                v_u, v_c = v_all.chunk(2)
                v_hat = v_u + guidance_scale * (v_c - v_u)
            else:
                t_b = t.expand(x.shape[0])
                v_hat = pipe.transformer(
                    hidden_states=x,
                    timestep=t_b,
                    encoder_hidden_states=emb_cat,
                    pooled_projections=pooled_cat,
                    return_dict=False,
                )[0]

            x = scheduler.step(v_hat, t, x, return_dict=False)[0]

        latents_dec = x / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        imgs_latent = pipe.vae.decode(latents_dec.to(pipe.vae.dtype), return_dict=False)[0]
        images = pipe.image_processor.postprocess(imgs_latent, output_type="pil")
        all_images.extend(images)

    return all_images


def make_image_grid(images: List[Image.Image], resize_to: int = 384) -> Image.Image:
    import math

    n = len(images)
    if n == 0:
        return Image.new("RGB", (resize_to, resize_to), color=(0, 0, 0))

    grid_size = math.ceil(math.sqrt(n))
    images = [img.resize((resize_to, resize_to), Image.BICUBIC) for img in images]
    grid_w = grid_size * resize_to
    grid_h = grid_size * resize_to
    grid_img = Image.new("RGB", (grid_w, grid_h), color=(0, 0, 0))

    for idx, img in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size
        grid_img.paste(img, (col * resize_to, row * resize_to))
    return grid_img


# ------------------------------
#   Main distributed evaluation API
# ------------------------------
def evaluate_diversity_clip_and_dino_from_checkpoint_distributed(
    base_model_skpt_dir: str,
    model_ckpt_dir: str,
    txt_path: str,
    output_dir: str,
    accelerator,
    guidance_scale: float = 1.0,
    resolution: int = 1024,
    use_bfloat16: bool = True,
    num_inference_steps: int = 4,
    num_seeds: int = 16,
    gen_batch_size: int = 4,
    seed_start: int = 0,
    clip_name: str = "openai/clip-vit-large-patch14",
    clip_batch_size: int = 32,
    dino_name: str = "/home/notebook/data/group/wth/ddirl/weights/dinov3-vitl16-pretrain-lvd1689m",
    dino_batch_size: int = 32,
    scorer_dtype: torch.dtype = torch.float32,
    save_grids: bool = False,
    grid_resize_to: int = 384,
    max_sequence_length: int = 256,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:

    is_main = accelerator.is_main_process
    device = accelerator.device
    world_size = accelerator.num_processes
    rank = accelerator.process_index

    if not os.path.isfile(txt_path):
        if is_main:
            raise FileNotFoundError(f"Prompt 文件不存在: {txt_path}")
        else:
            return None, None, None, None

    with open(txt_path, "r", encoding="utf-8") as f:
        prompts = [line.rstrip("\n\r") for line in f if line.strip()]

    if len(prompts) == 0:
        if is_main:
            raise ValueError(f"Prompt 文件为空: {txt_path}")
        else:
            return None, None, None, None

    if is_main:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    global_indices = list(range(len(prompts)))
    shard_indices = global_indices[rank::world_size]
    shard_prompts = [prompts[i] for i in shard_indices]

    if len(shard_prompts) == 0:
        local_sum_clip = torch.zeros(1, device=device, dtype=torch.float32)
        local_sum_dino = torch.zeros(1, device=device, dtype=torch.float32)
        local_count = torch.zeros(1, device=device, dtype=torch.float32)

        accelerator.wait_for_everyone()
        all_sums_clip = accelerator.gather(local_sum_clip)
        all_sums_dino = accelerator.gather(local_sum_dino)
        all_counts = accelerator.gather(local_count)

        if is_main:
            total_count = float(all_counts.sum().item())
            if total_count == 0:
                return None, None, None, None
            avg_cos_clip = float(all_sums_clip.sum().item()) / total_count
            avg_cos_dino = float(all_sums_dino.sum().item()) / total_count
            return avg_cos_clip, 1.0 - avg_cos_clip, avg_cos_dino, 1.0 - avg_cos_dino
        return None, None, None, None

    dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    pipe = StableDiffusion3Pipeline.from_pretrained(base_model_skpt_dir, torch_dtype=dtype,)

    state_dict = torch.load(model_ckpt_dir, map_location="cpu")
    missing, unexpected = pipe.transformer.load_state_dict(state_dict, strict=False)
    if is_main:
        print("missing keys:", missing)
        print("unexpected keys:", unexpected)

    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # CLIP
    processor = CLIPImageProcessor.from_pretrained(clip_name)
    vision_model = CLIPVisionModel.from_pretrained(clip_name)
    vision_model.to(device)
    vision_model.eval()

    # DINOv3
    dino_processor = AutoImageProcessor.from_pretrained(dino_name)
    dino_model = AutoModel.from_pretrained(dino_name)
    dino_model.to(device)
    dino_model.eval()

    from tqdm import tqdm
    iterator = tqdm(
        range(len(shard_prompts)),
        total=len(shard_prompts),
        disable=not is_main,
        desc=f"Diversity(rank{rank})",
        dynamic_ncols=True,
        leave=False,
    )

    local_sum_clip = torch.zeros(1, device=device, dtype=torch.float32)
    local_sum_dino = torch.zeros(1, device=device, dtype=torch.float32)
    local_count = torch.zeros(1, device=device, dtype=torch.float32)

    seeds = [seed_start + i for i in range(num_seeds)]

    with torch.inference_mode():
        for li in iterator:
            prompt = shard_prompts[li]
            gidx = shard_indices[li]

            images = generate_n_images_for_prompt(
                pipe=pipe,
                prompt=prompt,
                seeds=seeds,
                device=device,
                resolution=resolution,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                gen_batch_size=gen_batch_size,
                max_sequence_length=max_sequence_length,
            )

            if save_grids:
                grid = make_image_grid(images, resize_to=grid_resize_to)
                grid.save(os.path.join(output_dir, f"{gidx}_rank{rank}_grid.png"))
                try:
                    grid.close()
                except Exception:
                    pass

            # --- CLIP (UNCHANGED) ---
            embs_clip = clip_cls_embeddings(
                images=images,
                vision_model=vision_model,
                processor=processor,
                device=device,
                dtype=scorer_dtype,
                batch_size=clip_batch_size,
            )
            mean_cos_clip = mean_pairwise_cosine(embs_clip).to(dtype=torch.float32)
            local_sum_clip += mean_cos_clip

            # --- DINO (UPDATED) ---
            embs_dino = dino_embeddings(
                images=images,
                dino_model=dino_model,
                dino_processor=dino_processor,
                device=device,
                dtype=scorer_dtype,
                batch_size=dino_batch_size,
            )
            mean_cos_dino = mean_pairwise_cosine(embs_dino).to(dtype=torch.float32)
            local_sum_dino += mean_cos_dino

            local_count += 1.0

            for img in images:
                try:
                    img.close()
                except Exception:
                    pass

    accelerator.wait_for_everyone()

    all_sums_clip = accelerator.gather(local_sum_clip)
    all_sums_dino = accelerator.gather(local_sum_dino)
    all_counts = accelerator.gather(local_count)

    avg_cos_clip = None
    diversity_clip = None
    avg_cos_dino = None
    diversity_dino = None

    if is_main:
        total_count = float(all_counts.sum().item())
        if total_count > 0:
            avg_cos_clip = float(all_sums_clip.sum().item()) / total_count
            diversity_clip = 1.0 - avg_cos_clip

            avg_cos_dino = float(all_sums_dino.sum().item()) / total_count
            diversity_dino = 1.0 - avg_cos_dino

            print(f"[Eval-Diversity] model={model_ckpt_dir}")
            print(f"[Eval-Diversity] prompts={int(total_count)} num_seeds={num_seeds} gen_batch_size={gen_batch_size}")
            print(f"[Eval-Diversity][CLIP] Avg Cosine Similarity: {avg_cos_clip:.6f} (higher => less diverse)")
            print(f"[Eval-Diversity][CLIP] Diversity (1-AvgCos):     {diversity_clip:.6f} (higher => more diverse)")
            print(f"[Eval-Diversity][DINO] Avg Cosine Similarity: {avg_cos_dino:.6f} (higher => less diverse)")
            print(f"[Eval-Diversity][DINO] Diversity (1-AvgCos):     {diversity_dino:.6f} (higher => more diverse)")

    return avg_cos_clip, diversity_clip, avg_cos_dino, diversity_dino



if __name__ == "__main__":
    
    accelerator = Accelerator(mixed_precision="bf16")
    avg_clip, div_clip, avg_dino, div_dino = evaluate_diversity_clip_and_dino_from_checkpoint_distributed(
        base_model_skpt_dir="/home/notebook/data/group/wth/ddirl/weights/stabilityai/stable-diffusion-3.5-medium", # change
        model_ckpt_dir="", # input your model skpt path
        txt_path="data/pick2pic_sfw_test.txt", # change
        output_dir="./div_test/dpdmd_sd35", # change
        accelerator=accelerator,
        guidance_scale=1.0,
        resolution=1024,
        use_bfloat16=True,
        num_inference_steps=4,
        num_seeds=16,
        seed_start=0,
        clip_name="/home/notebook/data/group/wth/ddirl/weights/clip-vit-large-patch14", # change
        clip_batch_size=32,
        dino_name="/home/notebook/data/group/wth/ddirl/weights/dinov3-vitl16-pretrain-lvd1689m", # change
        dino_batch_size=32,
        scorer_dtype=torch.float32,
        save_grids=True,
        grid_resize_to=384
    )

# one-line launch:
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29519 --num_processes 4 test_diversity.py

