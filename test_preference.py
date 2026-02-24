import os
import json
from math import ceil
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from accelerate import Accelerator

# -----------------------------
# Scorers
# -----------------------------
import ImageReward as RM
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime


def _write_eval_txt(result_path: str, payload: dict, append: bool = True):
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    mode = "a" if append else "w"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append(f"[{ts}] SD3.5 preference eval")
    lines.append(f"model_ckpt_dir: {payload.get('model_ckpt_dir')}")
    lines.append(f"prompts_count: {payload.get('count')}")
    lines.append(f"num_inference_steps: {payload.get('num_inference_steps')}")
    lines.append(f"guidance_scale: {payload.get('guidance_scale')}")
    lines.append(f"pickscore_avg: {payload.get('pickscore')}")
    lines.append(f"imagereward_avg: {payload.get('imagereward')}")
    lines.append("json: " + json.dumps(payload, ensure_ascii=False))
    lines.append("")

    with open(result_path, mode, encoding="utf-8") as f:
        f.write("\n".join(lines))



class PickScoreScorer(torch.nn.Module):
    """
    PickScore scorer:
      - input: prompts(list[str]), images(list[PIL.Image])
      - output: tensor[B] (float)
    """
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        processor_path: str = "/home/notebook/data/group/wth/ddirl/weights/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_path: str = "/home/notebook/data/group/wth/ddirl/weights/PickScore_v1",
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(device=device, dtype=dtype)
        self.model.requires_grad_(False)

    @torch.no_grad()
    def forward(self, prompts: List[str], images: List[Image.Image]) -> torch.Tensor:
        image_inputs = self.processor(
            images=images, padding=True, truncation=True, max_length=77, return_tensors="pt"
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}

        text_inputs = self.processor(
            text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt"
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}

        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs @ image_embs.T)
        scores = scores.diag()

        # norm to ~[0,1]
        scores = scores / 26
        # scores = scores
        return scores


class ImageRewardScorer(torch.nn.Module):
    """
    ImageReward scorer:
      - input: prompts(list[str]), images(list[PIL.Image])
      - output: tensor[B] (float)
    """
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        model_path: str = "/home/notebook/data/group/wth/ddirl/weights/ImageReward/ImageReward.pt",
        med_config: str = "/home/notebook/data/group/wth/ddirl/weights/ImageReward/med_config.json",
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model = RM.load(model_path, device=device, med_config=med_config).eval().to(dtype=dtype)
        self.model.requires_grad_(False)

    @torch.no_grad()
    def forward(self, prompts: List[str], images: List[Image.Image]) -> torch.Tensor:
        assert len(prompts) == len(images), "number of prompts and images are not consistency"
        rewards: List[float] = []
        for p, img in zip(prompts, images):
            _, reward = self.model.inference_rank(p, [img])
            rewards.append(float(reward))
        return torch.tensor(rewards, device=self.device, dtype=self.dtype)


# -----------------------------
# Sampling helper (FlowMatch Euler)
# -----------------------------
@torch.no_grad()
def _sample_images_sd35(
    pipe: StableDiffusion3Pipeline,
    prompts: List[str],
    device: torch.device,
    resolution: int,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int = 42,
) -> List[Image.Image]:
    do_cfg = guidance_scale > 1.0
    dtype = pipe.transformer.dtype

    emb, neg_emb, pooled, neg_pooled = pipe.encode_prompt(
        prompt=prompts,
        prompt_2=prompts,
        prompt_3=prompts,
        negative_prompt="",
        negative_prompt_2="",
        negative_prompt_3="",
        do_classifier_free_guidance=do_cfg,
        device=device,
        max_sequence_length=256,
    )

    if do_cfg:
        emb = torch.cat([neg_emb, emb], dim=0)
        pooled = torch.cat([neg_pooled, pooled], dim=0)

    num_channels_latents = pipe.transformer.config.in_channels
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = pipe.prepare_latents(
        len(prompts),
        num_channels_latents,
        resolution,
        resolution,
        dtype,
        device,
        generator,
    )

    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    x = latents
    for t in timesteps:
        if do_cfg:
            latent_in = torch.cat([x] * 2, dim=0)
            t_b = t.expand(latent_in.shape[0])

            v_all = pipe.transformer(
                hidden_states=latent_in,
                timestep=t_b,
                encoder_hidden_states=emb,
                pooled_projections=pooled,
                return_dict=False,
            )[0]
            v_u, v_c = v_all.chunk(2)
            v_hat = v_u + guidance_scale * (v_c - v_u)
        else:
            t_b = t.expand(x.shape[0])
            v_hat = pipe.transformer(
                hidden_states=x,
                timestep=t_b,
                encoder_hidden_states=emb,
                pooled_projections=pooled,
                return_dict=False,
            )[0]

        x = scheduler.step(v_hat, t, x, return_dict=False)[0]

    # decode
    latents_dec = x / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    imgs_latent = pipe.vae.decode(latents_dec.to(pipe.vae.dtype), return_dict=False)[0]
    images = pipe.image_processor.postprocess(imgs_latent, output_type="pil")
    return images


# -----------------------------
# Public API
# -----------------------------
def evaluate_preference_from_checkpoint(
    model_ckpt_dir: str,
    prompts_txt: str,
    output_dir: str,
    accelerator: Accelerator,
    base_model_path: str = "/home/notebook/data/group/wth/ddirl/weights/stabilityai/stable-diffusion-3.5-medium",
    guidance_scale: float = 1.0,
    resolution: int = 1024,
    use_bfloat16: bool = True,
    batch_size: int = 4,
    num_inference_steps: int = 1,
    scorer_dtype: torch.dtype = torch.float32,
    compute_imagereward: bool = True,
    seed: int = 42,
    pick_processor_path: str = "/home/notebook/data/group/wth/ddirl/weights/CLIP-ViT-H-14-laion2B-s32B-b79K",
    pick_model_path: str = "/home/notebook/data/group/wth/ddirl/weights/PickScore_v1",
    ir_model_path: str = "/home/notebook/data/group/wth/ddirl/weights/ImageReward/ImageReward.pt",
    ir_med_config: str = "/home/notebook/data/group/wth/ddirl/weights/ImageReward/med_config.json",
    result_txt: Optional[str] = None,
    append_result: bool = True,
    write_file=False
) -> Dict[str, Optional[float]]:
    """
    return:
      {
        "pickscore": float|None,
        "imagereward": float|None,
        "count": int,
      }
    """
    is_main = accelerator.is_main_process
    device = accelerator.device
    world_size = accelerator.num_processes
    rank = accelerator.process_index

    if not os.path.isfile(prompts_txt):
        if is_main:
            raise FileNotFoundError(f"Prompt are not exist: {prompts_txt}")
        return {"pickscore": None, "imagereward": None, "count": 0}

    with open(prompts_txt, "r", encoding="utf-8") as f:
        prompts_all = [line.rstrip("\n\r") for line in f if line.strip()]

    if len(prompts_all) == 0:
        if is_main:
            raise ValueError(f"Prompt is empty: {prompts_txt}")
        return {"pickscore": None, "imagereward": None, "count": 0}

    # rank0
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # shard
    global_indices = list(range(len(prompts_all)))
    shard_indices = global_indices[rank::world_size]
    shard_prompts = [prompts_all[i] for i in shard_indices]

    if len(shard_prompts) == 0:
        local_pick_sum = torch.zeros(1, device=device)
        local_ir_sum = torch.zeros(1, device=device)
        local_count = torch.zeros(1, device=device)
        accelerator.wait_for_everyone()

        all_pick_sums = accelerator.gather(local_pick_sum)
        all_ir_sums = accelerator.gather(local_ir_sum)
        all_counts = accelerator.gather(local_count)

        if is_main:
            total_count = int(all_counts.sum().item())
            return {"pickscore": None, "imagereward": None, "count": total_count}
        
        return {"pickscore": None, "imagereward": None, "count": 0}

    # load pipeline
    dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    pipe = StableDiffusion3Pipeline.from_pretrained(base_model_path, torch_dtype=dtype)
    ckpt_path = model_ckpt_dir

    if not os.path.isfile(ckpt_path):
        if is_main:
            raise FileNotFoundError(f"Checkpoint not exist: {ckpt_path}")
        return {"pickscore": None, "imagereward": None, "count": 0}

    state_dict = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = pipe.transformer.load_state_dict(state_dict, strict=False)
    if is_main:
        print("[Eval] load_state_dict missing keys:", missing)
        print("[Eval] load_state_dict unexpected keys:", unexpected)

    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # generate & save per-rank
    local_images: List[Image.Image] = []
    local_prompts: List[str] = []

    local_num_batches = ceil(len(shard_prompts) / batch_size)

    with torch.inference_mode():
        for bi in range(local_num_batches):
            start = bi * batch_size
            end = min((bi + 1) * batch_size, len(shard_prompts))
            if start >= end:
                break

            sub_prompts = shard_prompts[start:end]
            cur_indices = shard_indices[start:end]

            images = _sample_images_sd35(
                pipe=pipe,
                prompts=sub_prompts,
                device=device,
                resolution=resolution,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )

            for p, img, gidx in zip(sub_prompts, images, cur_indices):
                save_path = os.path.join(output_dir, f"{gidx}.png")
                img.save(save_path)
                local_images.append(img)
                local_prompts.append(p)

    # scoring
    pick_scorer = PickScoreScorer(
        device=str(device),
        dtype=scorer_dtype,
        processor_path=pick_processor_path,
        model_path=pick_model_path,
    )
    with torch.no_grad():
        pick_scores = pick_scorer(local_prompts, local_images)

    local_pick_sum = pick_scores.sum().unsqueeze(0).to(device)
    local_count = torch.tensor([pick_scores.shape[0]], device=device, dtype=torch.float32)

    if compute_imagereward:
        ir_scorer = ImageRewardScorer(
            device=str(device),
            dtype=scorer_dtype,
            model_path=ir_model_path,
            med_config=ir_med_config,
        )
        with torch.no_grad():
            ir_scores = ir_scorer(local_prompts, local_images)
        local_ir_sum = ir_scores.sum().unsqueeze(0).to(device)
    else:
        local_ir_sum = torch.zeros(1, device=device)

    for img in local_images:
        try:
            img.close()
        except Exception:
            pass

    accelerator.wait_for_everyone()

    all_pick_sums = accelerator.gather(local_pick_sum)
    all_counts = accelerator.gather(local_count)
    all_ir_sums = accelerator.gather(local_ir_sum)

    avg_ir = None
    avg_pick = None

    if is_main:
        total_count = float(all_counts.sum().item())
        if total_count <= 0:
            return {"pickscore": None, "imagereward": None, "count": 0}

        avg_pick = float(all_pick_sums.sum().item()) / total_count
        avg_ir = None
        if compute_imagereward:
            avg_ir = float(all_ir_sums.sum().item()) / total_count

        print(f"[Eval] model_ckpt_dir={model_ckpt_dir}")
        print(f"[Eval] prompts={int(total_count)} steps={num_inference_steps} cfg={guidance_scale}")
        print(f"[Eval] PickScore Average: {avg_pick:.6f}")
        if compute_imagereward and avg_ir is not None:
            print(f"[Eval] ImageReward Average: {avg_ir:.6f}")

        res = {"pickscore": avg_pick, "imagereward": avg_ir, "count": int(total_count)}

        if write_file:
            payload = dict(
                model_ckpt_dir=model_ckpt_dir,
                prompts_txt=prompts_txt,
                output_dir=output_dir,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                resolution=resolution,
                seed=seed,
                **res,
            )
            _write_eval_txt(result_txt, payload, append=append_result)
            print(f"[Eval] wrote result to: {result_txt}")

    return avg_ir, avg_pick


# -----------------------------
# CLI entry
# -----------------------------
def _build_argparser():
    import argparse
    ap = argparse.ArgumentParser("SD3.5 preference eval (PickScore + ImageReward)")

    ap.add_argument("--model_ckpt_dir", type=str, default="") # input the transformer ckpt path
    ap.add_argument("--prompts_txt", type=str, default="") # input your test prompts
    ap.add_argument("--output_dir", type=str, default="") # input the generated images folder
    ap.add_argument("--result_txt", type=str, default="./metric_results/eval_results_dpdmd_sd35.txt", # input your stored path
                    help="write eval result to this txt; default: <output_dir>/eval_result.txt")

    ap.add_argument("--base_model_path", type=str, default="/home/notebook/data/group/wth/ddirl/weights/stabilityai/stable-diffusion-3.5-medium") # change

    ap.add_argument("--guidance_scale", type=float, default=1.0)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_inference_steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no_imagereward", action="store_true", default=False)

    # scorer paths (please change)
    ap.add_argument("--pick_processor_path", type=str, default="/home/notebook/data/group/wth/ddirl/weights/CLIP-ViT-H-14-laion2B-s32B-b79K")
    ap.add_argument("--pick_model_path", type=str, default="/home/notebook/data/group/wth/ddirl/weights/PickScore_v1")
    ap.add_argument("--ir_model_path", type=str, default="/home/notebook/data/group/wth/ddirl/weights/ImageReward/ImageReward.pt")
    ap.add_argument("--ir_med_config", type=str, default="/home/notebook/data/group/wth/ddirl/weights/ImageReward/med_config.json")

    
    ap.add_argument("--append_result", action="store_true", default=True,
                    help="append to result_txt (default True); if set False, overwrite")

    return ap


def main():
    ap = _build_argparser()
    args = ap.parse_args()

    accelerator = Accelerator()
    avg_ir, avg_pick = evaluate_preference_from_checkpoint(
        model_ckpt_dir=args.model_ckpt_dir,
        prompts_txt=args.prompts_txt,
        output_dir=args.output_dir,
        accelerator=accelerator,
        base_model_path=args.base_model_path,
        guidance_scale=args.guidance_scale,
        resolution=args.resolution,
        use_bfloat16=args.bf16,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        compute_imagereward=(not args.no_imagereward),
        seed=args.seed,
        pick_processor_path=args.pick_processor_path,
        pick_model_path=args.pick_model_path,
        ir_model_path=args.ir_model_path,
        ir_med_config=args.ir_med_config,
        result_txt=args.result_txt,
        append_result=args.append_result,
        write_file=False
    )


if __name__ == "__main__":
    main()


# accelerate launch --main_process_port 29512 test_preference.py 
