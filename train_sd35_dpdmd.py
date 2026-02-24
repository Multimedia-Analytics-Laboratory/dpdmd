import os, math, argparse, json, time, random
import io
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging as pylog

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import List
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_

from test_diversity import evaluate_diversity_clip_and_dino_from_checkpoint_distributed
from test_preference import evaluate_preference_from_checkpoint


logging.set_verbosity_error()


class PromptDataset(Dataset):
    def __init__(self, prompt_txt: str):
        self.prompts = []
        pat = re.compile(r"^\s*\d+\s*-+\s*(.*\S)\s*$")

        with io.open(prompt_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue

                m = pat.match(line)
                if m:
                    p = m.group(1).strip()
                else:
                    p = line.strip()

                if p:
                    self.prompts.append(p)

        if len(self.prompts) == 0:
            raise RuntimeError(f"No prompts found in {prompt_txt}")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_model(model, accelerator: Accelerator):
    m = accelerator.unwrap_model(model)
    if hasattr(m, "_orig_mod"):
        return m._orig_mod
    return m


def save_student_transformer(accelerator, student_transformer, outdir, name):
    accelerator.wait_for_everyone()

    fsdp_model = student_transformer
    state_dict = accelerator.get_state_dict(fsdp_model)

    if accelerator.is_main_process:
        os.makedirs(outdir, exist_ok=True)
        ckpt_path = os.path.join(outdir, f"{name}_transformer.pt")
        state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}
        torch.save(state_dict, ckpt_path)
        print(f"[CKPT] saved student transformer state_dict to {ckpt_path}")

    accelerator.wait_for_everyone()


def get_x0_from_v(pred_v, xt, sigma):
    return xt - sigma * pred_v


def compute_dmd_loss(args, accelerator, student_output, fake_transformer,
        pipe_teacher, emb_fake, pooled_fake, emb_teacher, pooled_teacher, device):
    
    original_latents = student_output
    dtype = student_output.dtype

    num_train_timesteps = pipe_teacher.scheduler.config.num_train_timesteps
    sched: FlowMatchEulerDiscreteScheduler = pipe_teacher.scheduler
    sched.set_timesteps(num_train_timesteps, device=device)
    timesteps = sched.timesteps

    idx_dmd = torch.randint(low=0, high=num_train_timesteps, size=(1,), device=device,)
    t_k_dmd = timesteps[idx_dmd]
    idx_t_k_dmd = sched.index_for_timestep(t_k_dmd)
    sigma_t_k_dmd = sched.sigmas[idx_t_k_dmd].to(student_output.dtype)

    noise = torch.randn_like(student_output, dtype=student_output.dtype)
    xt_dmd = (1.0 - sigma_t_k_dmd) * student_output + sigma_t_k_dmd * noise
    xt_dmd = xt_dmd.to(student_output.dtype)

    with torch.no_grad():
        fake_transformer.eval()

        # teacher model
        latent_model_input = torch.cat([xt_dmd] * 2, dim=0)
        t_b = t_k_dmd.expand(latent_model_input.shape[0])
        v_all = pipe_teacher.transformer(
            hidden_states=latent_model_input,
            timestep=t_b,
            encoder_hidden_states=emb_teacher,
            pooled_projections=pooled_teacher,
            return_dict=False,
        )[0]

        v_u, v_c = v_all.chunk(2)
        pred_teacher_v = v_u + args.teacher_guidance_scale * (v_c - v_u)

        # fake model
        latent_model_input = xt_dmd
        t_b = t_k_dmd.expand(latent_model_input.shape[0])
        pred_fake_v = fake_transformer(
            hidden_states=latent_model_input,
            timestep=t_b,
            encoder_hidden_states=emb_fake,
            pooled_projections=pooled_fake,
            return_dict=False,
        )[0]

    pred_fake_image = get_x0_from_v(pred_fake_v, xt_dmd, sigma_t_k_dmd)
    pred_real_image = get_x0_from_v(pred_teacher_v, xt_dmd, sigma_t_k_dmd)

    p_real = xt_dmd - pred_real_image
    p_fake = xt_dmd - pred_fake_image

    denom = torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True)
    grad = (p_real - p_fake) / denom
    grad = torch.nan_to_num(grad)

    dmd_loss = 0.5 * F.mse_loss(original_latents.float(), (original_latents - grad).detach().float(), reduction="mean")
    return dmd_loss


def compute_fake_loss(student_output, noise, fake_transformer, xt_dmd, t_k_dmd, emb_fake, pooled_fake, device):
    fake_transformer.train()
    latent_model_input = xt_dmd
    t_b = t_k_dmd.expand(latent_model_input.shape[0])
    pred_fake_v = fake_transformer(
        hidden_states=latent_model_input,
        timestep=t_b,
        encoder_hidden_states=emb_fake,
        pooled_projections=pooled_fake,
        return_dict=False,
    )[0]

    loss_fake_model = F.mse_loss(pred_fake_v.float(), (noise - student_output).detach().float(), reduction="mean")
    return loss_fake_model


def only_trainable_params(module: nn.Module):
    return [p for p in module.parameters() if p.requires_grad]


def count_params(model: torch.nn.Module):
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def log_trainable_params(accelerator: Accelerator, file_logger, name: str, model: torch.nn.Module):
    m = unwrap_model(model, accelerator)
    total, trainable = count_params(m)
    if accelerator.is_main_process:
        pct = 100.0 * trainable / max(1, total)
        file_logger.info(f"[PARAMS] {name}: trainable={trainable:,} / total={total:,} ({pct:.4f}%)")


# ------------------------- main code -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_id", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--student_id", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--fake_id", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--pick_processor_path", type=str, default="/home/notebook/data/group/wth/ddirl/weights/CLIP-ViT-H-14-laion2B-s32B-b79K")
    ap.add_argument("--pick_model_path", type=str, default="/home/notebook/data/group/wth/ddirl/weights/PickScore_v1")
    ap.add_argument("--ir_model_path", type=str, default="/home/notebook/data/group/wth/ddirl/weights/ImageReward/ImageReward.pt")
    ap.add_argument("--ir_med_config", type=str, default="/home/notebook/data/group/wth/ddirl/weights/ImageReward/med_config.json")
    ap.add_argument("--dino_path", type=str, default="/home/notebook/data/group/wth/ddirl/weights/dinov2-base")
    ap.add_argument("--clip_path", type=str, default="/home/notebook/data/group/wth/ddirl/weights/clip-vit-large-patch14")

    ap.add_argument("--prompt_txt", type=str, default="data/prompts.txt")
    ap.add_argument("--log_path", type=str, default="outputs/sd35_dpdmd/sd35m_t30_1024_lr1e5_4nfe_anchor5/log")
    ap.add_argument("--ckpt_dir", type=str, default="outputs/sd35_dpdmd/sd35m_t30_1024_lr1e5_4nfe_anchor5/ckpts")
    ap.add_argument("--eval_dir", type=str, default="outputs/sd35_dpdmd/sd35m_t30_1024_lr1e5_4nfe_anchor5/eval_images")
    ap.add_argument("--process_folder_name", type=str, default="outputs/sd35_dpdmd/sd35m_t30_1024_lr1e5_4nfe_anchor5/process_vis")
    ap.add_argument("--diversity_folder_name", type=str, default="outputs/sd35_dpdmd/sd35m_t30_1024_lr1e5_4nfe_anchor5/div_vis")

    ap.add_argument("--teacher_infer_steps", type=int, default=30)
    ap.add_argument("--stu_inference_steps", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--teacher_guidance_scale", type=float, default=3.5)
    ap.add_argument("--per_train_fake_num", type=int, default=5)
    ap.add_argument("--k_anchor", type=int, default=5)

    ap.add_argument("--lr_student", type=float, default=1e-5)
    ap.add_argument("--lr_fake", type=float, default=1e-5)
    ap.add_argument("--div_weight", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_epoch", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--ckpt_every", type=int, default=300)
    
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true", default=False)

    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)
    os.makedirs(args.process_folder_name, exist_ok=True)
    os.makedirs(args.diversity_folder_name, exist_ok=True)
    seed_everything(args.seed)

    accelerator = Accelerator(
        mixed_precision="bf16" if args.bf16 else ("fp16" if args.fp16 else "no"),
        project_config=ProjectConfiguration(
            project_dir=args.ckpt_dir,
            logging_dir=args.log_path,
            automatic_checkpoint_naming=True,
        ),
        gradient_accumulation_steps=args.grad_accum,
        log_with="wandb",
    )

    device = accelerator.device
    dtype = torch.bfloat16

    # ---------- log ----------
    log_dir = Path(args.log_path)
    if accelerator.is_main_process:
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.set_verbosity_info()
        file_logger = logging.get_logger("sd35_dpdmd")
        file_logger.handlers = []
        fmt = pylog.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = pylog.FileHandler(log_dir / "log.txt", encoding="utf-8")
        fh.setFormatter(fmt)
        file_logger.addHandler(fh)
        file_logger.setLevel(pylog.INFO)
        file_logger.propagate = False
        file_logger.info("CONFIG\n" + json.dumps(vars(args), indent=2, ensure_ascii=False))
        (log_dir / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))
    else:
        file_logger = logging.get_logger("sd35_dummy")

    run_name = f"sd35_dpdmd_{time.strftime('%Y%m%d_%H%M%S')}"
    accelerator.init_trackers(
        project_name="sd35_dpdmd",
        config=vars(args),
        init_kwargs={
            "wandb": {
                "name": run_name,
                "group": "SD3.5-dpdmd",
                "tags": ["sd3.5", "ode", "distill"],
            }
        },
    )

    # ---------------- data ----------------
    dataset = PromptDataset(args.prompt_txt)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=lambda b: b,
    )

    pipe_teacher = StableDiffusion3Pipeline.from_pretrained(args.teacher_id, torch_dtype=dtype)
    pipe_teacher.to(accelerator.device)
    student_transformer = SD3Transformer2DModel.from_pretrained(f"{args.student_id}/transformer", torch_dtype=dtype)
    fake_transformer = SD3Transformer2DModel.from_pretrained(f"{args.fake_id}/transformer", torch_dtype=dtype)

    student_scheduler: FlowMatchEulerDiscreteScheduler = deepcopy(pipe_teacher.scheduler)
    fake_scheduler: FlowMatchEulerDiscreteScheduler = deepcopy(pipe_teacher.scheduler)
    teacher_scheduler: FlowMatchEulerDiscreteScheduler = pipe_teacher.scheduler

    # freeze teacher
    for p in [pipe_teacher]:
        p.vae.requires_grad_(False)
        p.text_encoder.requires_grad_(False)
        p.text_encoder_2.requires_grad_(False)
        p.text_encoder_3.requires_grad_(False)
        p.transformer.requires_grad_(False)
        p.vae.eval()
        p.text_encoder.eval()
        p.text_encoder_2.eval()
        p.text_encoder_3.eval()
        p.transformer.eval()

    student_transformer.enable_gradient_checkpointing()
    fake_transformer.enable_gradient_checkpointing()
    pipe_teacher._guidance_scale = args.teacher_guidance_scale

    student_transformer.requires_grad_(True)
    fake_transformer.requires_grad_(True)
    student_transformer.train()
    fake_transformer.train()
    
    student_params = only_trainable_params(student_transformer)
    fake_params = only_trainable_params(fake_transformer)

    optimizer_student = torch.optim.AdamW(student_params, lr=args.lr_student, weight_decay=args.weight_decay)
    optimizer_fake = torch.optim.AdamW(fake_params, lr=args.lr_fake, weight_decay=args.weight_decay)

    student_transformer, fake_transformer, optimizer_student, optimizer_fake, dataloader = accelerator.prepare(
        student_transformer, fake_transformer, optimizer_student, optimizer_fake, dataloader
    )

    log_trainable_params(accelerator, file_logger, "student_transformer", student_transformer)
    log_trainable_params(accelerator, file_logger, "fake_score_transformer", fake_transformer)
    log_trainable_params(accelerator, file_logger, "teacher_transformer", pipe_teacher.transformer)

    print("opt_student #params:", sum(p.numel() for g in optimizer_student.param_groups for p in g["params"]))
    print("opt_fake #params:", sum(p.numel() for g in optimizer_fake.param_groups for p in g["params"]))
    print("trainable params student:", sum(p.numel() for p in student_transformer.parameters() if p.requires_grad))
    
    # ---------------- training loop ----------------
    avg = {
        "total_loss": 0.0,
        "dmd_loss": 0.0,
        "div_loss": 0.0,
        "fake_loss": 0.0,
    }
    start_time = time.time()
    iteration = 0
    global_epoch = 0

    best_pickscore = 0.0
    best_imgr = -5
    best_clip = 0.0
    best_dino = 0.0
    best_step = 0.0

    def encode_prompts_for(pipe, prompts: List[str], use_cfg=True):
        with torch.no_grad():
            emb, neg_emb, pooled, neg_pooled = pipe.encode_prompt(
                prompt=prompts,
                prompt_2=prompts,
                prompt_3=prompts,
                negative_prompt="",
                negative_prompt_2="",
                negative_prompt_3="",
                do_classifier_free_guidance=use_cfg,
                device=device,
                max_sequence_length=256,
            )
        
        if use_cfg:
            emb = torch.cat([neg_emb, emb], dim=0)
            pooled = torch.cat([neg_pooled, pooled], dim=0)
        return emb, pooled


    while global_epoch < args.max_epoch:
        data_iter = tqdm(
            dataloader,
            total=len(dataloader),
            disable=not accelerator.is_main_process,
            dynamic_ncols=True,
            leave=False,
        )

        for prompts in data_iter:
            B = len(prompts)

            emb_student, pooled_student = encode_prompts_for(pipe_teacher, prompts, use_cfg=False)
            emb_fake, pooled_fake = encode_prompts_for(pipe_teacher, prompts, use_cfg=False)
            emb_teacher, pooled_teacher = encode_prompts_for(pipe_teacher, prompts, use_cfg=True)
            
            num_channels_latents = (unwrap_model(pipe_teacher.transformer, accelerator).config.in_channels)
            teacher_scheduler.set_timesteps(args.teacher_infer_steps, device=device)
            t_teacher = teacher_scheduler.timesteps
            T = len(t_teacher)

            train_student = ((iteration + 1) % (args.per_train_fake_num + 1) == 0)
            train_fake = not train_student

            dmd_loss_val = 0.0
            div_loss_val = 0.0
            fake_loss_val = 0.0
            total_loss_val = 0.0
            
            g_init = torch.Generator(device=device).manual_seed(args.seed + iteration)
            latents0 = pipe_teacher.prepare_latents(
                B,
                num_channels_latents,
                args.height,
                args.width,
                emb_student.dtype,
                device,
                g_init,
            )

            # initial noise
            x1 = latents0.detach()

            # ====== student branch ======
            if train_student:
                with torch.no_grad():
                    x = latents0
                    x_traj = [x]

                    for i in range(args.k_anchor):
                        t_i = t_teacher[i]
                        latent_model_input = torch.cat([x] * 2, dim=0)
                        t_b = t_i.expand(latent_model_input.shape[0])
                        v_all = pipe_teacher.transformer(
                            hidden_states=latent_model_input,
                            timestep=t_b,
                            encoder_hidden_states=emb_teacher,
                            pooled_projections=pooled_teacher,
                            return_dict=False,
                        )[0]
                        v_u, v_c = v_all.chunk(2)
                        v = v_u + args.teacher_guidance_scale * (v_c - v_u)
                        x = teacher_scheduler.step(v, t_i, x, return_dict=False)[0]

                        if i == args.k_anchor - 1:
                            z_tk = x.detach()
                            if i == args.teacher_infer_steps - 1:
                                t_k = 0
                            else:
                                t_k = t_teacher[i + 1]        
                        
                        x_traj.append(x)

                    x1 = x_traj[0].detach()
                    x0 = x_traj[-1].detach()

                    if t_k != 0:
                        idx_tk = teacher_scheduler.index_for_timestep(t_k)
                        t_k_cont = teacher_scheduler.sigmas[idx_tk].to(dtype=x1.dtype)
                    else:
                        t_k_cont = 0

                with accelerator.accumulate(student_transformer):
                    optimizer_student.zero_grad(set_to_none=True)
                    student_scheduler.set_timesteps(args.stu_inference_steps, device=device)
                    stu_timesteps = student_scheduler.timesteps
                    x_stu = x1
                    
                    v_first = None
                    for step_idx, t_i in enumerate(stu_timesteps):
                        latent_model_input = x_stu
                        t_b = t_i.expand(latent_model_input.shape[0])
                        v_step = student_transformer(
                            hidden_states=latent_model_input,
                            timestep=t_b,
                            encoder_hidden_states=emb_student,
                            pooled_projections=pooled_student,
                            return_dict=False,
                        )[0]

                        x_stu = student_scheduler.step(v_step, t_i, x_stu, return_dict=False)[0]

                        if step_idx == 0:
                            v_first = v_step

                            # detach, important
                            x_stu = x_stu.detach()

                    student_output = x_stu.to(dtype)  # [B,C,H,W]
                    v_target = (x1 - z_tk) / (1.0 - t_k_cont)
                    div_loss = F.mse_loss(v_first.float(), v_target.detach().float())

                    # visualize generated images
                    if accelerator.is_main_process:
                        xT_stu = student_output.detach().to(dtype)
                        for i_img, _ in enumerate(xT_stu):
                            latents_img = (
                                xT_stu[i_img].unsqueeze(0)
                                / pipe_teacher.vae.config.scaling_factor
                            ) + pipe_teacher.vae.config.shift_factor

                            latents_img = latents_img.to(dtype=pipe_teacher.vae.dtype)
                            image = pipe_teacher.vae.decode(latents_img, return_dict=False)[0]
                            image = pipe_teacher.image_processor.postprocess(image, output_type="pil")[0]
                            image_save_path = os.path.join(args.process_folder_name, f"stu_{i_img}.png")
                            image.save(image_save_path)

                    dmd_loss = compute_dmd_loss(
                        args,
                        accelerator,
                        student_output,
                        fake_transformer,
                        pipe_teacher,
                        emb_fake,
                        pooled_fake,
                        emb_teacher,
                        pooled_teacher,
                        device
                    )

                    loss_total = args.div_weight * div_loss + dmd_loss
                    accelerator.backward(loss_total)
                    if accelerator.sync_gradients:
                        gn_student = clip_grad_norm_(only_trainable_params(student_transformer), 1.0)
                        accelerator.print(f"[DEBUG] student grad norm = {gn_student}")

                    optimizer_student.step()
                    optimizer_student.zero_grad(set_to_none=True)

                    dmd_loss_val = float(dmd_loss.detach())
                    div_loss_val = float(div_loss.detach())
                    total_loss_val = dmd_loss_val + args.div_weight * div_loss_val

            # ====== fake branch ======
            else:
                with accelerator.accumulate(fake_transformer):
                    optimizer_fake.zero_grad(set_to_none=True)

                    with torch.no_grad():
                        student_scheduler.set_timesteps(args.stu_inference_steps, device=device)
                        stu_timesteps = student_scheduler.timesteps
                        x_stu = x1

                        for t_i in stu_timesteps:
                            latent_model_input = x_stu
                            t_b = t_i.expand(latent_model_input.shape[0])
                            v_step = student_transformer(
                                hidden_states=latent_model_input,
                                timestep=t_b,
                                encoder_hidden_states=emb_student,
                                pooled_projections=pooled_student,
                                return_dict=False,
                            )[0]

                            v_step = v_step.to(torch.bfloat16)
                            x_stu = student_scheduler.step(v_step, t_i, x_stu, return_dict=False)[0]

                        student_output = x_stu.to(dtype)

                    num_train_timesteps = (pipe_teacher.scheduler.config.num_train_timesteps)
                    sched: FlowMatchEulerDiscreteScheduler = (pipe_teacher.scheduler)
                    sched.set_timesteps(num_train_timesteps, device=device)
                    timesteps = sched.timesteps

                    idx_dmd = torch.randint(low=0, high=num_train_timesteps, size=(1,), device=device)
                    t_k_dmd = timesteps[idx_dmd]
                    idx_t_k_dmd = sched.index_for_timestep(t_k_dmd)
                    sigma_t_k_dmd = sched.sigmas[idx_t_k_dmd].to(student_output.dtype)

                    noise = torch.randn_like(student_output, dtype=student_output.dtype)
                    xt_dmd = (1.0 - sigma_t_k_dmd) * student_output + sigma_t_k_dmd * noise
                    xt_dmd = xt_dmd.to(student_output.dtype)

                    loss_fake_model = compute_fake_loss(
                        student_output,
                        noise,
                        fake_transformer,
                        xt_dmd,
                        t_k_dmd,
                        emb_fake,
                        pooled_fake,
                        device,
                    )

                    accelerator.backward(loss_fake_model)
                    if accelerator.sync_gradients:
                        gn_fake = clip_grad_norm_(only_trainable_params(fake_transformer), 1.0)
                        accelerator.print(f"[DEBUG] fake grad norm = {gn_fake}")

                    optimizer_fake.step()
                    optimizer_fake.zero_grad(set_to_none=True)

                    fake_loss_val = float(loss_fake_model.detach())
                    total_loss_val = fake_loss_val

            # ---------------- log ----------------
            avg["total_loss"] = total_loss_val
            avg["dmd_loss"] = dmd_loss_val
            avg["fake_loss"] = fake_loss_val
            avg["div_loss"] = div_loss_val

            if accelerator.is_main_process:
                data_iter.set_postfix(
                    {
                        "total": f"{avg['total_loss']:.4f}",
                        "dmd": f"{avg['dmd_loss']:.4f}",
                        "div": f"{avg['div_loss']:.4f}",
                        "fake": f"{avg['fake_loss']:.4f}"
                    }
                )
                log_dict = {
                    "loss/total": avg["total_loss"],
                    "loss/dmd": avg["dmd_loss"],
                    "loss/div": avg["div_loss"],
                    "loss/fake": avg["fake_loss"],
                }
                accelerator.log(log_dict, step=iteration)

                if iteration % args.log_every == 0:
                    elapsed = time.time() - start_time
                    line = (
                        f"[{iteration:06d}] "
                        f"total={avg['total_loss']:.4f} "
                        f"dmd={avg['dmd_loss']:.4f} "
                        f"div={avg['div_loss']:.4f} "
                        f"fake={avg['fake_loss']:.4f} "
                        f"{elapsed / max(1, args.log_every):.2f}s/it"
                    )
                    file_logger.info(line)
                    start_time = time.time()

            iteration += 1

            # ====== checkpoint & eval ======
            if iteration % args.ckpt_every == 0:
                ckpt_dir = os.path.join(args.ckpt_dir, f"ckpt_step-{iteration}")
                save_student_transformer(accelerator, student_transformer, ckpt_dir, "student")
                accelerator.wait_for_everyone()

                # test diversity on pick2pic
                _, div_clip_pic, _, div_dino_pic = evaluate_diversity_clip_and_dino_from_checkpoint_distributed(
                    base_model_skpt_dir=args.student_id,
                    model_ckpt_dir=f"{ckpt_dir}/student_transformer.pt",
                    txt_path="data/pick2pic_sfw_test.txt",
                    output_dir=f"{args.diversity_folder_name}/pick2pic",
                    accelerator=accelerator,
                    guidance_scale=1.0,
                    resolution=1024,
                    use_bfloat16=True,
                    num_inference_steps=4,
                    num_seeds=9,
                    seed_start=0,
                    clip_name=args.clip_path,
                    clip_batch_size=32,
                    dino_name=args.dino_path,
                    dino_batch_size=32,
                    scorer_dtype=torch.float32,
                    save_grids=True,
                    grid_resize_to=384
                )

                # test diversity on coco
                _, div_clip_coco, _, div_dino_coco = evaluate_diversity_clip_and_dino_from_checkpoint_distributed(
                    base_model_skpt_dir=args.student_id,
                    model_ckpt_dir=f"{ckpt_dir}/student_transformer.pt",
                    txt_path="data/training_div_test_coco.txt",
                    output_dir=f"{args.diversity_folder_name}/coco",
                    accelerator=accelerator,
                    guidance_scale=1.0,
                    resolution=1024,
                    use_bfloat16=True,
                    num_inference_steps=4,
                    num_seeds=9,
                    seed_start=0,
                    clip_name=args.clip_path,
                    clip_batch_size=32,
                    dino_name=args.dino_path,
                    dino_batch_size=32,
                    scorer_dtype=torch.float32,
                    save_grids=True,
                    grid_resize_to=384
                )

                # test preference on pick2pic
                imgr_pic, pickscore_pic = evaluate_preference_from_checkpoint(
                    model_ckpt_dir=f"{ckpt_dir}/student_transformer.pt",
                    prompts_txt="data/pick2pic_sfw_test.txt",
                    output_dir=f"{args.eval_dir}/pick2pic",
                    accelerator=accelerator,
                    base_model_path=args.student_id,
                    guidance_scale=1.0,
                    resolution=1024,
                    use_bfloat16=True,
                    batch_size=4,
                    num_inference_steps=4,
                    compute_imagereward=True,
                    seed=42,
                    pick_processor_path=args.pick_processor_path,
                    pick_model_path=args.pick_model_path,
                    ir_model_path=args.ir_model_path,
                    ir_med_config=args.ir_med_config,
                    result_txt=None,
                    append_result=True,
                    write_file=False
                )

                # test preference on coco
                imgr_coco, pickscore_coco = evaluate_preference_from_checkpoint(
                    model_ckpt_dir=f"{ckpt_dir}/student_transformer.pt",
                    prompts_txt="data/training_div_test_coco.txt",
                    output_dir=f"{args.eval_dir}/coco",
                    accelerator=accelerator,
                    base_model_path=args.student_id,
                    guidance_scale=1.0,
                    resolution=1024,
                    use_bfloat16=True,
                    batch_size=4,
                    num_inference_steps=4,
                    compute_imagereward=True,
                    seed=42,
                    pick_processor_path=args.pick_processor_path,
                    pick_model_path=args.pick_model_path,
                    ir_model_path=args.ir_model_path,
                    ir_med_config=args.ir_med_config,
                    result_txt=None,
                    append_result=True,
                    write_file=False
                )

                if accelerator.is_main_process:
                    file_logger.info(
                        f"[EVAL-DIST] step={iteration} | "
                        f"Pic PicS={pickscore_pic:.6f} | "
                        f"Pic ImgR={imgr_pic:.6f} | "
                        f"COCO PicS={pickscore_coco:.6f} | "
                        f"COCO ImgR={imgr_coco:.6f} | "
                        f"Pic Div CLIP={div_clip_pic:.6f} | "
                        f"Pic Div DINO={div_dino_pic:.6f} | "
                        f"COCO Div CLIP={div_clip_coco:.6f} | "
                        f"COCO Div DINO={div_dino_coco:.6f}"
                    )
                    accelerator.log(
                        {
                            "eval/pic_preference_pickscore": pickscore_pic,
                            "eval/pic_preference_imagereward": imgr_pic,
                            "eval/coco_preference_pickscore": pickscore_coco,
                            "eval/coco_preference_imagereward": imgr_coco,
                            "eval/pic_div_clip": div_clip_pic,
                            "eval/pic_div_dino": div_dino_pic,
                            "eval/coco_div_clip": div_clip_coco,
                            "eval/coco_div_dino": div_dino_coco,
                        },
                        step=iteration,
                    )

                    cur_pickscore = (pickscore_pic + pickscore_coco) / 2
                    cur_imgr = (imgr_pic + imgr_coco) / 2
                    cur_clip = (div_clip_pic + div_clip_coco) / 2
                    cur_dino = (div_dino_pic + div_dino_coco) / 2

                    if cur_pickscore > best_pickscore and cur_imgr > best_imgr:
                        best_pickscore = cur_pickscore
                        best_imgr = cur_imgr
                        best_clip = cur_clip
                        best_dino = cur_dino
                        best_step = iteration

            accelerator.wait_for_everyone()
        
        global_epoch += 1

    ckpt_dir = os.path.join(args.ckpt_dir, "ckpt_latest")
    save_student_transformer(accelerator, student_transformer, ckpt_dir, "student")

    accelerator.save_state()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("Training done...")
        accelerator.print("[TRAIN DONE]")
        file_logger.info("[TRAIN DONE]")
        file_logger.info(
            f"[BEST EVAL] iter={best_step} | "
            f"[BEST ITERATION] step={best_step} | "
            f"best PickScore={best_pickscore:.6f} | "
            f"best ImageReward={best_imgr:.6f} | "
            f"best CLIP={best_clip:.6f} | "
            f"best DINO={best_dino:.6f} "
        )

    for tracker in getattr(accelerator, "trackers", []):
        try:
            tracker.finish()
        except Exception:
            pass
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
