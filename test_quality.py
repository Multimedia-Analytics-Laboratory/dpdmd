# test_quality.py
# ------------------------------------------------------------
# Evaluate image quality for a folder with:
#   1) VQR1 (TianheWu/VisualQuality-R1-7B, Qwen2.5-VL based)
#   2) MANIQA (given implementation + provided ckpt)
#
# For a folder with N images:
#   - compute N VQR1 scores and average -> vqr1_mean
#   - compute N MANIQA avg-scores and average -> maniqa_mean
#
# Output:
#   - per-image scores saved to a text file
#   - overall means printed
#
# NOTE:
# - VQR1 evaluation logic is kept as in your code (prompt, batching, generation config).
# - MANIQA evaluation logic is kept as in your code (seed reset per image, random crop num_crops,
#   same transforms, same model forward math). We only make the forward() tensor allocation device-safe.
# ------------------------------------------------------------

import os
import re
import random
import argparse
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# -----------------------------
# VQR1 imports (as in your code)
# -----------------------------
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# =============================
# Utils: list images in folder
# =============================
def get_image_paths(folder_path: str) -> List[str]:
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    image_paths: List[str] = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    image_paths.sort()
    return image_paths


# =============================
# VQR1 scoring (keep logic)
# =============================
def score_batch_image_vqr1(
    image_paths: List[str],
    model,
    processor,
    device: torch.device,
    bsz: int = 32,
) -> Dict[str, float]:
    PROMPT = (
        "You are doing the image quality assessment task. Here is the question: "
        "What is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, "
        "rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality."
    )
    QUESTION_TEMPLATE = (
        "{Question} Please only output the final answer with only one score in <answer> </answer> tags."
    )

    messages = []
    for img_path in image_paths:
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=PROMPT)},
                ],
            }
        ]
        messages.append(message)

    all_outputs: List[str] = []
    for i in tqdm(range(0, len(messages), bsz), desc="VQR1 scoring"):
        batch_messages = messages[i : i + bsz]

        text = [
            processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True, add_vision_id=True
            )
            for msg in batch_messages
        ]

        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        generated_ids = model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=512,
            do_sample=True,
            top_k=50,
            top_p=1,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        all_outputs.extend(batch_output_text)

    path_score_dict: Dict[str, float] = {}
    for img_path, model_output in zip(image_paths, all_outputs):
        try:
            model_output_matches = re.findall(r"<answer>(.*?)</answer>", model_output, re.DOTALL)
            model_answer = model_output_matches[-1].strip() if model_output_matches else model_output.strip()
            score = float(re.search(r"\d+(\.\d+)?", model_answer).group())
        except Exception:
            print(f"[VQR1] Meet error with {img_path}, please generate again. Fallback random score used.")
            score = random.randint(1, 5)
        path_score_dict[img_path] = float(score)

    return path_score_dict


def build_vqr1(model_path: str, device: torch.device, use_flash_attn2: bool = True):
    attn_impl = "flash_attention_2" if use_flash_attn2 else None
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    return model, processor


# =============================
# MANIQA (your code, same logic)
# =============================

def five_point_crop(idx, d_img, config):
    new_h = config.crop_size
    new_w = config.crop_size
    b, c, h, w = d_img.shape
    if idx == 0:
        top = 0
        left = 0
    elif idx == 1:
        top = 0
        left = w - new_w
    elif idx == 2:
        top = h - new_h
        left = 0
    elif idx == 3:
        top = h - new_h
        left = w - new_w
    elif idx == 4:
        center_h = h // 2
        center_w = w // 2
        top = center_h - new_h // 2
        left = center_w - new_w // 2
    d_img_org = crop_image(top, left, config.crop_size, img=d_img)
    return d_img_org


def random_crop(d_img, config):
    b, c, h, w = d_img.shape
    top = np.random.randint(0, h - config.crop_size)
    left = np.random.randint(0, w - config.crop_size)
    d_img_org = crop_image(top, left, config.crop_size, img=d_img)
    return d_img_org


def crop_image(top, left, patch_size, img=None):
    tmp_img = img[:, :, top : top + patch_size, left : left + patch_size]
    return tmp_img


class RandCrop(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, sample):
        d_img = sample["d_img_org"]
        d_name = sample["d_name"]

        c, h, w = d_img.shape
        new_h = self.patch_size
        new_w = self.patch_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        ret_d_img = d_img[:, top : top + new_h, left : left + new_w]
        sample = {"d_img_org": ret_d_img, "d_name": d_name}
        return sample


class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        d_img = sample["d_img_org"]
        d_name = sample["d_name"]
        d_img = (d_img - self.mean) / self.var
        sample = {"d_img_org": d_img, "d_name": d_name}
        return sample


class RandHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        d_img = sample["d_img_org"]
        d_name = sample["d_name"]
        prob_lr = np.random.random()
        if prob_lr > 0.5:
            d_img = np.fliplr(d_img).copy()
        sample = {"d_img_org": d_img, "d_name": d_name}
        return sample


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        d_img = sample["d_img_org"]
        d_name = sample["d_name"]
        d_img = torch.from_numpy(d_img).type(torch.FloatTensor)
        sample = {"d_img_org": d_img, "d_name": d_name}
        return sample


import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 dim_mlp=1024.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.dim_mlp = dim_mlp
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=self.dim_mlp, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7,
                 dim_mlp=1024, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                dim_mlp=dim_mlp,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=self.input_resolution[0], w=self.input_resolution[1])
        x = F.relu(self.conv(x))
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


class SwinTransformer(nn.Module):
    def __init__(self, patches_resolution, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 embed_dim=256, drop=0.1, drop_rate=0.0, drop_path_rate=0.1, dropout=0.0, window_size=7,
                 dim_mlp=1024, qkv_bias=True, qk_scale=None, attn_drop_rate=0.0, norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False, scale=0.8, **kwargs):
        super().__init__()
        self.scale = scale
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.dropout = nn.Dropout(p=drop)
        self.num_features = embed_dim
        self.num_layers = len(depths)
        self.patches_resolution = (patches_resolution[0], patches_resolution[1])
        self.downsample = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, stride=2, padding=1)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=self.embed_dim,
                input_resolution=patches_resolution,
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                dim_mlp=dim_mlp,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=dropout,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=downsample,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

    def forward(self, x):
        x = self.dropout(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        for layer in self.layers:
            _x = x
            x = layer(x)
            x = self.scale * x + _x
        x = rearrange(x, "b (h w) c -> b c h w", h=self.patches_resolution[0], w=self.patches_resolution[1])
        return x


import timm
from timm.models.vision_transformer import Block

class TABlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)
        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


class MANIQA(nn.Module):
    def __init__(self, embed_dim=72, num_outputs=1, patch_size=8, drop=0.1,
                 depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                 img_size=224, num_tab=2, scale=0.8, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)

        self.vit = timm.create_model("vit_base_patch8_224", pretrained=False)
        self.save_output = SaveOutput()
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                layer.register_forward_hook(self.save_output)

        self.tablock1 = nn.ModuleList([TABlock(self.input_size ** 2) for _ in range(num_tab)])

        self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale,
        )

        self.tablock2 = nn.ModuleList([TABlock(self.input_size ** 2) for _ in range(num_tab)])

        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.swintransformer2 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim // 2,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale,
        )

        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.ReLU(),
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.Sigmoid(),
        )

    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x

    def forward(self, x):
        _ = self.vit(x)
        x = self.extract_feature(self.save_output)
        self.save_output.outputs.clear()

        # stage 1
        x = rearrange(x, "b (h w) c -> b c (h w)", h=self.input_size, w=self.input_size)
        for tab in self.tablock1:
            x = tab(x)
        x = rearrange(x, "b c (h w) -> b c h w", h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        x = self.swintransformer1(x)

        # stage 2
        x = rearrange(x, "b c h w -> b c (h w)", h=self.input_size, w=self.input_size)
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(x, "b c (h w) -> b c h w", h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        x = self.swintransformer2(x)

        x = rearrange(x, "b c h w -> b (h w) c", h=self.input_size, w=self.input_size)

        # IMPORTANT: same math; just allocate on correct device (instead of hard-coded .cuda()).
        score = torch.tensor([], device=x.device)
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        return score


def setup_seed(seed: int = 20):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_image_as_chw_float(image_path: str) -> np.ndarray:
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    img = Image.open(image_path).convert("RGB")
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    return img


def _random_crop_patches(img_chw: np.ndarray, num_crops: int, crop_size: int = 224) -> np.ndarray:
    c, h, w = img_chw.shape
    if h < crop_size or w < crop_size:
        raise ValueError(f"Image too small: {h}x{w}")
    patches = []
    for _ in range(num_crops):
        top = np.random.randint(0, h - crop_size + 1)
        left = np.random.randint(0, w - crop_size + 1)
        patch = img_chw[:, top : top + crop_size, left : left + crop_size]
        patches.append(patch)
    return np.stack(patches, axis=0)  # (N,C,H,W)


def _build_maniqa_model(device: torch.device):
    net = MANIQA(
        embed_dim=768,
        num_outputs=1,
        dim_mlp=768,
        patch_size=8,
        img_size=224,
        window_size=4,
        depths=[2, 2],
        num_heads=[4, 4],
        num_tab=2,
        scale=0.8,
    )
    return net.to(device)


@torch.no_grad()
def predict_maniqa_batch_patches(
    net: MANIQA,
    image_path: str,
    seed: int,
    device: torch.device,
    num_crops: int,
    transform,
) -> Tuple[float, List[float]]:
    """
    Keep MANIQA inference logic:
      - setup_seed(seed) per image
      - random crop num_crops patches
      - Normalize(0.5,0.5) + ToTensor()
      - score each patch -> patch_scores -> avg_score

    Speed-up: run patches as a batch in one forward.
    """
    setup_seed(seed)
    img_chw = _load_image_as_chw_float(image_path)
    patches = _random_crop_patches(img_chw, num_crops=num_crops)  # (N,C,224,224)

    # transform each patch (same logic), then stack -> (N,3,224,224)
    x_list = []
    for i in range(num_crops):
        sample = {
            "d_img_org": patches[i],
            "score": 0,
            "d_name": os.path.basename(image_path),
        }
        sample = transform(sample)
        x_list.append(sample["d_img_org"])
    x = torch.stack(x_list, dim=0).to(device)

    scores = net(x)  # (N,)
    patch_scores = [float(s.item()) for s in scores.detach().cpu()]
    avg_score = float(np.mean(patch_scores))
    return avg_score, patch_scores


# =============================
# Main evaluation
# =============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, default="") # input the generated image folder path (please first test preference metrics, then use saved preference images folder)
    parser.add_argument("--out", type=str, default="", help="Output txt path.") # input the output results path
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda / cuda:0 / cpu. Default: auto.")

    # # download link (VisualQuality-R1): https://huggingface.co/TianheWu/VisualQuality-R1-7B
    parser.add_argument("--vqr1_model", type=str, default="/home/notebook/data/group/wth/ddirl/weights/VisualQuality-R1-7B") # model ckpt path
    parser.add_argument("--vqr1_bsz", type=int, default=32)
    parser.add_argument("--vqr1_seed", type=int, default=1, help="Python random seed used in fallback + overall.")
    parser.add_argument("--vqr1_use_flash_attn2", default=True, help="Use flash_attention_2 (if available).")

    # download link (MANIQA): https://huggingface.co/chaofengc/IQA-PyTorch-Weights/blob/main/MANIQA_PIPAL-ae6d356b.pth
    parser.add_argument("--maniqa_ckpt", type=str, default="/home/notebook/data/group/wth/ddirl/weights/MANIQA_PIPAL-ae6d356b.pth", help="Path to MANIQA .pth checkpoint.") # model skpt path
    parser.add_argument("--maniqa_seed", type=int, default=20, help="Seed reset per image (same as your code).")
    parser.add_argument("--maniqa_num_crops", type=int, default=10, help="Num random crops (matches your code default=10).")

    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # gather images
    image_paths = get_image_paths(args.image_root)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found under: {args.image_root}")

    print(f"Found {len(image_paths)} images under: {args.image_root}")
    print(f"Using device: {device}")

    # -----------------
    # VQR1
    # -----------------
    random.seed(args.vqr1_seed)

    vqr1_model, vqr1_processor = build_vqr1(
        model_path=args.vqr1_model,
        device=device,
        use_flash_attn2=args.vqr1_use_flash_attn2,
    )
    vqr1_scores = score_batch_image_vqr1(
        image_paths=image_paths,
        model=vqr1_model,
        processor=vqr1_processor,
        device=device,
        bsz=args.vqr1_bsz,
    )

    vqr1_mean = float(np.mean(list(vqr1_scores.values())))
    print(f"[VQR1] mean over {len(vqr1_scores)} images: {vqr1_mean:.6f}")

    # -----------------
    # MANIQA
    # -----------------
    transform = transforms.Compose([Normalize(0.5, 0.5), ToTensor()])

    maniqa_net = _build_maniqa_model(device)
    state = torch.load(args.maniqa_ckpt, map_location="cpu")
    maniqa_net.load_state_dict(state, strict=True)
    maniqa_net.eval()

    maniqa_scores: Dict[str, float] = {}
    # optional: also store per-patch if you want (not written by default)
    # maniqa_patch_scores: Dict[str, List[float]] = {}

    for p in tqdm(image_paths, desc="MANIQA scoring"):
        try:
            avg_score, patch_scores = predict_maniqa_batch_patches(
                net=maniqa_net,
                image_path=p,
                seed=args.maniqa_seed,
                device=device,
                num_crops=args.maniqa_num_crops,
                transform=transform,
            )
            maniqa_scores[p] = float(avg_score)
            # maniqa_patch_scores[p] = patch_scores
        except Exception as e:
            print(f"[MANIQA] Error on {p}: {e}")
            maniqa_scores[p] = float("nan")

    maniqa_valid = [v for v in maniqa_scores.values() if not (isinstance(v, float) and np.isnan(v))]
    maniqa_mean = float(np.mean(maniqa_valid)) if len(maniqa_valid) > 0 else float("nan")
    print(f"[MANIQA] mean over {len(maniqa_valid)}/{len(maniqa_scores)} valid images: {maniqa_mean:.6f}")

    # -----------------
    # Save per-image results
    # -----------------
    # Format: path \t vqr1_score \t maniqa_score
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# path\tvqr1\tmaniqa\n")
        for p in image_paths:
            f.write(f"{p}\t{vqr1_scores.get(p, float('nan'))}\t{maniqa_scores.get(p, float('nan'))}\n")
        f.write("\n")
        f.write(f"# vqr1_mean\t{vqr1_mean}\n")
        f.write(f"# maniqa_mean\t{maniqa_mean}\n")

    print(f"Saved per-image scores + means to: {args.out}")
    print("Done!")


if __name__ == "__main__":
    main()




# python test_quality.py



