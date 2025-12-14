# Copyright (c) 2025 Jaerin Lee

from typing import Tuple, List, Literal, Optional, Union
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange
from tqdm import tqdm
from PIL import Image

from util import gaussian_lowpass, blend, get_panorama_views, shift_to_mask_bbox_center
from .pipeline_semantic_draw import SemanticDrawPipeline


class SemanticDrawFreqSplitPipeline(SemanticDrawPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        negative_prompts: Union[str, List[str]] = '',
        suffix: Optional[str] = None,
        background: Optional[Union[torch.Tensor, Image.Image]] = None,
        background_prompt: Optional[str] = None,
        background_negative_prompt: str = '',
        height: int = 512,
        width: int = 512,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        prompt_strengths: Optional[Union[torch.Tensor, float, List[float]]] = None,
        masks: Optional[Union[Image.Image, List[Image.Image]]] = None,
        mask_strengths: Optional[Union[torch.Tensor, float, List[float]]] = None,
        mask_stds: Optional[Union[torch.Tensor, float, List[float]]] = None,
        use_boolean_mask: bool = True,
        do_blend: bool = True,
        tile_size: int = 768,
        bootstrap_steps: Optional[int] = None,
        boostrap_mix_steps: Optional[float] = None,
        bootstrap_leak_sensitivity: Optional[float] = None,
        preprocess_mask_cover_alpha: Optional[float] = None,
        output_type: str = 'pil',
        use_freq_split: bool = True,
        freq_split_std: float = 3.0,
    ) -> Union[Image.Image, torch.Tensor]:
        r"""Arbitrary-size image generation with Frequency-Split Latent Fusion.
        """

        ### Simplest cases

        if prompts is None or (isinstance(prompts, (list, tuple, str)) and len(prompts) == 0):
            if background is None and background_prompt is not None:
                return self.sample(background_prompt, background_negative_prompt, height, width, num_inference_steps, guidance_scale)
            return background
        elif masks is None or (isinstance(masks, (list, tuple)) and len(masks) == 0):
            return self.sample(prompts, negative_prompts, height, width, num_inference_steps, guidance_scale)


        ### Prepare generation

        if num_inference_steps is not None:
            self.prepare_lcm_schedule(list(range(num_inference_steps)), num_inference_steps)

        if guidance_scale is None:
            guidance_scale = self.default_guidance_scale


        ### Prompts & Masks

        if isinstance(masks, Image.Image):
            masks = [masks]
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        num_masks = len(masks)
        num_prompts = len(prompts)
        num_nprompts = len(negative_prompts)
        assert num_prompts in (num_masks, 1), \
            f'The number of prompts {num_prompts} should match the number of masks {num_masks}!'
        assert num_nprompts in (num_prompts, 1), \
            f'The number of negative prompts {num_nprompts} should match the number of prompts {num_prompts}!'

        fg_masks, masks_g, std = self.process_mask(
            masks,
            mask_strengths,
            mask_stds,
            height=height,
            width=width,
            use_boolean_mask=use_boolean_mask,
            timesteps=self.timesteps,
            preprocess_mask_cover_alpha=preprocess_mask_cover_alpha,
        )  # (p, t, 1, H, W)
        bg_masks = (1 - fg_masks.sum(dim=0)).clip_(0, 1)  # (T, 1, h, w)
        has_background = bg_masks.sum() > 0

        h = (height + self.vae_scale_factor - 1) // self.vae_scale_factor
        w = (width + self.vae_scale_factor - 1) // self.vae_scale_factor


        ### Background

        bg_latent = None
        if has_background:
            if background is None and background_prompt is not None:
                fg_masks = torch.cat((bg_masks[None], fg_masks), dim=0)
                if suffix is not None:
                    prompts = [p + suffix + background_prompt for p in prompts]
                prompts = [background_prompt] + prompts
                negative_prompts = [background_negative_prompt] + negative_prompts
                has_background = False # Regard that background does not exist.
            else:
                if background is None and background_prompt is None:
                    background = torch.ones(1, 3, height, width, dtype=self.dtype, device=self.device)
                    background_prompt = 'simple white background image'
                elif background is not None and background_prompt is None:
                    background_prompt = self.get_text_prompts(background)
                if suffix is not None:
                    prompts = [p + suffix + background_prompt for p in prompts]
                prompts = [background_prompt] + prompts
                negative_prompts = [background_negative_prompt] + negative_prompts
                if isinstance(background, Image.Image):
                    background = T.ToTensor()(background).to(dtype=self.dtype, device=self.device)[None]
                background = F.interpolate(background, size=(height, width), mode='bicubic', align_corners=False)
                bg_latent = self.encode_imgs(background)

        # Bootstrapping stage preparation.

        if bootstrap_steps is None:
            bootstrap_steps = self.default_bootstrap_steps
        if boostrap_mix_steps is None:
            boostrap_mix_steps = self.default_boostrap_mix_steps
        if bootstrap_leak_sensitivity is None:
            bootstrap_leak_sensitivity = self.default_bootstrap_leak_sensitivity
        if bootstrap_steps > 0:
            height_ = min(height, tile_size)
            width_ = min(width, tile_size)
            white = self.get_white_background(height, width) # (1, 4, h, w)


        ### Prepare text embeddings (optimized for the minimal encoder batch size)

        uncond_embeds, text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2 * len(prompts), 77, 768]
        if has_background:
            s = prompt_strengths
            if prompt_strengths is None:
                s = self.default_prompt_strength
            if isinstance(s, (int, float)):
                s = [s] * num_prompts
            if isinstance(s, (list, tuple)):
                assert len(s) == num_prompts, \
                    f'The number of prompt strengths {len(s)} should match the number of prompts {num_prompts}!'
                s = torch.as_tensor(s, dtype=self.dtype, device=self.device)
            s = s[:, None, None]

            be = text_embeds[:1]
            bu = uncond_embeds[:1]
            fe = text_embeds[1:]
            fu = uncond_embeds[1:]
            if num_prompts > num_nprompts:
                assert fu.shape[0] == 1 and fe.shape == num_prompts
                fu = fu.repeat(num_prompts, 1, 1)
            text_embeds = torch.lerp(be, fe, s)  # (p, 77, 768)
            uncond_embeds = torch.lerp(bu, fu, s)  # (n, 77, 768)
        elif num_prompts > num_nprompts:
            assert uncond_embeds.shape[0] == 1 and text_embeds.shape[0] == num_prompts
            uncond_embeds = uncond_embeds.repeat(num_prompts, 1, 1)
        assert uncond_embeds.shape[0] == text_embeds.shape[0] == num_prompts
        if num_masks > num_prompts:
            assert masks.shape[0] == num_masks and num_prompts == 1
            text_embeds = text_embeds.repeat(num_masks, 1, 1)
            uncond_embeds = uncond_embeds.repeat(num_masks, 1, 1)
        text_embeds = torch.cat([uncond_embeds, text_embeds])


        ### Run

        # Latent initialization.
        if self.timesteps[0] < 999 and has_background:
            latent = self.scheduler_add_noise(bg_latent, None, 0)
        else:
            latent = torch.randn((1, self.unet.config.in_channels, h, w), dtype=self.dtype, device=self.device)

        # Tiling (if needed).
        if height > tile_size or width > tile_size:
            t = (tile_size + self.vae_scale_factor - 1) // self.vae_scale_factor
            views, tile_masks = get_panorama_views(h, w, t)
            tile_masks = tile_masks.to(self.device)
        else:
            views = [(0, h, 0, w)]
            tile_masks = latent.new_ones((1, 1, h, w))
        value = torch.zeros_like(latent)
        count_all = torch.zeros_like(latent)

        with torch.autocast('cuda'):
            for i, t in enumerate(tqdm(self.timesteps)):
                fg_mask = fg_masks[:, i]
                bg_mask = bg_masks[i:i + 1]

                value.zero_()
                count_all.zero_()
                for j, (h_start, h_end, w_start, w_end) in enumerate(views):
                    fg_mask_ = fg_mask[..., h_start:h_end, w_start:w_end]
                    latent_ = latent[..., h_start:h_end, w_start:w_end].repeat(num_masks, 1, 1, 1)

                    # Bootstrap for tight background.
                    if i < bootstrap_steps:
                        mix_ratio = min(1, max(0, boostrap_mix_steps - i))
                        bg_latent_ = bg_latent[..., h_start:h_end, w_start:w_end] if has_background else latent_[:1]
                        white_ = white[..., h_start:h_end, w_start:w_end]
                        bg_latent_ = mix_ratio * white_ + (1.0 - mix_ratio) * bg_latent_
                        bg_latent_ = self.scheduler_add_noise(bg_latent_, None, i)
                        latent_ = (1.0 - fg_mask_) * bg_latent_ + fg_mask_ * latent_

                        latent_ = shift_to_mask_bbox_center(latent_, fg_mask_, reverse=True)

                    # Perform one step of the reverse diffusion.
                    noise_pred = self.unet(torch.cat([latent_] * 2), t, encoder_hidden_states=text_embeds)['sample']
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latent_ = self.scheduler_step(noise_pred, i, latent_)

                    if i < bootstrap_steps:
                        latent_ = shift_to_mask_bbox_center(latent_, fg_mask_)
                        leak = (latent_ - bg_latent_).pow(2).mean(dim=1, keepdim=True)
                        leak_sigmoid = torch.sigmoid(leak / bootstrap_leak_sensitivity) * 2 - 1
                        fg_mask_ = fg_mask_ * leak_sigmoid

                    # Mix the latents with frequency split if enabled
                    fg_mask_ = fg_mask_ * tile_masks[:, j:j + 1, h_start:h_end, w_start:w_end]

                    if use_freq_split:
                        # 1. Decompose
                        latent_low = gaussian_lowpass(latent_, freq_split_std)
                        latent_high = latent_ - latent_low

                        # 2. Low Freq Fusion (Weighted Average)
                        # We accumulate weighted sum now, and divide by global weights (count_all) later.
                        value[..., h_start:h_end, w_start:w_end] += (fg_mask_ * latent_low).sum(dim=0, keepdim=True)
                        
                        # 3. High Freq Fusion (Winner-Takes-All)
                        # Identify the winner mask index at each pixel
                        # fg_mask_: (num_masks, 1, H, W)
                        # We need a strict winner. If masks are zero, no winner (handled by count_all logic later, but for high freq we add brute force?)
                        # Logic: High freq component of the pixel comes ONLY from the latent associated with the max mask value.
                        
                        # winner_indices: (1, H, W) -> (1, 1, H, W)
                        winner_vals, winner_indices = fg_mask_.max(dim=0, keepdim=True) 
                        
                        # Create a one-hot-like tensor for winners. 
                        # Note: we only want to add high freq if there IS a valid mask coverage.
                        # If winner_val is 0, it means all masks are 0.
                        # So we filter by winner_vals > some_epsilon? Or effectively just let it be.
                        # But wait, we want to SUM the high freq components. 
                        # If we have winner mask M_w (one-hot), we do sum(M_w * latent_high).
                        # But wait, we need to add this to `value_low_sum`.
                        # Final formula: Result = (Sum(w * Low) / Sum(w)) + Component_High_Winner
                        # This means we should ADD (Component_High_Winner * Sum(w)) to the accumulator? 
                        # No that's getting complicated because `count_all` (Sum(w)) varies.
                        
                        # Alternative: Store High and Low separately?
                        # `value` currently stores the weighted sum.
                        # Let's introduce `value_high` accumulator?
                        # If we do that, we need to handle tiling logic for it too.
                        # But since we are inside the loop, we can just modify how we update `value`.
                        # `value` is reset to zero for each timestep.
                        # `count_all` is reset to zero.
                        
                        # Current final step: `latent = torch.where(count_all > 0, value / count_all, value)`
                        # So `latent` = (Masked_Sum_Low + Masked_High_Contribution) / Sum_Masks ? No.
                        
                        # We want: `latent` = (Masked_Sum_Low / Sum_Masks) + Winner_High
                        # So `Masked_Sum_Low / Sum_Masks + Winner_High`
                        # = (Masked_Sum_Low + Winner_High * Sum_Masks) / Sum_Masks
                        
                        # So we can calculate `winner_high` and add `winner_high * fg_mask_.sum(dim=0)` to `value`?
                        # `fg_mask_.sum(dim=0)` is exactly what `count_all` accumulates (per tile).
                        # But wait, `count_all` accumulates across tiles too (though tiles are disjoint/blended? Tiling adds overlaps).
                        # Assuming simple tiling or standard case:
                        
                        # Let's compute `winner_high` for this tile.
                        # `winner_mask` should be 1 where mask k is max, 0 otherwise.
                        winner_mask = torch.zeros_like(fg_mask_)
                        winner_mask.scatter_(0, winner_indices, 1.0)
                        
                        # Ensure we handle multiple maxes or zeros safely?
                        # If multiple maxes, scatter writes to one.
                        # If all zeros? argmax is 0. winner_mask has 1 at index 0. 
                        # If index 0 mask is 0.0, then it doesn't matter much if we add 0?
                        # Well, latent_high is not 0.
                        # We should mask `winner_mask` by `(winner_vals > 0)`.
                        winner_mask = winner_mask * (winner_vals > 0).float()
                        
                        # `selected_high` = (winner_mask * latent_high).sum(dim=0, keepdim=True)
                        selected_high = (winner_mask * latent_high).sum(dim=0, keepdim=True)
                        
                        # Now add to value.
                        # We want `value / count` to be `avg_low + selected_high`.
                        # `value / count = avg_low + selected_high`
                        # `value = avg_low * count + selected_high * count`
                        # We know `(fg_mask_ * latent_low).sum` corresponds to `avg_low * count` (roughly, locally).
                        # Current tile mask sum is `current_weight_sum = fg_mask_.sum(dim=0, keepdim=True)`.
                        
                        current_weight_sum = fg_mask_.sum(dim=0, keepdim=True)
                        
                        # So we add `selected_high * current_weight_sum` to `value`?
                        # Yes, this allows `value / count_all` to recover `selected_high` (assuming `current_weight_sum` matches `count_all` finally).
                        # In overlapping tiles, `count_all` is sum of tile weights. 
                        # This should work reasonably well.
                        
                        value[..., h_start:h_end, w_start:w_end] += selected_high * current_weight_sum

                    else:
                        # Original Logic
                        value[..., h_start:h_end, w_start:w_end] += (fg_mask_ * latent_).sum(dim=0, keepdim=True)

                    count_all[..., h_start:h_end, w_start:w_end] += fg_mask_.sum(dim=0, keepdim=True)

                latent = torch.where(count_all > 0, value / count_all, value)
                bg_mask = (1 - count_all).clip_(0, 1)  # (T, 1, h, w)
                if has_background:
                    latent = (1 - bg_mask) * latent + bg_mask * bg_latent

                # Noise is added after mixing.
                if i < len(self.timesteps) - 1:
                    latent = self.scheduler_add_noise(latent, None, i + 1)

        if output_type == 'latent':
            return latent

        # Return PIL Image.
        image = self.decode_latents(latent.to(dtype=self.dtype))[0]
        if has_background and do_blend:
            fg_mask = torch.sum(masks_g, dim=0).clip_(0, 1)
            image = blend(image, background[0], fg_mask)
        else:
            image = T.ToPILImage()(image)
        return image
