import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from PIL import Image
import torchvision.transforms as T
from typing import Union, List, Optional
from scipy.ndimage import gaussian_filter

from util import blend, get_panorama_views, shift_to_mask_bbox_center
from model.pipeline_semantic_draw import SemanticDrawPipeline

class SemanticDrawSuperResPipeline(SemanticDrawPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def upsample_latents(self, latents: torch.Tensor, target_height: int, target_width: int, mode='bicubic') -> torch.Tensor:
        """Upscale latents to target resolution."""
        dtype_original = latents.dtype
        latents = latents.to(torch.float32)
        target_latent_h = target_height // self.vae_scale_factor
        target_latent_w = target_width // self.vae_scale_factor
        upscaled = F.interpolate(
            latents, 
            size=(target_latent_h, target_latent_w), 
            mode=mode, 
            align_corners=False if mode in ['bicubic', 'bilinear'] else None
        )
        return upscaled.to(dtype_original)

    def create_soft_latent_mask(self, mask: torch.Tensor, target_height: int, target_width: int, feather_width: int = 5) -> torch.Tensor:
        """
        Convert region mask to soft latent-space mask with Gaussian blur.
        
        Args:
            mask: Binary mask tensor [H, W] or [C, H, W] or [1, C, H, W]
            target_height, target_width: Target image dimensions
            feather_width: Gaussian blur sigma in latent pixels (default 5 per spec)
        
        Returns:
            Soft mask [1, 1, H_lat, W_lat] with values in [0, 1]
        """
        # Handle different input dimensions and ensure single channel
        if mask.dim() == 4:  # [B, C, H, W]
            mask = mask[0]  # Take first batch
        if mask.dim() == 3:  # [C, H, W]
            if mask.shape[0] == 3:  # RGB
                mask = mask.mean(dim=0, keepdim=True)  # Convert to grayscale
            elif mask.shape[0] > 1:  # Multi-channel
                mask = mask[0:1]  # Take first channel
        elif mask.dim() == 2:  # [H, W]
            mask = mask.unsqueeze(0)  # [1, H, W]
        
        # Now mask is [1, H, W], add batch dimension
        mask = mask.unsqueeze(0)  # [1, 1, H, W]
        
        # Resize to latent resolution
        latent_h = target_height // self.vae_scale_factor
        latent_w = target_width // self.vae_scale_factor
        mask_latent = F.interpolate(mask.float(), size=(latent_h, latent_w), mode='bilinear', align_corners=False)
        
        # Apply Gaussian blur for soft feathering
        mask_np = mask_latent.squeeze().cpu().numpy()  # [H_lat, W_lat]
        mask_np = gaussian_filter(mask_np, sigma=feather_width)
        mask_soft = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(mask.device)  # [1, 1, H_lat, W_lat]
        
        return mask_soft.clamp(0, 1)

    # In src/model/pipeline_semantic_draw_super_res.py

    @torch.no_grad()
    def __call__(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        negative_prompts: Union[str, List[str]] = '',
        masks: Optional[Union[Image.Image, List[Image.Image], torch.Tensor]] = None,
        mask_strengths: Optional[Union[torch.Tensor, float, List[float]]] = None,
        mask_stds: Optional[Union[torch.Tensor, float, List[float]]] = None,
        bootstrap_steps: Optional[int] = None,
        # Dimensions
        height: int = 1024,
        width: int = 1024,
        low_res_height: int = 512,
        low_res_width: int = 512,
        
        # === FIX IS HERE ===
        # Stage 1 (Low-Res) must use LCM parameters, 
        # as it calls the LCM-based SemanticDrawPipeline
        num_inference_steps_low: int = 4,   # Changed from 16
        guidance_scale_low: float = 1.0,  # Changed from 7.5
        # === END FIX ===

        # Stage 2 (High-Res Refinement) uses standard diffusion parameters
        num_inference_steps_high: int = 20, 
        guidance_scale_high: float = 9.0,  # Spec (TS#18) recommends higher guidance (8-12)
        
        # Refinement params
        refinement_strength: float = 0.5,  # Per spec (t_r = 0.5)
        use_renoise: bool = True,           # Per spec
        mask_feather_width: int = 5,       # latent pixels
        
        # Misc
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        """
        Two-stage latent super-resolution pipeline:
        1. Generate low-res latents (fast, coarse semantics)
        2. Upscale to target resolution
        3. Masked refinement (short, detail-focused)
        """
        
        if seed is not None:
            torch.manual_seed(seed)
            self.generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            self.generator = None

        # =========================================
        # STAGE 1: Low-Res Coarse Generation
        # =========================================
        print(f"[SuperRes] Stage 1: Generating coarse {low_res_height}x{low_res_width} latents (LCM)...")
        
        # This super().__call__ points to SemanticDrawPipeline, which is an LCM pipeline
        latents_low = super().__call__(
            prompts=prompts,
            negative_prompts=negative_prompts,
            masks=masks,
            mask_strengths=mask_strengths,
            mask_stds=mask_stds,
            bootstrap_steps=bootstrap_steps,
            height=low_res_height,
            width=low_res_width,
            num_inference_steps=num_inference_steps_low,  # Use 4 steps
            guidance_scale=guidance_scale_low,    # Use 1.0 guidance
            output_type='latent',
            **kwargs
        )

        # =========================================
        # INTERMEDIATE: Upscaling
        # =========================================
        print(f"[SuperRes] Upscaling latents to {height}x{width}...")
        # latents_high is the CLEAN upscaled latent
        latents_high = self.upsample_latents(latents_low, height, width, mode='bicubic')

        # =========================================
        # STAGE 2: High-Res Masked Refinement
        # =========================================
        
        # 1. Setup scheduler for high-res refinement
        self.scheduler.set_timesteps(num_inference_steps_high)
        timesteps = self.scheduler.timesteps
        
        # Calculate starting timestep
        t_start_idx = int(refinement_strength * len(timesteps))
        t_start = timesteps[t_start_idx]
        num_refinement_steps = len(timesteps) - t_start_idx
        print(f"[SuperRes] Stage 2: Refining from strength {refinement_strength} (timestep {t_start.item()}, {num_refinement_steps} steps)...")

        # 2. Get consistent noise
        noise = torch.randn(latents_high.shape, device=self.device, dtype=self.dtype, generator=self.generator)
        
        # 3. Re-noise the upscaled latents
        if use_renoise:
            latents_refined = self.scheduler.add_noise(latents_high, noise, t_start.unsqueeze(0))
        else:
            latents_refined = latents_high 
            t_start_idx = int((1.0 - refinement_strength) * len(timesteps))
            
        # 4. Prepare text embeddings (Regional)
        if isinstance(prompts, str): 
            prompts = [prompts]
        if isinstance(negative_prompts, str): 
            negative_prompts = [negative_prompts] * len(prompts)
        
        num_prompts = len(prompts)
        uncond_embeds, text_embeds = self.get_text_embeds(prompts, negative_prompts)
        text_embeds_cfg = torch.cat([uncond_embeds, text_embeds])
        
        # 5. Prepare soft masks
        all_soft_masks = None
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]
            
            if len(masks) != num_prompts:
                 if len(masks) == 1:
                    masks = masks * num_prompts # Broadcast mask
                 else:
                    print(f"[SuperRes] Warning: Mismatch between {num_prompts} prompts and {len(masks)} masks. Truncating masks.")
                    masks = masks[:num_prompts]
                
            soft_masks_list = []
            for mask in masks:
                if isinstance(mask, Image.Image):
                    mask = T.ToTensor()(mask.convert('L'))
                elif isinstance(mask, torch.Tensor) and mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                
                soft_mask = self.create_soft_latent_mask(mask, height, width, mask_feather_width)
                soft_masks_list.append(soft_mask.to(self.device))
            
            all_soft_masks = torch.cat(soft_masks_list, dim=0) # [P, 1, H_lat, W_lat]
            all_soft_masks = all_soft_masks.expand(-1, latents_high.shape[1], -1, -1) # [P, 4, H_lat, W_lat]
            
            combined_mask = all_soft_masks.max(dim=0, keepdim=True)[0] # [1, 4, H_lat, W_lat]
            bg_mask = (1.0 - combined_mask).clamp(0, 1)
            print(f"[SuperRes] Using masked refinement for {num_prompts} regions.")
        else:
            # Fallback to global refinement
            num_prompts = 1
            text_embeds_avg = text_embeds.mean(dim=0, keepdim=True)
            uncond_embeds_avg = uncond_embeds.mean(dim=0, keepdim=True)
            text_embeds_cfg = torch.cat([uncond_embeds_avg, uncond_embeds_avg])
            bg_mask = torch.zeros_like(latents_refined) # No background mask
            all_soft_masks = None
            print(f"[SuperRes] Using global refinement.")

        # 6. Refinement denoising loop
        with torch.autocast('cuda'):
            for i, t in enumerate(tqdm(timesteps[t_start_idx:], desc="Stage 2 (Refinement)")):
                
                i_real = t_start_idx + i
                
                if i_real + 1 < len(timesteps):
                    t_next = timesteps[i_real + 1].unsqueeze(0)
                else:
                    t_next = torch.tensor([0], device=timesteps.device, dtype=timesteps.dtype) # Final step
                
                if t_next > 0:
                    bg_target = self.scheduler.add_noise(latents_high, noise, t_next)
                else:
                    bg_target = latents_high # Final target is the clean latent

                latent_model_input = latents_refined.repeat(num_prompts, 1, 1, 1) 
                latent_model_input_cfg = torch.cat([latent_model_input] * 2) 
                latent_model_input_cfg = self.scheduler.scale_model_input(latent_model_input_cfg, t)

                noise_pred = self.unet(
                    latent_model_input_cfg, t, encoder_hidden_states=text_embeds_cfg
                )['sample'] 

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale_high * (noise_pred_text - noise_pred_uncond) 

                latents_updated = self.scheduler.step(noise_pred, t, latent_model_input)['prev_sample'] 
                
                if all_soft_masks is not None:
                    value = (all_soft_masks * latents_updated).sum(dim=0, keepdim=True) 
                    count = all_soft_masks.sum(dim=0, keepdim=True) 
                    latents_fg = torch.where(count > 1e-4, value / count, 0)
                    
                    latents_refined = (1 - bg_mask) * latents_fg + bg_mask * bg_target
                else:
                    latents_refined = latents_updated[0:1] 

        # 7. Final latent is the result of the loop
        latents_final = latents_refined

        # 8. Decode to image
        image = self.decode_latents(latents_final.to(self.dtype))[0]
        return T.ToPILImage()(image.cpu())