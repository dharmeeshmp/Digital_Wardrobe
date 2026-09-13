import os
import io
import time
import logging
from typing import Tuple, List, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
import torch
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoProcessor,
    AutoModelForSemanticSegmentation
)
from rembg import new_session, remove

logger = logging.getLogger("ApparelMatcher.Vision")

# -------------------------------------------------------------
# 1. Taxonomies & Palettes
# -------------------------------------------------------------
ALL_PATTERNS = [
    "solid plain color", "gingham check", "houndstooth pattern", "plaid or tartan",
    "glen plaid or prince of wales check", "harlequin diamond pattern", "buffalo check",
    "windowpane check", "argyle diamond pattern", "polka dot", "vertical stripes or pinstripe",
    "horizontal stripes", "awning wide stripes", "chevron or zig-zag pattern",
    "herringbone weave pattern", "greek key pattern", "trellis geometric pattern",
    "quatrefoil pattern", "floral print", "ditsy small floral print", "paisley pattern",
    "toile de jouy pattern", "chintz floral pattern", "damask ornate pattern",
    "jacquard or brocade weave", "fleur-de-lis pattern", "ikat pattern", "batik pattern",
    "tie-dye or shibori", "camouflage pattern", "ombre gradient pattern",
    "abstract graphic print", "geometric print", "leopard print", "cheetah print",
    "zebra stripe print", "snake or python skin print", "tiger stripe print",
    "cow print", "tortoiseshell pattern"
]

ALL_GARMENT_TYPES = [
    "casual shirt", "formal dress shirt", "t-shirt", "polo shirt",
    "tank top or vest", "hoodie or sweatshirt", "sweater or cardigan",
    "kurta or kurti", "sherwani", "saree", "salwar kameez",
    "lehenga choli", "dhoti or veshti or lungi",
    "jeans or denim pants", "formal trousers or slacks", "chinos or casual trousers",
    "cargo pants", "shorts", "skirt", "track pants or sweatpants",
    "suit or blazer", "jacket or coat", "leather jacket",
    "raincoat or windbreaker", "maxi dress or long gown",
    "cocktail or party dress", "casual summer dress", "jumpsuit or romper"
]

COLOR_PALETTE = {
    "Pure White": np.array([255, 128, 128]),
    "Off-White / Pearl": np.array([242, 127, 131]),
    "Ivory / Cream": np.array([238, 126, 142]),
    "Silver / Light Grey": np.array([215, 128, 128]),
    "Heather Grey": np.array([145, 128, 128]),
    "Charcoal / Dark Grey": np.array([75, 128, 128]),
    "Jet Black": np.array([20, 128, 128]),
    "Baby Pink / Light Pink": np.array([210, 150, 134]),
    "Rose Pink": np.array([170, 162, 137]),
    "Hot Pink": np.array([150, 202, 138]),
    "Fuchsia / Dark Pink": np.array([130, 192, 140]),
    "Magenta / Berry": np.array([105, 186, 146]),
    "Lavender / Lilac": np.array([195, 142, 116]),
    "Deep Purple / Plum": np.array([85, 165, 96]),
    "Bright Red": np.array([135, 200, 168]),
    "Maroon / Wine": np.array([75, 170, 145]),
    "Peach / Coral": np.array([195, 148, 158]),
    "Orange": np.array([170, 155, 185]),
    "Rust / Terracotta": np.array([120, 155, 165]),
    "Lemon Yellow": np.array([230, 120, 185]),
    "Mustard Yellow": np.array([165, 135, 180]),
    "Beige / Khaki": np.array([220, 130, 140]),
    "Tan / Camel": np.array([155, 136, 152]),
    "Chocolate Brown": np.array([70, 138, 142]),
    "Sky Blue": np.array([205, 118, 114]),
    "Teal / Turquoise": np.array([140, 102, 118]),
    "Royal Blue": np.array([95, 140, 80]),
    "Navy Blue": np.array([45, 134, 98]),
    "Mint Green": np.array([205, 108, 136]),
    "Olive Green": np.array([120, 122, 154]),
    "Emerald Green": np.array([85, 105, 138]),
    "Dark Forest Green": np.array([55, 112, 130])
}

VALIDATION_PROMPTS = [
    "a photograph of a person wearing clothing, a dress, a saree, or a fashion outfit",
    "a clean single photograph of a piece of clothing, apparel, or wearable garment",
    "a close-up portrait of only a human face without clothes visible",
    "a photo of an animal or pet",
    "a vehicle, car, bicycle, or motorcycle",
    "an electronic device, screen, laptop, or mobile phone",
    "furniture, indoor room background, or household object",
    "food, beverage, dish, or meal",
    "a document, text, paper, meme, or graphic artwork"
]

# -------------------------------------------------------------
# 2. Vision Service Implementation
# -------------------------------------------------------------
class VisionService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Initializing Vision Service on %s...", self.device)

        # 1. Rembg Session for isolated product cutouts
        self.rembg_session = new_session("isnet-general-use")

        # 2. CLIP Model for semantics & classification
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # 3. SegFormer Clothes Parsing Model (Human Body to Apparel Isolator)
        seg_model_name = "mattmdjaga/segformer_b2_clothes"
        self.seg_processor = AutoProcessor.from_pretrained(seg_model_name)
        self.seg_model = AutoModelForSemanticSegmentation.from_pretrained(seg_model_name).to(self.device)

        logger.info("Vision Service initialized successfully.")

    def is_human_model(self, pil_img: Image.Image) -> bool:
        """Determines if a human model is present by checking SegFormer's face/hair/skin predictions."""
        # Work on a thumbnail for fast zero-overhead detection
        thumb = pil_img.copy()
        thumb.thumbnail((256, 256))
        
        inputs = self.seg_processor(images=thumb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.seg_model(**inputs)
            pred_seg = outputs.logits.argmax(dim=1)[0].cpu().numpy()

        # SegFormer Clothes label definitions:
        # 1: Hat, 2: Hair, 3: Sunglasses, 11: Face/Skin, 12: Left Arm, 13: Right Arm, 14: Left Leg, 15: Right Leg
        human_body_classes = [1, 2, 3, 11, 12, 13, 14, 15]
        human_pixels = np.isin(pred_seg, human_body_classes).sum()

        # If more than 3% of the image contains face, arms, or human skin, it is an on-model photo
        return human_pixels > (pred_seg.size * 0.03)

    def parse_garment_from_human(self, pil_img: Image.Image, target: str = "upper") -> Image.Image:
        """Isolates clothing from a human model photo with sharp native edges and zero blur."""
        orig_w, orig_h = pil_img.size
        inputs = self.seg_processor(images=pil_img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.seg_model(**inputs)
            logits = outputs.logits

        # 1. Upsample SegFormer mask
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        # SegFormer classes: 4: Upper, 5: Skirt, 6: Pants, 7: Dress, 10: Scarf
        if target == "upper":
            target_ids = [4, 7, 10]
        elif target == "lower":
            target_ids = [5, 6]
        else:
            target_ids = [4, 5, 6, 7, 10]

        raw_mask = np.isin(pred_seg, target_ids).astype(np.uint8) * 255

        if np.count_nonzero(raw_mask) < 2000:
            return self.dewrinkle_and_cutout(pil_img)

        # 2. Refine mask edges (eliminate stepping without introducing blur)
        # Use a tight binary threshold to lock silhouette boundaries
        _, crisp_mask = cv2.threshold(raw_mask, 127, 255, cv2.THRESH_BINARY)

        # Remove single-pixel noise specs
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        crisp_mask = cv2.morphologyEx(crisp_mask, cv2.MORPH_OPEN, kernel)

        # Subtle 1-pixel anti-aliasing along the boundary
        smooth_mask = cv2.GaussianBlur(crisp_mask, (3, 3), 0.8)

        # 3. Use 100% UNMODIFIED original camera pixels (no iron, no filter)
        orig_rgb = np.array(pil_img)
        rgba = np.dstack([orig_rgb, smooth_mask])
        result_img = Image.fromarray(rgba, mode="RGBA")

        # 4. Crop tightly to garment
        bbox = result_img.getbbox()
        if bbox:
            pad = 16
            x1 = max(0, bbox[0] - pad)
            y1 = max(0, bbox[1] - pad)
            x2 = min(orig_w, bbox[2] + pad)
            y2 = min(orig_h, bbox[3] + pad)
            result_img = result_img.crop((x1, y1, x2, y2))

        return result_img
    def digital_iron(self, rgb_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Digitally irons out fabric wrinkles by normalizing lighting shadows

        while locking text, graphics, and weave intact.
        """
        lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # 1. Base illumination isolation
        kernel_size = 31
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background_l = cv2.morphologyEx(l_channel, cv2.MORPH_CLOSE, kernel)

        # 2. Smooth the shadow map
        smooth_illumination = cv2.bilateralFilter(background_l, d=15, sigmaColor=50, sigmaSpace=50)

        # 3. High-frequency preservation
        high_frequency = cv2.subtract(l_channel, background_l)

        # 4. Reconstruct flattened lighting
        target_luminance = cv2.add(smooth_illumination, high_frequency)

        # 5. Blend 70% ironed surface with 30% original
        blended_l = cv2.addWeighted(target_luminance, 0.70, l_channel, 0.30, 0)

        mask_normalized = (mask.astype(np.float32) / 255.0)
        final_l = (blended_l * mask_normalized + l_channel * (1.0 - mask_normalized)).astype(np.uint8)

        ironed_lab = cv2.merge([final_l, a_channel, b_channel])
        return cv2.cvtColor(ironed_lab, cv2.COLOR_LAB2RGB)

    def smooth_fabric_wrinkles(self, rgba_img: Image.Image) -> Image.Image:
        """Edge-gated ironing that preserves prints, stitching, and fabric texture."""
        np_img = np.array(rgba_img)
        if np_img.ndim != 3 or np_img.shape[2] != 4:
            return rgba_img

        bgr = cv2.cvtColor(np_img[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha = np_img[:, :, 3]

        # Process strictly inside the valid cloth mask.
        mask = (alpha > 30).astype(np.uint8) * 255
        if cv2.countNonZero(mask) == 0:
            return rgba_img

        # Protect prints, logos, and other high-contrast details from ironing.
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = cv2.magnitude(grad_x, grad_y)
        cloth_pixels = gradient_mag[mask > 0]
        max_grad = np.percentile(cloth_pixels, 98) if cloth_pixels.size else 1.0
        max_grad = max(max_grad, 1.0)
        norm_grad = np.clip(gradient_mag / max_grad, 0.0, 1.0)
        print_protection_weight = cv2.threshold(
            norm_grad, 0.22, 1.0, cv2.THRESH_BINARY
        )[1]
        print_protection_weight = cv2.dilate(
            print_protection_weight.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        ).astype(np.float32)

        # Process luminance only so the original fabric colors remain unchanged.
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_float = l_channel.astype(np.float32)

        # Extract micro-texture such as weave, stitching, ribs, and prints.
        l_micro_base = cv2.GaussianBlur(
            l_float, (0, 0), sigmaX=1.2, sigmaY=1.2
        )
        fabric_micro_texture = l_float - l_micro_base

        # Fill fold troughs on the low-frequency lighting layer.
        morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        l_base_uint8 = np.clip(l_micro_base, 0, 255).astype(np.uint8)
        l_closed = cv2.morphologyEx(
            l_base_uint8, cv2.MORPH_CLOSE, morph_k
        ).astype(np.float32)
        shadow_valleys = cv2.subtract(l_closed, l_micro_base)
        l_shadows_lifted = l_micro_base + (shadow_valleys * 0.60)

        # Smooth the lighting channel while preserving boundaries.
        l_smooth = cv2.bilateralFilter(
            np.clip(l_shadows_lifted, 0, 255).astype(np.uint8),
            d=7,
            sigmaColor=25,
            sigmaSpace=7,
        ).astype(np.float32)

        # Reinject micro-texture and selectively blend the ironed luminance.
        l_ironed = l_smooth + fabric_micro_texture
        iron_weight = (1.0 - print_protection_weight) * 0.75
        l_result_float = (
            l_float * (1.0 - iron_weight) + l_ironed * iron_weight
        )

        # Keep the outer silhouette pixel-identical to the original.
        safe_mask = cv2.erode(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        l_final_float = np.where(safe_mask > 0, l_result_float, l_float)
        l_final = np.clip(l_final_float, 0, 255).astype(np.uint8)

        merged_lab = cv2.merge([l_final, a_channel, b_channel])
        result_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        output_np = np.dstack((result_rgb, alpha))
        return Image.fromarray(output_np)

    def dewrinkle_and_cutout(self, pil_img: Image.Image) -> Image.Image:
        """Performs background cutout and fabric smoothing."""
        try:
            cutout_rgba = remove(pil_img)

            if cutout_rgba is None or not isinstance(cutout_rgba, Image.Image):
                return pil_img.convert("RGBA")

            try:
                ironed = self.smooth_fabric_wrinkles(cutout_rgba)
                if ironed is not None and isinstance(ironed, Image.Image):
                    return ironed
            except Exception as wrinkle_err:
                logger.warning("Dewrinkling step skipped: %s", wrinkle_err)
                return cutout_rgba

            return cutout_rgba
        except Exception as e:
            logger.error("Cutout failed: %s", e)
            return pil_img.convert("RGBA")

    

    

    def process_any_garment(self, pil_img: Image.Image, target: str = "upper") -> Image.Image:
        """Unified entry: guarantees a valid PIL Image object is returned."""
        if pil_img is None:
            raise ValueError("Input image cannot be None")

        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        pil_img = ImageOps.exif_transpose(pil_img)

        try:
            cleaned = self.dewrinkle_and_cutout(pil_img)
            if cleaned is not None and isinstance(cleaned, Image.Image):
                return cleaned
        except Exception as e:
            logger.error("Error in dewrinkle_and_cutout: %s", e)

        return pil_img.convert("RGBA")

    def analyze_dress_shade(self, image_path: str) -> Tuple[str, np.ndarray]:
        """Calculates dominant color, prioritizing non-transparent pixels."""
        img = Image.open(image_path)
        
        if img.mode == 'RGBA':
            np_rgba = np.array(img)
            rgb = np_rgba[:, :, :3]
            alpha = np_rgba[:, :, 3]
            valid_mask = alpha > 128
        else:
            rgb = np.array(img.convert('RGB'))
            valid_mask = np.ones((rgb.shape[0], rgb.shape[1]), dtype=bool)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Filter out extremely desaturated or dark pixels
        color_mask = (hsv[:, :, 1] > 20) & (hsv[:, :, 2] > 30) & (hsv[:, :, 2] < 250) & valid_mask
        valid_pixels = lab[color_mask]

        if len(valid_pixels) < 100:
            valid_pixels = lab[valid_mask]
            if len(valid_pixels) == 0:
                return "Unknown", np.array([128, 128, 128])

        n_clusters = 3 if len(valid_pixels) >= 3 else 1
        kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        kmeans.fit(valid_pixels)
        counts = np.bincount(kmeans.labels_)
        dominant_lab = kmeans.cluster_centers_[np.argmax(counts)]

        closest_name = "Custom Shade"
        min_dist = float("inf")
        for name, ref_lab in COLOR_PALETTE.items():
            dL = dominant_lab[0] - ref_lab[0]
            da = dominant_lab[1] - ref_lab[1]
            db = dominant_lab[2] - ref_lab[2]
            dist = np.sqrt(0.4 * (dL ** 2) + 1.3 * (da ** 2) + 1.3 * (db ** 2))
            if dist < min_dist:
                min_dist = dist
                closest_name = name

        return closest_name, dominant_lab

    def crop_to_garment(self, input_path: str) -> None:
        img = Image.open(input_path).convert("RGB")
        max_dim = 1024
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        nobg = remove(img, session=self.rembg_session)
        bbox = nobg.getbbox()
        if bbox:
            cropped = img.crop(bbox)
            cropped.save(input_path, "JPEG", quality=95)
        else:
            img.save(input_path, "JPEG", quality=95)

    def is_garment(self, image_path: str, min_confidence: float = 0.20) -> Tuple[bool, float, str]:
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(
            text=VALIDATION_PROMPTS, images=image, return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        combined = float(probs[0] + probs[1])
        is_valid = (best_idx in [0, 1]) or (combined >= min_confidence)
        detected_label = "Apparel" if is_valid else VALIDATION_PROMPTS[best_idx]
        return is_valid, confidence, detected_label

    def classify_garment_type(self, image_path: str) -> Tuple[str, float]:
        image = Image.open(image_path).convert("RGB")
        prompts = [f"a photograph of a {item}" for item in ALL_GARMENT_TYPES]
        inputs = self.clip_processor(text=prompts, images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            probs = self.clip_model(**inputs).logits_per_image.softmax(dim=1).cpu().numpy()[0]
        best_idx = int(np.argmax(probs))
        return ALL_GARMENT_TYPES[best_idx].title(), float(probs[best_idx])

    def classify_pattern_type(self, image_path: str) -> Tuple[str, float]:
        image = Image.open(image_path).convert("RGB")
        prompts = [f"clothing fabric with a {pattern}" for pattern in ALL_PATTERNS]
        inputs = self.clip_processor(text=prompts, images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            probs = self.clip_model(**inputs).logits_per_image.softmax(dim=1).cpu().numpy()[0]
        best_idx = int(np.argmax(probs))
        return ALL_PATTERNS[best_idx].title(), float(probs[best_idx])

    def extract_visual_embedding(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                tensor = outputs.pooler_output
            elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                tensor = outputs.image_embeds
            elif isinstance(outputs, (tuple, list)):
                tensor = outputs[0]
            else:
                tensor = outputs
            features = tensor.squeeze().cpu().numpy()

        norm = np.linalg.norm(features)
        return (features / norm) if norm > 0 else features


# Singleton instance exported for the application
vision_service = VisionService()