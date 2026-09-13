import os
import io
import time
import logging
from PIL import ImageOps
from typing import Tuple, List, Dict, Any

import cv2
import numpy as np
from PIL import Image
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
# 2. Model Initialization & Singleton Class
# -------------------------------------------------------------
class VisionService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Initializing Vision Service on %s...", self.device)

        # 1. Rembg Session
        # self.rembg_session = new_session("u2netp")
        self.rembg_session = new_session("isnet-general-use")
        # 2. CLIP
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # 3. SegFormer Clothes Segmentation
        seg_model_name = "mattmdjaga/segformer_b2_clothes"
        self.seg_processor = AutoProcessor.from_pretrained(seg_model_name)
        self.seg_model = AutoModelForSemanticSegmentation.from_pretrained(seg_model_name).to(self.device)
        logger.info("Vision Service initialized successfully.")

    def digital_iron(self, rgb_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Digitally irons out fabric wrinkles by normalizing lighting shadows

        while locking text, graphics, and weave intact.
        """
        # Work in LAB color space to avoid changing colors/tones
        lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # 1. Isolate the base illumination (crease valleys & shadows)
        # Large morphological closing bridges dark wrinkle valleys
        kernel_size = 31
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background_l = cv2.morphologyEx(l_channel, cv2.MORPH_CLOSE, kernel)

        # 2. Smooth the shadow map with guided surface blur
        smooth_illumination = cv2.bilateralFilter(background_l, d=15, sigmaColor=50, sigmaSpace=50)

        # 3. High-frequency preservation (keeps text "1992", graphics, tags sharp)
        # High frequency = Original Luminance - Local Base
        high_frequency = cv2.subtract(l_channel, background_l)

        # 4. Reconstruct flattened lighting
        # Blend smooth illumination into regions covered by the garment
        target_luminance = cv2.add(smooth_illumination, high_frequency)

        # 5. Blend 70% ironed surface with 30% original to maintain natural drape
        blended_l = cv2.addWeighted(target_luminance, 0.70, l_channel, 0.30, 0)

        # Apply only within garment bounds
        mask_normalized = (mask.astype(np.float32) / 255.0)
        final_l = (blended_l * mask_normalized + l_channel * (1.0 - mask_normalized)).astype(np.uint8)

        ironed_lab = cv2.merge([final_l, a_channel, b_channel])
        return cv2.cvtColor(ironed_lab, cv2.COLOR_LAB2RGB)

    def dewrinkle_and_cutout(self, pil_img: Image.Image) -> Image.Image:
        """Removes background cleanly, disconnecting external zipper pulls/cords."""
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        orig_w, orig_h = pil_img.size

        # 1. Native resolution boundary segmentation via rembg
        cutout_rgba = remove(
            pil_img, 
            session=self.rembg_session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=20,
            alpha_matting_erode_size=10
        )
        raw_mask = np.array(cutout_rgba.split()[-1])

        # 2. Stronger threshold to reject bedsheet/zipper artifacts (raised to 120)
        _, binary_mask = cv2.threshold(raw_mask, 120, 255, cv2.THRESH_BINARY)

        # 3. Disconnect thin bridges (zipper cords/threads touching the bottom hem)
        # An OPEN operation (erosion followed by dilation) severs thin connecting cords
        disc_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, disc_kernel)

        # 4. Extract only the single dominant garment contour
        contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid_mask = np.zeros_like(binary_mask)

        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            
            # Fill the main garment body and collar
            cv2.drawContours(solid_mask, [main_contour], -1, 255, thickness=cv2.FILLED)
        else:
            solid_mask = binary_mask

        # 5. Smooth the clean garment perimeter without re-attaching external objects
        smooth_mask = cv2.GaussianBlur(solid_mask, (5, 5), 1.5)

        # 6. Composite with untouched native RGB pixels
        orig_rgb = np.array(pil_img)
        ironed_rgb = self.digital_iron(orig_rgb, smooth_mask)
        rgba = np.dstack([orig_rgb, smooth_mask])
        result_img = Image.fromarray(rgba, mode="RGBA")

        # 7. Crop tightly to the true garment boundary
        bbox = result_img.getbbox()
        if bbox:
            pad = 16
            x1 = max(0, bbox[0] - pad)
            y1 = max(0, bbox[1] - pad)
            x2 = min(orig_w, bbox[2] + pad)
            y2 = min(orig_h, bbox[3] + pad)
            result_img = result_img.crop((x1, y1, x2, y2))

        return result_img

    def analyze_dress_shade(self, image_path: str) -> Tuple[str, np.ndarray]:
        """Calculates dominant color ignoring alpha/transparent background."""
        img = Image.open(image_path)
        
        # If PNG has alpha channel, only sample non-transparent pixels
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

        # Color pixels with minimal saturation
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
            dL, da, db = dominant_lab[0] - ref_lab[0], dominant_lab[1] - ref_lab[1], dominant_lab[2] - ref_lab[2]
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
        image = Image.open(image_path).convert("RGB")  # Ensure .convert("RGB") is present
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

    def analyze_dress_shade(self, image_path: str) -> Tuple[str, np.ndarray]:
        img = cv2.imread(image_path)
        if img is None:
            return "Unknown", np.array([128, 128, 128])

        h, w, _ = img.shape
        crop = img[int(h * 0.20):int(h * 0.80), int(w * 0.25):int(w * 0.75)]
        if crop.size == 0:
            crop = img

        crop = cv2.resize(crop, (150, 150))
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        mask = (hsv[:, :, 1] > 25) & (hsv[:, :, 2] > 30) & (hsv[:, :, 2] < 245)
        valid_pixels = lab[mask]
        if len(valid_pixels) < 200:
            valid_pixels = lab[35:115, 35:115].reshape(-1, 3)

        n_clusters = 3 if len(valid_pixels) >= 3 else 1
        kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        kmeans.fit(valid_pixels)
        counts = np.bincount(kmeans.labels_)
        dominant_lab = kmeans.cluster_centers_[np.argmax(counts)]

        closest_name = "Custom Shade"
        min_dist = float("inf")
        for name, ref_lab in COLOR_PALETTE.items():
            dL, da, db = dominant_lab[0] - ref_lab[0], dominant_lab[1] - ref_lab[1], dominant_lab[2] - ref_lab[2]
            dist = np.sqrt(0.4 * (dL ** 2) + 1.3 * (da ** 2) + 1.3 * (db ** 2))
            if dist < min_dist:
                min_dist = dist
                closest_name = name

        return closest_name, dominant_lab


# Singleton instance exported for the application
vision_service = VisionService()