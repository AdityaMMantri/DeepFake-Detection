"""
Agentic Reasoning Module for Deepfake Detection.

A rule-based forensic investigator agent that:
1. Analyzes 9 forensic features extracted from the ViT
2. Computes a trust score (0.0 to 1.0)
3. Calibrates the final probability: P_final = P_vit × trust_score
4. Generates a human-readable forensic explanation

The agent is NOT trained — it uses expert-defined rules and thresholds.
"""

from utils import config


class ForensicAgent:
    """
    Forensic Investigator Agent.

    Analyzes the ViT's internal signals to determine how much to trust
    its prediction and generate a human-readable explanation.
    """

    def __init__(self):
        # Trust score thresholds from config
        self.entropy_high = config.AGENT_ENTROPY_THRESHOLD_HIGH
        self.entropy_low = config.AGENT_ENTROPY_THRESHOLD_LOW
        self.margin_low = config.AGENT_MARGIN_THRESHOLD_LOW
        self.margin_high = config.AGENT_MARGIN_THRESHOLD_HIGH
        self.agreement_threshold = config.AGENT_AGREEMENT_THRESHOLD
        self.cls_norm_min = config.AGENT_CLS_NORM_MIN
        self.cls_norm_max = config.AGENT_CLS_NORM_MAX

    def analyze(self, features):
        """
        Full agent analysis pipeline.

        Args:
            features: dict with 9 forensic features from ForensicFeatureExtractor

        Returns:
            dict with:
                - prediction: 'real' or 'fake'
                - p_vit: raw ViT probability
                - trust_score: agent's trust in the prediction (0-1)
                - p_final: calibrated probability (p_vit × trust_score)
                - confidence: final confidence percentage
                - explanation: human-readable forensic analysis
                - trust_factors: breakdown of trust score components
        """
        p_vit = features["vit_fake_probability"]
        cls_norm = features["cls_token_norm"]
        entropy = features["attention_entropy"]
        patch_var = features["patch_variance"]
        rgb_attn = features["rgb_attention_weight"]
        fft_attn = features["fft_attention_weight"]
        noise_attn = features["noise_attention_weight"]
        agreement = features["modality_agreement"]
        margin = features["prediction_margin"]

        # === Compute Trust Score ===
        trust_score, trust_factors = self._compute_trust_score(
            entropy, margin, agreement, cls_norm, patch_var
        )

        # === Calibrate Final Probability ===
        # P_final = P_vit × trust_score (for fake probability)
        p_final = p_vit * trust_score

        # Determine prediction
        if p_final >= 0.5:
            prediction = "fake"
            confidence = p_final * 100
        else:
            prediction = "real"
            confidence = (1 - p_final) * 100

        # === Generate Explanation ===
        explanation = self._generate_explanation(
            prediction, p_vit, trust_score, p_final, confidence,
            entropy, margin, agreement, cls_norm,
            rgb_attn, fft_attn, noise_attn, trust_factors
        )

        return {
            "prediction": prediction,
            "p_vit": p_vit,
            "trust_score": trust_score,
            "p_final": p_final,
            "confidence": round(confidence, 2),
            "explanation": explanation,
            "trust_factors": trust_factors,
            "features": features,
        }

    def _compute_trust_score(self, entropy, margin, agreement, cls_norm, patch_var):
        """
        Compute trust score based on multiple forensic signals.

        Starts at 1.0 (full trust) and applies penalties/bonuses.

        Returns:
            tuple: (trust_score, trust_factors_dict)
        """
        trust = 1.0
        factors = {}

        # --- Factor 1: Attention Entropy ---
        # High entropy → unfocused attention → less trustworthy
        if entropy > self.entropy_high:
            penalty = min(0.3, (entropy - self.entropy_high) * 0.1)
            trust -= penalty
            factors["attention_focus"] = f"PENALTY -{penalty:.2f} (entropy={entropy:.2f}, unfocused)"
        elif entropy < self.entropy_low:
            bonus = 0.05
            trust += bonus
            factors["attention_focus"] = f"BONUS +{bonus:.2f} (entropy={entropy:.2f}, very focused)"
        else:
            factors["attention_focus"] = f"NEUTRAL (entropy={entropy:.2f})"

        # --- Factor 2: Prediction Margin ---
        # Small margin → uncertain prediction → less trustworthy
        if margin < self.margin_low:
            penalty = 0.25
            trust -= penalty
            factors["prediction_certainty"] = f"PENALTY -{penalty:.2f} (margin={margin:.2f}, very uncertain)"
        elif margin < self.margin_high:
            penalty = 0.10
            trust -= penalty
            factors["prediction_certainty"] = f"PENALTY -{penalty:.2f} (margin={margin:.2f}, somewhat uncertain)"
        else:
            bonus = 0.05
            trust += bonus
            factors["prediction_certainty"] = f"BONUS +{bonus:.2f} (margin={margin:.2f}, confident)"

        # --- Factor 3: Modality Agreement ---
        # Low agreement → modalities disagree → less trustworthy
        if agreement < self.agreement_threshold:
            penalty = min(0.2, (self.agreement_threshold - agreement) * 0.5)
            trust -= penalty
            factors["modality_agreement"] = f"PENALTY -{penalty:.2f} (agreement={agreement:.2f}, modalities disagree)"
        else:
            bonus = 0.05
            trust += bonus
            factors["modality_agreement"] = f"BONUS +{bonus:.2f} (agreement={agreement:.2f}, modalities agree)"

        # --- Factor 4: CLS Token Norm ---
        # Abnormal CLS norm → unusual model behavior
        if cls_norm < self.cls_norm_min or cls_norm > self.cls_norm_max:
            penalty = 0.10
            trust -= penalty
            factors["cls_activation"] = f"PENALTY -{penalty:.2f} (cls_norm={cls_norm:.2f}, abnormal)"
        else:
            factors["cls_activation"] = f"NEUTRAL (cls_norm={cls_norm:.2f}, normal range)"

        # Clamp trust to [0.0, 1.0]
        trust = max(0.0, min(1.0, trust))

        factors["final_trust"] = f"{trust:.3f}"

        return trust, factors

    def _generate_explanation(
        self, prediction, p_vit, trust_score, p_final, confidence,
        entropy, margin, agreement, cls_norm,
        rgb_attn, fft_attn, noise_attn, trust_factors
    ):
        """
        Generate a human-readable forensic explanation.

        Returns:
            str: multi-line explanation of the analysis
        """
        lines = []

        # Header
        pred_emoji = "🚨 FAKE" if prediction == "fake" else "✅ REAL"
        lines.append(f"{'='*60}")
        lines.append(f"  FORENSIC ANALYSIS REPORT")
        lines.append(f"{'='*60}")
        lines.append(f"")
        lines.append(f"  Verdict: {pred_emoji} (Confidence: {confidence:.1f}%)")
        lines.append(f"")

        # Probability breakdown
        lines.append(f"  Probability Breakdown:")
        lines.append(f"    • ViT Raw Prediction (P_vit):  {p_vit:.4f}")
        lines.append(f"    • Agent Trust Score:            {trust_score:.4f}")
        lines.append(f"    • Calibrated Probability:       {p_final:.4f}")
        lines.append(f"")

        # Modality analysis
        lines.append(f"  Modality Attention Analysis:")
        dominant_modality = max(
            [("RGB (visual)", rgb_attn), ("FFT (frequency)", fft_attn), ("Noise (residual)", noise_attn)],
            key=lambda x: x[1]
        )
        lines.append(f"    • RGB  (visual features):   {rgb_attn:.3f}")
        lines.append(f"    • FFT  (frequency domain):  {fft_attn:.3f}")
        lines.append(f"    • Noise (residual pattern):  {noise_attn:.3f}")
        lines.append(f"    → Dominant modality: {dominant_modality[0]}")
        lines.append(f"    → Modality agreement: {agreement:.3f}")
        lines.append(f"")

        # Forensic reasoning
        lines.append(f"  Forensic Reasoning:")

        if prediction == "fake":
            if fft_attn > rgb_attn and fft_attn > noise_attn:
                lines.append(f"    • The model heavily relied on FREQUENCY DOMAIN analysis.")
                lines.append(f"      This suggests GAN-generated artifacts in the spectral domain,")
                lines.append(f"      which are invisible to the human eye but detectable by FFT.")
            elif noise_attn > rgb_attn and noise_attn > fft_attn:
                lines.append(f"    • The model heavily relied on NOISE RESIDUAL analysis.")
                lines.append(f"      This suggests synthetic noise patterns inconsistent with")
                lines.append(f"      real camera sensor noise, typical of generated images.")
            else:
                lines.append(f"    • The model primarily used VISUAL (RGB) features.")
                lines.append(f"      Visible artifacts like unnatural skin texture, lighting")
                lines.append(f"      inconsistencies, or boundary artifacts were detected.")
        else:
            lines.append(f"    • The image shows consistent patterns across all modalities.")
            lines.append(f"      Natural camera noise, expected frequency distribution, and")
            lines.append(f"      authentic visual features all align with a real photograph.")

        lines.append(f"")

        # Trust analysis
        lines.append(f"  Trust Score Breakdown:")
        for factor_name, factor_detail in trust_factors.items():
            if factor_name != "final_trust":
                lines.append(f"    • {factor_name}: {factor_detail}")

        # Confidence assessment
        lines.append(f"")
        if confidence > 90:
            lines.append(f"  Assessment: HIGH CONFIDENCE — Result is highly reliable.")
        elif confidence > 70:
            lines.append(f"  Assessment: MODERATE CONFIDENCE — Result is reasonably reliable.")
        elif confidence > 50:
            lines.append(f"  Assessment: LOW CONFIDENCE — Result should be verified manually.")
        else:
            lines.append(f"  Assessment: VERY LOW CONFIDENCE — Result is uncertain, manual review needed.")

        lines.append(f"{'='*60}")

        return "\n".join(lines)
