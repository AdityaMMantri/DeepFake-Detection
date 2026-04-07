import numpy as np


def predict(model, image, model_type):
    """
    Unified prediction interface

    Args:
        model: loaded model system
        image: PIL Image
        model_type: "cnn" | "vit_9ch" | "vit_single"

    Returns:
        dict with standardized output
    """

    # Convert PIL → numpy (ALL models expect numpy)
    image_np = np.array(image)

    # ================= CNN =================
    if model_type == "cnn":
        result = model.predict(image_np)

        return {
            "model": "CNN",
            "prediction": result["prediction"],
            "confidence": float(result["confidence"]),
            "p_fake": float(result["p_fake"]),
            "extra": {}
        }

    # ================= ViT SINGLE =================
    elif model_type == "vit_single":
        result = model.predict(image_np)

        return {
            "model": "ViT Single",
            "prediction": result["prediction"],
            "confidence": float(result["confidence"]),
            "p_fake": float(result["p_fake"]),
            "extra": {}
        }

    # ================= ViT 9 CHANNEL =================
    elif model_type == "vit_9ch":
        result = model.predict(image_np)

        # 🔴 This model returns richer info
        return {
            "model": "ViT 9ch",
            "prediction": result.get("prediction", "unknown"),
            "confidence": float(result.get("confidence", 0.0)),
            "p_fake": float(result.get("p_final", 0.0)),
            "extra": {
                "p_vit": result.get("p_vit"),
                "trust_score": result.get("trust_score"),
                "explanation": result.get("explanation")
            }
        }

    else:
        raise ValueError(f"Invalid model_type: {model_type}")