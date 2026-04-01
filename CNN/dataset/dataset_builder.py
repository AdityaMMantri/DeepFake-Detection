import os


def build_dataset(root_dir):
    """
    Builds image path list and labels.

    Expected folder structure:

    root_dir/
        real/
        fake/
    """

    image_paths = []
    labels = []

    classes = ["real", "fake"]

    for label, cls in enumerate(classes):

        cls_path = os.path.join(root_dir, cls)

        for img_name in os.listdir(cls_path):

            if not img_name.lower().endswith(
                (".jpg",".jpeg",".png",".bmp")
            ):
                continue

            img_path = os.path.join(cls_path, img_name)

            image_paths.append(img_path)
            labels.append(label)

    return image_paths, labels