import os
import cv2
import numpy as np
import argparse
import pickle

def keypoints_to_list(keypoints):
    return [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            for kp in keypoints]


def extract_features(folder, max_features=200):
    orb = cv2.ORB_create(nfeatures=max_features)
    features = {}

    image_paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
    ]

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping {path} (could not read)")
            continue

        keypoints, descriptors = orb.detectAndCompute(img, None)
        if descriptors is not None:
            features[os.path.basename(path)] = {
                "keypoints": keypoints_to_list(keypoints),
                "descriptors": descriptors.tolist()
            }
            print(f"Extracted {len(keypoints)} keypoints from {os.path.basename(path)}")
        else:
            print(f"No features found in {os.path.basename(path)}")

    return features


def save_features(features, output_path="features.pkl"):
    with open(output_path, "wb") as f:
        pickle.dump(features, f)
    print(f"Saved features to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save ORB keypoints from images.")
    parser.add_argument("--folder", required=True, help="Path to folder containing images.")
    parser.add_argument("--max_features", type=int, default=400, help="Maximum ORB features per image.")
    parser.add_argument("--output", default="features.pkl", help="Output file for saved features.")
    args = parser.parse_args()

    features = extract_features(args.folder, args.max_features)
    save_features(features, args.output)
