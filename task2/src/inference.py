import cv2
import pickle
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np


def reconstruct_keypoints(kp_data):
    return [
        cv2.KeyPoint(
            x=pt[0],
            y=pt[1],
            size=pt[2],
            angle=pt[3],
            response=pt[4],
            octave=int(pt[5]),
            class_id=int(pt[6])
        )
        for pt in kp_data
    ]



def draw_keypoints(image_path, keypoints):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Could not read {image_path}")
        return

    img_with_kp = cv2.drawKeypoints(
        img, keypoints, None, color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
    plt.title(f"With Keypoints ({len(keypoints)})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()



def match_descriptors(desc1, desc2, num_matches):
    matcher = cv2.BFMatcher()
    matches = matcher.match(desc1, desc2, None)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches[:num_matches]


def visualize_matches(img1_path, img2_path, kp1, kp2, good_matches):
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        print("Error reading one of the images.")
        return

    matched_img = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.title(f"{os.path.basename(img1_path)} vs {os.path.basename(img2_path)} | Matches: {len(good_matches)}")
    plt.axis("off")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare images using saved ORB features.")
    parser.add_argument("--features", default="features.pkl", help="Path to saved features file.")
    parser.add_argument("--img1", required=True, help="Path to first image (reference).")
    parser.add_argument("--img2", help="Path to second image (optional). If omitted, compares img1 with all others.")
    parser.add_argument("--num_matches", type=float, default=20, help="Number of matches to keep.")
    args = parser.parse_args()

    with open(args.features, "rb") as f:
        features = pickle.load(f)

    image_name = os.path.basename(args.img1)
    if image_name not in features:
        print(f"{image_name} not found in features. Run train.py first.")
        return

    data1 = features[image_name]
    kp1 = reconstruct_keypoints(data1["keypoints"])
    desc1 = np.array(data1["descriptors"], dtype=np.uint8)

    draw_keypoints(args.img1, kp1)

    if args.img2:
        image2_name = os.path.basename(args.img2)
        if image2_name not in features:
            print(f"{image2_name} not found in features.")
            return

        data2 = features[image2_name]
        kp2 = reconstruct_keypoints(data2["keypoints"])
        desc2 = np.array(data2["descriptors"], dtype=np.uint8)

        draw_keypoints(args.img2, kp2)
        good = match_descriptors(desc1, desc2, args.num_matches)
        print(f"Matched {len(good)} good features between selected images.")
        visualize_matches(args.img1, args.img2, kp1, kp2, good)

    else:
        for other_name, data2 in features.items():
            if other_name == image_name:
                continue

            kp2 = reconstruct_keypoints(data2["keypoints"])
            desc2 = np.array(data2["descriptors"], dtype=np.uint8)
            good = match_descriptors(desc1, desc2, args.num_matches)
            print(f"{image_name} vs {other_name}: {len(good)} good matches.")
            visualize_matches(args.img1, os.path.join(os.path.dirname(args.img1), other_name), kp1, kp2, good)


if __name__ == "__main__":
    main()
