"""
test.py — Evaluate Best Model on Test Dataset
=============================================
Runs inference using predict.py's functions on all videos in dataset/test.
"""

import os
import glob
import time
# We ONLY need to import predict_video now
from predict import predict_video

def main():
    model_path = "checkpoints/best_model.pth"
    test_dir = "dataset/test"
    
    print("========================================")
    print("  TEST DATASET EVALUATION")
    print("========================================")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}.")
        return
        
    print(f"Using model weights from {model_path}...")
    
    classes = ["NonViolence", "Violence"]
    correct = 0
    total = 0
    
    col = [30, 15, 15, 7]
    print(f"\n  {'File':<{col[0]}} {'True':<{col[1]}} {'Predicted':<{col[2]}} {'Conf':>{col[3]}}")
    print("  " + "─" * (sum(col)+5))
    
    start_time = time.time()
    
    for cls in classes:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"  [WARN] Test directory {cls_dir} not found.")
            continue
            
        # Get all mp4 and avi videos
        videos = sorted(glob.glob(os.path.join(cls_dir, "*.mp4")) + glob.glob(os.path.join(cls_dir, "*.avi")))
        
        for video_path in videos:
            # FIXED CALL: Pass the model path string. 
            # It returns two values: label string ("Violence"/"NonViolence") and confidence float.
            pred_label, conf = predict_video(video_path, model_path=model_path, threshold=0.5)
            
            true_label = "Violence" if cls == "Violence" else "NonViolence"
            
            if pred_label == true_label:
                correct += 1
                mark = "OK"
            else:
                mark = "WRONG"
                
            total += 1
            print(f"  {os.path.basename(video_path):<{col[0]}} {cls:<{col[1]}} {pred_label:<{col[2]}} {conf:>{col[3]}.3f}  {mark}")

    total_time = time.time() - start_time
    
    print(f"\n========================================")
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"  Result : {correct}/{total} correct ({accuracy:.2f}%)")
        print(f"  Time   : {total_time:.1f}s")
    else:
        print("  No videos found in the test dataset.")
    print(f"========================================")

if __name__ == "__main__":
    main()