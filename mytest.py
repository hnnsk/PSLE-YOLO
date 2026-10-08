from ultralytics import YOLO

if __name__ == "__main__":
    pth_path = "/attached/remote-home2//ultralytics-main/ultralytics-main/runs/detect/TISD_ours/weights/best.pt"

    test_path = "/attached/remote-home2//ultralytics-main/dataset/TISD_for_train/images/test"

    model = YOLO(pth_path)  # load a custom model

    # Predict with the model
    results = model(test_path, save=True, conf=, name='TISD_ours_test', save_txt=True, line_width=1)  # predict image (not neccessory size 640)

