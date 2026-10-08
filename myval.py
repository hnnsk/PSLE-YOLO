from ultralytics import YOLO
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":

    pth_path = "/remote-home//yolov8_lite/ultralytics-main/runs/detect/NUDT/weights/best.pt"
    model = YOLO(pth_path)

    metrics = model.val(
        data='/remote-home//yolov8_lite/ultralytics-main/ultralytics/cfg/datasets/NUDT.yaml',
        split='val',   # 或 val
        name='triple2_val',  # 结果保存目录名
    )

    map_list = [metrics.box.map50, metrics.box.map75, metrics.box.map]
    formatted_numbers = [format(num, '.3f') for num in map_list]
    print(formatted_numbers)
    print("size bucket metrics:")
    for size_name in ("small", "medium", "large"):
        result = getattr(metrics, "size_metrics", {}).get(size_name, {})
        print(
            f"{size_name}: "
            f"images={result.get('images', 0)}, "
            f"instances={result.get('instances', 0)}, "
            f"P={result.get('precision', 0.0):.3f}, "
            f"R={result.get('recall', 0.0):.3f}, "
            f"AP50={result.get('ap50', 0.0):.3f}, "
            f"AP75={result.get('ap75', 0.0):.3f}, "
            f"mAP50-95={result.get('map5095', result.get('ap', 0.0)):.3f}"
        )











