from ultralytics import YOLO
import os


if __name__ == "__main__":
    ##detect##
    # model source guide/remote-home/lhkun/hnan/yolov8_lite/ultralytics-main/ultralytics/cfg/models/v8/myyolov8n-SPD-GFPN-DFEM-r16.yaml
    model_yaml = "/cfg/models/v8/myyolov8n-SPD-GFPN-DFEM-r16.yaml"
    # model_yaml = '/remote-home/hnan/yolov8_copy/ultralytics-main/ultralytics/cfg/models/v8/yolov8n-up2-detector.yaml'
    # data source guide
    data_yaml = "/cfg/datasets/NUDT.yaml"
    # pretrained weight
    # pre_model = r"D:\wj-bk1\ultralytics-main\ultralytics-main\ultralytics-main\runs\detect\train23\weights\best.pt"
    # Load a model
    # model = YOLO(model_yaml, task='detect').load(pre_model)  # build from YAML and transfer weights
    model = YOLO(model_yaml, task='detect')
    # Train the model    default train settings in D:\pytorch\ultralytics-main\ultralytics-main\ultralytics\cfg\default.yaml
    results = model.train(data=data_yaml, cache=False, epochs=, imgsz=, batch=, close_mosaic=,workers=,device=1,name='TISD_final',resume=False,amp=False)
    ##detect##

