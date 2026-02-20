import os
import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

def check_gpu():
    print("GPU Diagnostics")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Number of GPUs found: {device_count}")
        print(f"Active GPU: {torch.cuda.get_device_name(0)}")
        print("-----------------------")
        return "cuda"
    else:
        print("WARNING: CUDA not found! PyTorch will use the CPU.")
        print("-----------------------")
        return "cpu"
    
def main():
    
    device = check_gpu()
    
    if device == "cpu":
        print("Exiting to prevent CPU training. Please fix your PyTorch CUDA installation.")
        return
    
    # check gpu availability (make sure its not using the cpu)
    print("Checking GPU availability...")
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not found.")

    HOME = os.getcwd()
    print(f"Working directory set to: {HOME}")

    # roboflow dataset download
    print("Downloading dataset...")
    rf = Roboflow(api_key="1cvfgr6GI71LB8z4XPor")
    project = rf.workspace("wawan-pradana").project("cinta_v2")
    dataset = project.version(1).download("yolov5")

    print("Fixing Windows pathing in data.yaml...")
    data_yaml_path = os.path.join(dataset.location, "data.yaml")
    
    with open(data_yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
        
    # force yolo to be good and use the right dir
    yaml_data['path'] = dataset.location 
    
    # fix for roboflow nesting issue
    yaml_data['train'] = "train/images"
    yaml_data['val'] = "valid/images"
    yaml_data['test'] = "test/images"
    
    with open(data_yaml_path, 'w') as file:
        yaml.dump(yaml_data, file, sort_keys=False)
    
    # init yolov8
    print("Initializing YOLOv8 model...")
    model = YOLO("yolov8l.pt") 

    # training
    print("Starting training...")
    data_yaml_path = os.path.join(dataset.location, "data.yaml")
    
    # train using python api
    # pls keep workers at 0, you donut. windows will freeze otherwise
    results = model.train(
        data=data_yaml_path,
        epochs=80,
        imgsz=640,
        project="TrafficLightRuns",
        name="train_run",
        workers=0 
    )

    print("Validating model against test data...")
    metrics = model.val()

    print("Running inference on a test video...")
    test_video = "video1.mp4" 
    
    if os.path.exists(test_video):
        prediction_results = model.predict(
            source=test_video,
            conf=0.45,
            save=True,
            project="UAV_VIP Traffic_Light_Detection",
            name="inference_run"
        )
        print(f"Done! Check the 'UAV_VIP Traffic_Light_Detection/inference_run' folder for your video.")
    else:
        print(f"Could not find '{test_video}'. Drop an mp4 file in your VS Code folder to test it.")

if __name__ == '__main__':
    main()