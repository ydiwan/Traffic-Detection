import os
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

def main():
    print("1. Downloading 'cinta_v2' Dataset...")
    
    api_key = os.getenv("ROBOFLOW_API_KEY")
    
    # --- PASTE YOUR ROBOFLOW SNIPPET HERE ---
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("YOUR_WORKSPACE_NAME").project("cinta_v2")
    version = project.version(1) # Change the version number if needed
    dataset = version.download("yolov8")
    # ----------------------------------------

    print(f"Dataset downloaded to: {dataset.location}")
    print("2. Initializing YOLOv8 Nano...")
    
    # Start with the foundational Nano weights
    model = YOLO("yolov8n.pt") 

    print("3. Starting Training on RTX 4070...")
    
    # Run the training loop
    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=50,       # 50 epochs is perfect for a fast, overnight/demo model
        imgsz=640,       # Standard training resolution
        batch=16,        # Your 4070 Laptop GPU can easily handle a batch size of 16
        device=0,        # CRITICAL: Forces Ubuntu to use your NVIDIA GPU
        project="TrafficLightModels",
        name="nano_run",
        workers=8        # Speeds up data loading using your CPU cores
    )

    print("\n--- Training Complete! ---")
    print("Your custom weights are saved at: TrafficLightModels/nano_run/weights/best.pt")

if __name__ == '__main__':
    main()