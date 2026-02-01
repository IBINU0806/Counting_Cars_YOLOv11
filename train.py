import argparse
from ultralytics import YOLO
import os


def get_args():
    parser = argparse.ArgumentParser(description="Traffic Counting Training with YOLOv11")
    parser.add_argument("-d", "--data", type=str, default="data.yaml", help="Path to data.yaml file")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("-i", "--img-size", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("-m", "--model", type=str, default="yolo11n.pt", help="Base model (yolo11n.pt, yolo11s.pt...)")
    parser.add_argument("-l", "--lr", type=float, default=0.01, help="Initial learning rate (lr0)")
    parser.add_argument("-p", "--project", type=str, default="runs/train", help="Project name (path to save results)")
    parser.add_argument("-n", "--name", type=str, default="exp", help="Experiment name")
    parser.add_argument("-r", "--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("-dev", "--device", type=str, default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    args = parser.parse_args()
    return args


def train(args):
    print(f"Bắt đầu training với model: {args.model}")
    print(f"Dữ liệu: {args.data}")
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device,
        lr0=args.lr,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=True,
        save=True,
        val=True,
        resume = args.resume
    )

    print(f"Kết quả được lưu tại: {os.path.join(args.project, args.name)}")
    print(f"Best weights: {os.path.join(args.project, args.name, 'weights', 'best.pt')}")


if __name__ == "__main__":
    args = get_args()
    train(args)
    #python train_yolo.py
    #python train_yolo.py --model "runs/train/exp/weights/last.pt" -r
    #yolo train resume model="runs/train/exp/weights/last.pt"