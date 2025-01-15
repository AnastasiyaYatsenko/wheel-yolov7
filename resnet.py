import time

import torch
from PIL import Image
import os
import shutil
from torchvision import models, transforms
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_COLUMNS = ['1', '10', '100', '2', '200', '300', '5', '8']

def predict_image(model, image_path, transform, device):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        preds = torch.sigmoid(outputs).cpu().numpy()[0]

        predicted_class = torch.argmax(torch.tensor(preds)).item()

    return predicted_class


def load_model_weights(model, weights_path, device):
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def eval_sectors():
    # BASE_DIR = "8"
    BASE_DIR = "wheel/sectors"
    BAD_DIR = os.path.join(BASE_DIR, "bad")

    if not os.path.exists(BAD_DIR):
        os.makedirs(BAD_DIR)

    weights_path = "best_model.pth"

    t0 = time.time()

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, len(LABEL_COLUMNS)),
        torch.nn.Sigmoid()
    )

    model = load_model_weights(model, weights_path, DEVICE)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    file_count = sum(len(files) for _, _, files in os.walk(BASE_DIR))  # Get the number of files
    with tqdm(total=file_count) as pbar:
        for subdir, dirs, files in tqdm(os.walk(BASE_DIR)):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(subdir, file)
                    folder_name = os.path.basename(subdir)
                    predicted_class = predict_image(model, image_path, transform, DEVICE)

                    if int(LABEL_COLUMNS[predicted_class]) != int(folder_name):
                        print(
                            f"Переміщую картинку {file} у папку 'bad'... (Передбачений клас: {predicted_class}, Папка: {folder_name})")
                        shutil.move(image_path, os.path.join(BAD_DIR, file))
                    pbar.update(1)
                    # else:
                    #     print("Ok")
    print(f'Done evaluating.({time.time() - t0:.3f}s)')
