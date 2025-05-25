{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "44cd245a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# F1 Score achieved  on Kaggle : 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "583fdd42",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import os\n",
    "import pandas as pd\n",
    "from torchvision import transforms, models\n",
    "from PIL import Image\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "import json"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aa416fca",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load label map\n",
    "with open(\"/content/drive/MyDrive/soil-classification/label_map.json\", 'r') as f:\n",
    "    idx2label = json.load(f)\n",
    "idx2label = {int(k): v for k, v in idx2label.items()}  # Ensure keys are ints"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "06917830",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load model\n",
    "model = models.resnet18(pretrained=False)\n",
    "model.fc = torch.nn.Linear(model.fc.in_features, len(idx2label))\n",
    "model.load_state_dict(torch.load(\"/content/drive/MyDrive/soil-classification/resnet18_soil.pth\"))\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "model = model.to(device)\n",
    "model.eval()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d77fc6ee",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define test transform\n",
    "transform_test = transforms.Compose([\n",
    "    transforms.Resize((224, 224), interpolation=Image.BICUBIC),\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize([0.5]*3, [0.5]*3)\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d6049982",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# Dataset class\n",
    "class SoilDataset(Dataset):\n",
    "    def __init__(self, df, transform=None):\n",
    "        self.df = df.reset_index(drop=True)\n",
    "        self.transform = transform\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.df)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        row = self.df.iloc[idx]\n",
    "        image = Image.open(row['filename']).convert('RGB')\n",
    "        if self.transform:\n",
    "            image = self.transform(image)\n",
    "        return image, row['image_id']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1e148a85",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prepare test DataFrame\n",
    "TEST_DIR = \"/content/drive/MyDrive/soil-classification/soil_classification-2025/test/\"\n",
    "test_df = pd.read_csv(\"/content/drive/MyDrive/soil-classification/soil_classification-2025/test_ids.csv\")\n",
    "test_df['filename'] = test_df['image_id'].apply(lambda x: os.path.join(TEST_DIR, x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1ed8f1c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Inference\n",
    "test_dataset = SoilDataset(test_df, transform=transform_test)\n",
    "test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)\n",
    "\n",
    "predictions = []\n",
    "image_ids = []\n",
    "\n",
    "with torch.no_grad():\n",
    "    for images, ids in test_loader:\n",
    "        images = images.to(device)\n",
    "        outputs = model(images)\n",
    "        preds = outputs.argmax(1).cpu().numpy()\n",
    "        predictions.extend(preds)\n",
    "        image_ids.extend(ids)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d0b76a2f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save submission\n",
    "final_df = pd.DataFrame({\n",
    "    \"image_id\": image_ids,\n",
    "    \"soil_type\": [idx2label[i] for i in predictions]\n",
    "})\n",
    "final_df.to_csv(\"submission.csv\", index=False)\n",
    "print(\" submission.csv generated\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ae0c6ccc",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
