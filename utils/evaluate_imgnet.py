from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast


def create_imagenet_test_loader(test_path, batch_size=256, num_workers=4):
    test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    
    test_dataset = datasets.ImageNet(root=test_path, split='val', transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader

def evaluate_model(model, test_path, batch_size=512, num_workers=4, device='cuda'):
    
    test_loader = create_imagenet_test_loader(test_path, batch_size, num_workers)
    model = model.to(device).half()
    model.eval()  
    correct = 0
    total = 0
    
    with torch.no_grad(): 
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device).half(), labels.to(device)
            
            with autocast():
                # outputs = model.classifier(model.feature_extractor(images)) 
                outputs = model.classifier(model(images))
            _, predicted = torch.max(outputs, 1)  # Get the class with highest probability
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy