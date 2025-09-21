import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



def get_sample_input(dataset_directory):
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
    val_dataset = datasets.ImageFolder(os.path.join(dataset_directory,'val'),data_transform)
    dataloader = DataLoader(val_dataset,batch_size=1,shuffle=True)
    class_names = val_dataset.classes
    num_classes = len(class_names)
    sample_input_tensor,_ = next(iter(dataloader))
    return sample_input_tensor, num_classes


