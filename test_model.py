import os
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights


def init_function(resnet_layers=18):
    # Function should return a torch.nn.module
    if resnet_layers == 18:
        resnet = resnet18(weights=None)
    elif resnet_layers == 50:
        resnet = resnet50(weights=None)
    return resnet


def preprocess(data):
    transforms = ResNet18_Weights.DEFAULT.transforms()
    return transforms(data)


def postprocess(data):
    prediction = data.squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = ResNet18_Weights.DEFAULT.meta["categories"][class_id]
    return score, class_id, category_name


def install_packages():
    os.system("echo 'installing packages'")


def unpack_dependencies():
    os.system("echo 'unpacking dependencies'")
