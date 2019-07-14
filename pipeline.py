from torchvision import transforms
import cv2
import torch
from PIL import Image as PIL_Image
import torch.nn as nn
from models.configTrain import configTrain

default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((configTrain.img_weight, configTrain.img_height)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((configTrain.img_weight, configTrain.img_height)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((configTrain.img_weight, configTrain.img_height)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    generate a new face bounding box.
    param:
        face: dlib face class
        width: frame width
        height: frame height
        scale: scale to enlarge original face
        minsize: set minimum bounding box size
    return:
        x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def preprocess_image(image, cuda=True):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.
    param:
        image: numpy image in opencv form
    return:
        pytorch tensor of shape [1, 3, image_size, image_size]
    """
    # revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess using the preprocessing function used during training and casting it to PIL image
    preprocess = default_data_transforms['test']
    preprocessed_image = preprocess(PIL_Image.fromarray(image))

    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    else:
        preprocessed_image = preprocessed_image.cpu()
    return preprocessed_image


def predict_with_model(image, model, post_function=nn.Softmax(dim=1),
                       cuda=True):
    """
    predicts the label of an input image and preprocesses the input image
    param:
        image: numpy image
        model: torch model with linear layer at the end
        post_function: e.g., softmax
        cuda: enables cuda, must be the same parameter as the model
    return:
        prediction (0 = fake, 1 = real)
    """
    # preprocess
    preprocessed_image = preprocess_image(image, cuda)

    # model prediction
    output = model(preprocessed_image)
    output = post_function(output)

    # cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output