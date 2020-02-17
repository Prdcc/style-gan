from torchvision import transforms
import torch as nn

original_test = nn.load("data/28/test.pt")
original_training = nn.load("data/28/training.pt")

def resize(resolution,imgs,train=True):
    resizing_fun = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((resolution,resolution)),
        transforms.ToTensor()
    ])
    path = "data/%d/"%(resolution)
    path += "training.pt" if train else "test.pt"
    trans_imgs = [resizing_fun(img) for img in imgs]
    trans_imgs_tensor = nn.Tensor(len(trans_imgs), resolution, resolution)
    nn.cat(trans_imgs, out=trans_imgs_tensor)
    nn.save(trans_imgs_tensor,path)

resize(14,original_training)
resize(7,original_training)
resize(14,original_test,False)
resize(7,original_test,False)