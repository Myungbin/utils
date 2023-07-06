import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms


def gard_cam(image_path, model, target_layers, transform, reshape=None):
    """입력 이미지의 Grad-CAM(Gradient-weighted Class Activation Mapping)을 계산합니다.

     Args:
         image_path (str): 입력 이미지 파일 경로 입니다.
         model (torch.nn.Module): 학습된 모델 입니다.
         target_layers (list): Grad-CAM을 계산할 레이어 입니다.
         transform (transforms.Compose): 입력 이미지를 전처리하기 위한 변환 함수 입니다.
         reshape (optional): Grad-CAM 결과를 재구성하는 데 사용할 변환입니다. Vision Transformer 모델을 사용시 사용됩니다.
     """

    model.eval()

    image = Image.open(image_path)
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)

    if reshape is not None:
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    else:
        cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(1)]

    grayscale_cam = cam(input_batch, targets)
    grayscale_cam = grayscale_cam[0, :]
    img_pil = transforms.ToPILImage()(input_tensor.squeeze().cpu())

    plt.imshow(img_pil)
    plt.imshow(grayscale_cam, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()


def reshape_transform(tensor, height=16, width=16):
    # height, width = patch size
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result
