import torch
import torch.nn as nn
import torch.nn.functional as F

class VitExtractor(nn.Module):
    def __init__(self, model_name, device, usev2=False):
        super().__init__()
        
        if usev2:
            self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(device)
        else:
            self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)
        self.model.eval()
        self.last_block = None
        self.feature_output = None
        self.last_block = self.model.blocks[-1]
        self.last_block.register_forward_hook(self._get_block_hook())

    def _get_block_hook(self):
        def _get_block_output(model, input, output):
            self.feature_output = output
        return _get_block_output
    
    def get_vit_feature(self, input_img):
        mean = torch.tensor([0.485, 0.456, 0.406], device=input_img.device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=input_img.device).reshape(1, 3, 1, 1)
        input_img = (input_img - mean) / std
        self.model(input_img)
        return self.feature_output

if __name__ == "__main__":
    import cv2
    import numpy as np
    from sklearn.decomposition import PCA

    device = 'cuda:3'
    usev2 = False
    if usev2:
        dino = VitExtractor(model_name='dinov2_vits14', device=device)
        dimension = 384
        patch_size = 14
    else:
        dino = VitExtractor(model_name='dino_vits16', device=device, usev2=usev2)
        dimension = 384
        patch_size = 16

    def run_vitFeat_PCA_single_image(img_path, save_path):
        if type(img_path) != str:
            img = img_path
        else:
            img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).to(device).float()
        img = img.unsqueeze(0).permute(0, 3, 1, 2) / 255.
        img = F.interpolate(img, size=(224*5, 224*5))

        B, C, H, W = img.shape

        pca = PCA(n_components=3)

        if usev2:
            dino_ret = dino.model.forward_features(img)['x_norm_patchtokens']
        else:
            dino_ret = dino.get_vit_feature(img)[:, 1:, :]
        dino_ret = dino_ret.reshape([1, H//patch_size, W//patch_size, dimension])
        dino_ret = dino_ret.squeeze(0).detach().cpu().numpy()
        dino_ret = dino_ret.reshape([-1, dimension])

        component = pca.fit_transform(dino_ret)
        component = component.reshape([H//patch_size, W//patch_size, 3])
        component = ((component - component.min()) / (component.max() - component.min())).astype(np.float32) # normalize to [0,1]
        component *= 255.
        component = component.astype(np.uint8)

        if save_path is not None:
            cv2.imwrite(save_path, component.squeeze())
        else:
            return component

    with torch.no_grad():
        run_vitFeat_PCA_single_image('/mnt/sda/Projects/Hierarchy-Clip/images/desk_crop2.jpg', '/mnt/sda/Projects/Hierarchy-Clip/logs/dino_desk_crop2.png')

  