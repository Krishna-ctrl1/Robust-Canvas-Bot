import torch
import torch.nn as nn
import numpy as np
import cv2
import os

class enhance_net_nopool(nn.Module):
    """
    Exact Zero-DCE Neural Architecture.
    Matches the official Li-Chongyi PyTorch implementation structures to load .pth weights.
    """
    def __init__(self):
        super(enhance_net_nopool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True) 
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True) 
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True) 
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2, number_f, 3, 1, 1, bias=True) 
        self.e_conv6 = nn.Conv2d(number_f*2, number_f, 3, 1, 1, bias=True) 
        self.e_conv7 = nn.Conv2d(number_f*2, 24, 3, 1, 1, bias=True) 
        
    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        
        # Iterative curve application
        x = x + r1*(torch.pow(x,2)-x)
        x = x + r2*(torch.pow(x,2)-x)
        x = x + r3*(torch.pow(x,2)-x)
        enhance_image_1 = x + r4*(torch.pow(x,2)-x)
        x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)
        x = x + r6*(torch.pow(x,2)-x)
        x = x + r7*(torch.pow(x,2)-x)
        enhance_image_2 = x + r8*(torch.pow(x,2)-x)
        
        return enhance_image_2

def enhance_lowlight(image):
    if image is None:
        raise ValueError("Image is None")
        
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    model = enhance_net_nopool()
    weight_path = os.path.join(os.path.dirname(__file__), "../../../assets/Epoch99.pth")
    
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    else:
        print("Warning: Real Epoch99.pth not found. Generating empirical heuristic fallback.")
    
    model.eval()
    
    with torch.no_grad():
        if os.path.exists(weight_path):
            enhanced_tensor = model(img_tensor)
            enhanced = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            enhanced = (enhanced * 255).astype(np.uint8)
            return cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        else:
            # Fallback heuristic to guarantee execution if wget failed
            gamma = 0.4
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)
