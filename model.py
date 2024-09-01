import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 5, padding=2)
        
    def forward(self, x):
        x0 = F.relu(self.conv1(x))  # [batch_size, 64, H, W]
        
        x1 = F.max_pool2d(x0, 2, padding=0)  # [batch_size, 64, H/2, W/2]
        x1 = F.relu(self.conv2(x1))  # [batch_size, 32, H/2, W/2]
        
        x2 = F.max_pool2d(x1, 2, padding=0)  # [batch_size, 32, H/4, W/4]
        x2 = F.relu(self.conv3(x2))  # [batch_size, 16, H/4, W/4]
        
        x3 = F.max_pool2d(x2, 2, padding=0)  # [batch_size, 16, H/8, W/8]
        
        return x0, x1, x2, x3

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv0 = nn.ConvTranspose2d(16, 16, 7, padding=3)
        self.conv1 = nn.ConvTranspose2d(32, 16, 7, padding=3)  # 수정: 출력 채널을 16으로 변경
        self.conv2 = nn.ConvTranspose2d(48, 32, 3, padding=1)  # 수정: 입력 채널을 48로 변경
        self.conv3 = nn.ConvTranspose2d(96, 1, 3, padding=1)   # 수정: 입력 채널을 96으로 유지

    def forward(self, vals):
        x0, x1, x2, x3 = vals
        
        x = F.relu(self.conv0(x3))  # [batch_size, 16, H/8, W/8]
        x = F.interpolate(x, size=x2.size()[2:], mode='nearest')  # [batch_size, 16, H/4, W/4]
        x = torch.cat([x, x2], dim=1)  # [batch_size, 32, H/4, W/4]
        
        x = F.relu(self.conv1(x))  # [batch_size, 16, H/4, W/4]
        x = F.interpolate(x, size=x1.size()[2:], mode='nearest')  # [batch_size, 16, H/2, W/2]
        x = torch.cat([x, x1], dim=1)  # [batch_size, 48, H/2, W/2]
        
        x = F.relu(self.conv2(x))  # [batch_size, 32, H/2, W/2]
        x = F.interpolate(x, size=x0.size()[2:], mode='nearest')  # [batch_size, 32, H, W]
        x = torch.cat([x, x0], dim=1)  # [batch_size, 96, H, W]
        
        x = F.leaky_relu(self.conv3(x))  # [batch_size, 1, H, W]
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        vals = self.encoder(x)
        x = self.decoder(vals)
        return x
    
def test_autoencoder():
    # 임의의 입력 데이터 생성
    batch_size = 1
    height, width = 820, 821  # 데이터셋의 크기와 맞춰줌
    input_data = torch.randn(batch_size, 3, height, width)  # (batch_size, channels, height, width)

    # Autoencoder 모델 생성
    model = Autoencoder()

    # 모델의 출력 생성
    output_data = model(input_data)

    # 결과 출력
    print("Input shape: ", input_data.shape)  # (batch_size, 3, 820, 821)
    print("Output shape: ", output_data.shape)  # (batch_size, 1, 820, 821)

    # 모델의 파라미터 수 확인
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

# 테스트 함수 실행
# test_autoencoder()