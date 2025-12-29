import torch
import matplotlib.pyplot as plt 
import torch.nn.functional as F


def visualize_attention_map(image, attn_map, attention_heatmap_path, query_pos=(32, 32)):
    """
    inputs:
        images: [3, H, W] Tensor
        attn_map: [N, N] Attention Map
        query_pos: (y, x) 기준점 좌표
    """
    H, W = image.shape[1], image.shape[2]
    y, x = query_pos
    idx = y * W + x 
    map_size = int(attn_map.shape[0] ** 0.5)
    scale = H // map_size 

    # 특정 위치의 attention 행 추출 및 리사이즈 
    mask = attn_map[idx].reshape(H, W)

    # 가시성을 위해 정규화 진행 
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.numpy()

    # Visualize 
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np + 1) / 2   # -1~1 -> 0~1 변환 

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img_np)
    ax[0].scatter(x * scale, y * scale, c="red", edgecolors="white")
    ax[0].set_title("생성된 이미지")
    
    mask_resized = F.interpolate(
        torch.tensor(mask).unsqueeze(0).unsqueeze(0),
        size=(H, W), mode="bilinear"
    ).squeeze().numpy()

    ax[1].imshow(img_np)
    ax[1].imshow(mask_resized, cmap="jet", alpha=0.5)
    ax[1].set_title("Self Attention 히트맵")

    plt.savefig(attention_heatmap_path)
    plt.close()