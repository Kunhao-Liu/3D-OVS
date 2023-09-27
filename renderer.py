import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from models.tensoRF import raw2alpha, TensorVMSplit, AlphaGridMask
from funcs import *
from dataLoader.ray_utils import ndc_rays_blender
from sklearn.decomposition import PCA
from sklearn.metrics import jaccard_score, accuracy_score
from torchvision.utils import save_image
import clip


def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, 
                                render_feature=False, out_rgb=False, device='cuda'):

    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    features, selects = [], []

    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)

        if render_feature:
            feature_map, select_map, rgb_map = tensorf.render_feature_map(rays_chunk, out_rgb=out_rgb, is_train=is_train, ndc_ray=ndc_ray, N_samples=N_samples)
            features.append(feature_map)
            selects.append(select_map)
            rgbs.append(rgb_map)
        else:
            rgb_map, depth_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)

            rgbs.append(rgb_map)
            depth_maps.append(depth_map)
    
    if render_feature: 
        ret_rgbs = None if rgbs[0] is None else torch.cat(rgbs)
        return torch.cat(features), torch.cat(selects), ret_rgbs

    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None

def OctreeRender_trilinear_fast_depth(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, is_train=False, device='cuda'):

    depth_maps = []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        depth_map = tensorf.render_depth_map(rays_chunk, is_train=is_train, ndc_ray=ndc_ray, N_samples=N_samples)
        depth_maps.append(depth_map)
    
    return torch.cat(depth_maps)



@torch.no_grad()
def evaluation(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays_stack.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays_stack.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays_stack[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        if len(test_dataset.all_rgbs_stack):
            gt_rgb = test_dataset.all_rgbs_stack[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', depth_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

@torch.no_grad()
def evaluation_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

@torch.no_grad()
def evaluation_feature_pca_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                        ndc_ray=False,  device='cuda'):
    feature_maps = []
    os.makedirs(savePath, exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        feature_map, _, _ = renderer(rays, tensorf, chunk=1024, N_samples=N_samples, 
                            ndc_ray=ndc_ray, is_train=False, render_feature=True, device=device)
        
        feature_map = feature_map.squeeze().detach().cpu().numpy() 

        pca = PCA(n_components=3)

        component = pca.fit_transform(feature_map)
        component = component.reshape(H, W, 3)
        component = ((component - component.min()) / (component.max() - component.min())).astype(np.float32)
        component *= 255.
        component = component.astype(np.uint8)
        
        feature_maps.append(component)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', component)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(feature_maps), fps=30, quality=8)

@torch.no_grad()
def evaluation_feature_text_activation_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                        ndc_ray=False, text='', device='cuda'):
    activation_maps = []
    os.makedirs(savePath, exist_ok=True)

    model, _ = clip.load("ViT-B/16", device=device)
    text = clip.tokenize([text]).to(device)
    text_feature = model.encode_text(text)
    del model

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        feature_map, _, _ = renderer(rays, tensorf, chunk=1024, N_samples=N_samples, 
                            ndc_ray=ndc_ray, is_train=False, render_feature=True, device=device)
        
        feature_map = feature_map.reshape(H, W, -1).permute(2,0,1)[None,...]

        activation_map = F.cosine_similarity(feature_map, text_feature[:,:,None,None], dim=1)
        activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min()) # normalize [-1,1] to [0,1]
        activation_map = activation_map.permute(1,2,0).squeeze().detach().cpu().numpy().astype(np.float32)
        activation_map *= 255.
        activation_map = activation_map.astype(np.uint8)
 
        activation_maps.append(activation_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', activation_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(activation_maps), fps=30, quality=8)

@torch.no_grad()
def evaluation_select_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                        ndc_ray=False,  device='cuda'):
    select_maps = []
    os.makedirs(savePath, exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        _, select_map, _ = renderer(rays, tensorf, chunk=1024, N_samples=N_samples, 
                            ndc_ray=ndc_ray, is_train=False, render_feature=True, device=device)
        
        select_map = select_map.reshape(H, W, 3).cpu().numpy().astype(np.float32)
        select_map *= 255.
        select_map = select_map.astype(np.uint8)
        
        select_maps.append(select_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', select_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(select_maps), fps=30, quality=8)

@torch.no_grad()
def evaluation_segmentation_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                        ndc_ray=False, classes=None, device='cuda'):
    segmentation_maps = []
    os.makedirs(savePath, exist_ok=True)
    
    model, _ = clip.load("ViT-B/16", device=device)
    classes.sort()
    text_features = model.encode_text(clip.tokenize(classes).to(device)).float()
    del model

    # init color
    dc = DistinctColors()

    # save color of every class
    logger.info(classes)
    labels = np.ones((1, 64, 64)).astype(int)
    all_labels = []
    for i in range(len(classes)):
        all_labels.append(labels * i)
    all_labels = np.concatenate(all_labels, 0)
    shape = all_labels.shape
    labels_colored = dc.get_color_fast_numpy(all_labels.reshape(-1))
    labels_colored = (labels_colored.reshape(shape[0] * shape[1], shape[2], 3) * 255).astype(np.uint8)
    Image.fromarray(labels_colored).save(f"{savePath}/classes_color.png")


    try:
        tqdm._instances.clear()
    except Exception:
        pass

    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        feature_map, _, rgb = renderer(rays, tensorf, chunk=1024, N_samples=N_samples, 
                            ndc_ray=ndc_ray, is_train=False, render_feature=True, out_rgb=True, device=device)
        
        feature_map_normalized = F.normalize(feature_map, dim=1) # [N1,D]
        text_features_normalized = F.normalize(text_features, dim=1) # [N2,D]
        relevancy_map = torch.mm(feature_map_normalized, text_features_normalized.T) # [N1,N2]

        p_class = F.softmax(relevancy_map, dim=1) # [N1,N2]
        class_index = torch.argmax(p_class, dim=-1).cpu() # [N1]
        segmentation_map = vis_seg(dc, class_index, H, W, rgb=rgb.cpu())

        segmentation_maps.append(segmentation_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', segmentation_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(segmentation_maps), fps=30, quality=8)

@torch.no_grad()
def evaluation_segmentation_train(test_dataset, tensorf, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                        ndc_ray=False, classes=None, device='cuda'):
    os.makedirs(savePath, exist_ok=True)
    dc = DistinctColors()

    model, _ = clip.load("ViT-B/16", device=device)
    classes_token = clip.tokenize(classes).to(device)
    text_features = model.encode_text(classes_token).float()
    del model

    # init color for every class
    dc = DistinctColors()

    # save color of every class
    logger.info(classes)
    labels = np.ones((1, 64, 64)).astype(int)
    all_labels = []
    for i in range(len(classes)):
        all_labels.append(labels * i)
    all_labels = np.concatenate(all_labels, 0)
    shape = all_labels.shape
    labels_colored = dc.get_color_fast_numpy(all_labels.reshape(-1))
    labels_colored = (labels_colored.reshape(shape[0] * shape[1], shape[2], 3) * 255).astype(np.uint8)
    Image.fromarray(labels_colored).save(f"{savePath}/classes_color.png")

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    W, H = test_dataset.img_wh
    for i in tqdm(range(len(test_dataset))):
        
        rays = test_dataset.all_rays_stack[i].reshape(-1, 6)
        rgb = test_dataset.all_rgbs_stack[i].reshape(-1, 3)

        feature_map, _, _ = renderer(rays, tensorf, chunk=1024, N_samples=N_samples, 
                            ndc_ray=ndc_ray, is_train=False, render_feature=True, device=device)
        
        feature_map_normalized = F.normalize(feature_map, dim=1) # [N1,D]
        text_features_normalized = F.normalize(text_features, dim=1) # [N2,D]
        relevancy_map = torch.mm(feature_map_normalized, text_features_normalized.T) # [N1,N2]

        p_class = F.softmax(relevancy_map, dim=1) # [N1,N2]
        class_index = torch.argmax(p_class, dim=-1).cpu() # [N1]
        segmentation_map = vis_seg(dc, class_index, H, W, rgb=rgb)

        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{i:02d}.png', segmentation_map)

@torch.no_grad()
def evaluation_segmentation_test(test_dataset, tensorf, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                        ndc_ray=False, classes=None, device='cuda'):
    os.makedirs(savePath, exist_ok=True)
    dc = DistinctColors()

    model, _ = clip.load("ViT-B/16", device=device)
    classes_token = clip.tokenize(classes).to(device)
    text_features = model.encode_text(classes_token).float()
    del model

    # init color for every class
    dc = DistinctColors()

    # save color of every class
    logger.info(classes)
    labels = np.ones((1, 64, 64)).astype(int)
    all_labels = []
    for i in range(len(classes)):
        all_labels.append(labels * i)
    all_labels = np.concatenate(all_labels, 0)
    shape = all_labels.shape
    labels_colored = dc.get_color_fast_numpy(all_labels.reshape(-1))
    labels_colored = (labels_colored.reshape(shape[0] * shape[1], shape[2], 3) * 255).astype(np.uint8)
    Image.fromarray(labels_colored).save(f"{savePath}/classes_color.png")

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    IoUs, accuracies = [], []
    for i, frame_idx in tqdm(enumerate(test_dataset.idxes)):
        gt_seg = test_dataset.seg_maps[i] # [H*W=N1, n_classes]

        W, H = test_dataset.img_wh

        rays = test_dataset.all_rays_stack[frame_idx].reshape(-1, 6)
        rgb = test_dataset.all_rgbs_stack[frame_idx].reshape(-1, 3)

        feature_map, _, _ = renderer(rays, tensorf, chunk=1024, N_samples=N_samples, 
                            ndc_ray=ndc_ray, is_train=False, render_feature=True, device=device)
        
        feature_map_normalized = F.normalize(feature_map, dim=1) # [N1,D]
        text_features_normalized = F.normalize(text_features, dim=1) # [N2,D]
        relevancy_map = torch.mm(feature_map_normalized, text_features_normalized.T) # [N1,N2]

        p_class = F.softmax(relevancy_map, dim=1) # [N1,N2]
        class_index = torch.argmax(p_class, dim=-1).cpu() # [N1]
        segmentation_map = vis_seg(dc, class_index, H, W, rgb=rgb)

        one_hot = F.one_hot(class_index.long(), num_classes=gt_seg.shape[-1]) # [N1, n_classes]
        one_hot = one_hot.detach().cpu().numpy().astype(np.int8)
        IoUs.append(jaccard_score(gt_seg, one_hot, average=None))
        print('iou for classes:', IoUs[-1], 'mean iou:', np.mean(IoUs[-1]))
        accuracies.append(accuracy_score(gt_seg, one_hot))
        print('accuracy:', accuracies[-1])

        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{frame_idx:03d}.png', segmentation_map)

    # write IoUs and accuracies to file
    with open(f'{savePath}/{prtx}results.txt', 'w') as f:
        f.write(f'classes: {classes}\n')
        f.write(f'overall: mIoU={np.mean(IoUs)}, accuracy={np.mean(accuracies)}\n\n')
        for i, iou in enumerate(IoUs):
            f.write(f'test image {i}: mIoU={np.mean(iou)}, accuracy={accuracies[i]}\n')
            f.write(f'classes iou: {iou}\n')

@torch.no_grad()
def evaluation_segmentation_depth(test_dataset, tensorf, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                        ndc_ray=False, classes=None, device='cuda'):
    os.makedirs(savePath, exist_ok=True)
    dc = DistinctColors()
    
    model, _ = clip.load("ViT-B/16", device=device)
    classes_token = clip.tokenize(classes).to(device)
    text_features = model.encode_text(classes_token).float()
    del model

    # init color for every class
    dc = DistinctColors()

    # save color of every class
    logger.info(classes)
    labels = np.ones((1, 64, 64)).astype(int)
    all_labels = []
    for i in range(len(classes)):
        all_labels.append(labels * i)
    all_labels = np.concatenate(all_labels, 0)
    shape = all_labels.shape
    labels_colored = dc.get_color_fast_numpy(all_labels.reshape(-1))
    labels_colored = (labels_colored.reshape(shape[0] * shape[1], shape[2], 3) * 255).astype(np.uint8)
    Image.fromarray(labels_colored).save(f"{savePath}/classes_color.png")

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    W, H = test_dataset.img_wh

    i = 10
        
    rays = test_dataset.all_rays_stack[i].reshape(-1, 6)
    rgb = test_dataset.all_rgbs_stack[i].reshape(-1, 3)

    feature_map, _, _ = renderer(rays, tensorf, chunk=1024, N_samples=N_samples, 
                        ndc_ray=ndc_ray, is_train=False, render_feature=True, device=device)
    depth_map = OctreeRender_trilinear_fast_depth(rays, tensorf, chunk=1024, N_samples=N_samples,
                                                    ndc_ray=ndc_ray, is_train=False, device=device)
    
    # extract coordinates of points
    points_coordinates = construct_points_coordinates(rays.to(device), depth_map) # [N,3]
    points_coordinates = points_coordinates

    feature_map_normalized = F.normalize(feature_map, dim=1) # [N1,D]
    text_features_normalized = F.normalize(text_features, dim=1) # [N2,D]
    relevancy_map = torch.mm(feature_map_normalized, text_features_normalized.T) # [N1,N2]

    p_class = F.softmax(relevancy_map, dim=1) # [N1,N2]
    class_index = torch.argmax(p_class, dim=-1).cpu() # [N1]
    segmentation_map = dc.apply_colors_fast_torch(class_index)
    # segmentation_map = segmentation_map * alpha + rgb * (1 - alpha)
    segmentation_map = segmentation_map.reshape(-1, 3)
    
    save_points_to_ply(points_coordinates.cuda(), segmentation_map.cuda(), f'{savePath}/{prtx}seg{i}_points.ply')