import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from torchvision.utils import make_grid, save_image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import pytorch_lightning as pl
from einops import rearrange, repeat
from huggingface_hub import hf_hub_download

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.infer_util import remove_background, resize_foreground, save_video, apply_background_color, get_visible_vertices_with_occlusion
import rembg
import trimesh
from src.utils.mesh_util import save_obj, save_obj_with_mtl, get_mesh
from src.utils.infer_util import get_render_cameras, render_frames, get_input_image_camera
from src.utils.camera_util import get_rel_trafo
from src.evaluator.evaluator import Evaluator
import imageio


# Regulrarization loss for FlexiCubes
def sdf_reg_loss_batch(sdf, all_edges):
    sdf_f1x6x2 = sdf[:, all_edges.reshape(-1)].reshape(sdf.shape[0], -1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = F.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()) + \
               F.binary_cross_entropy_with_logits(
                   sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff


class MVRecon(pl.LightningModule):
    def __init__(
        self,
        lrm_generator_config,
        zero123plus_config,
        input_size=256,
        render_size=512,
        init_ckpt=None,
        zero123plus_ckpt=None,
        lrm_ckpt=None,
        no_rembg=False,
        scale=1,
        diffusion_steps=75,
        export_texmap=True,
        texture_resolution=1024,
        render_resolution=512,
        distance=4.5,
        store_vis=False,
        threshold_faces=100,
        fovy=0
    ):
        super(MVRecon, self).__init__()

        self.input_size = input_size
        self.render_size = render_size
        self.zero123plus_ckpt = zero123plus_ckpt
        self.lrm_ckpt = lrm_ckpt
        self.no_rembg = no_rembg
        self.scale = scale
        self.diffusion_steps = diffusion_steps
        self.export_texmap = export_texmap
        self.texture_resolution = texture_resolution
        self.render_resolution = render_resolution
        self.distance = distance
        self.store_vis = store_vis
        self.threshold_faces = threshold_faces
        self.fovy = fovy

        # init modules
        self.zero123plus = instantiate_from_config(zero123plus_config).to(self.device)
        self.lrm_generator = instantiate_from_config(lrm_generator_config).to(self.device)

        # Load weights from pretrained MVRecon model, and use the mlp 
        # weights to initialize the weights of sdf and rgb mlps.
        if False: # init_ckpt is not None:
            sd = torch.load(init_ckpt, map_location='cpu')['state_dict']
            sd = {k: v for k, v in sd.items() if k.startswith('lrm_generator')}
            sd_fc = {}
            for k, v in sd.items():
                if k.startswith('lrm_generator.synthesizer.decoder.net.'):
                    if k.startswith('lrm_generator.synthesizer.decoder.net.6.'):    # last layer
                        # Here we assume the density filed's isosurface threshold is t, 
                        # we reverse the sign of density filed to initialize SDF field.  
                        # -(w*x + b - t) = (-w)*x + (t - b)
                        if 'weight' in k:
                            sd_fc[k.replace('net.', 'net_sdf.')] = -v[0:1]
                        else:
                            sd_fc[k.replace('net.', 'net_sdf.')] = 10.0 - v[0:1]
                        sd_fc[k.replace('net.', 'net_rgb.')] = v[1:4]
                    else:
                        sd_fc[k.replace('net.', 'net_sdf.')] = v
                        sd_fc[k.replace('net.', 'net_rgb.')] = v
                else:
                    sd_fc[k] = v
            sd_fc = {k.replace('lrm_generator.', ''): v for k, v in sd_fc.items()}
            # missing `net_deformation` and `net_weight` parameters
            self.lrm_generator.load_state_dict(sd_fc, strict=False)
            print(f'Loaded weights from {init_ckpt}')
        
        self.validation_step_outputs = []
        self.evaluator_lrm = Evaluator(do_evaluate_mesh=True, do_evaluate_nvs=True)
    
    def load_checkpoints_composition(self):
        if os.path.exists(self.zero123plus_ckpt):
            unet_ckpt_path = self.zero123plus_ckpt
        else:
            unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
        state_dict = torch.load(unet_ckpt_path, map_location='cpu')
        state_dict = {'unet.' + k: v for k, v in state_dict.items()}
        self.zero123plus.pipeline.unet.load_state_dict(state_dict, strict=True)
        self.zero123plus = self.zero123plus.to(self.device)

        if os.path.exists(self.lrm_ckpt):
            lrm_ckpt_path = self.lrm_ckpt
        else:
            lrm_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="instant_mesh_large.ckpt", repo_type="model")
        state_dict = torch.load(lrm_ckpt_path, map_location='cpu')['state_dict']
        state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
        self.lrm_generator.load_state_dict(state_dict)
    
    def on_validation_start(self):
        device = torch.device(f'cuda:{self.global_rank}')
        self.lrm_generator.init_flexicubes_geometry(device, self.fovy)
        self.lrm_generator = self.lrm_generator.to(device)
        self.zero123plus.pipeline = self.zero123plus.pipeline.to(device)
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, 'images_val'), exist_ok=True)
    
    def get_adapted_img(self, image_adapted, rembg_session=None, bgcolor=None, original_scale_settings=True):
        image_adapted = remove_background(to_pil_image(image_adapted), rembg_session)
        if original_scale_settings:
            image_adapted = resize_foreground(image_adapted, 0.85)
        if bgcolor is not None:
            image_adapted = apply_background_color(image_adapted, bgcolor)
        return image_adapted
    
    def project(self, points_world, world2cam):
        proj_matrix = self.lrm_generator.geometry.renderer.camera.proj_mtx.to(self.device)
        points_world = torch.from_numpy(points_world).to(self.device)
        points_cam = torch.matmul(
            torch.nn.functional.pad(points_world, pad=(0, 1), mode='constant', value=1.0),
            torch.transpose(world2cam, 0, 1).to(self.device))
        points_xy = torch.matmul(
            points_cam,
            torch.transpose(proj_matrix, 1, 2))
        print(points_xy.shape)
        quit()
        return points_xy

    def prepare_validation_batch_data(self, batch, original_scale_settings=False):
        bs, v, _, _ = batch['target_c2ws'].shape
        zero123plus_input = {}
        lrm_generator_input = {}
        evaluator_input = {}

        # input images
        images = batch['cond_imgs']
        images = v2.functional.resize(
            images, self.input_size, interpolation=3, antialias=True).clamp(0, 1)
        rembg_session = None if self.no_rembg else rembg.new_session()
        images_adapted = list()
        for i in range(bs):
            adapted = pil_to_tensor(self.get_adapted_img(
                images[i], rembg_session, original_scale_settings=original_scale_settings))
            adapted = v2.functional.resize(
                adapted.unsqueeze(0), self.input_size, interpolation=3, antialias=True)
            images_adapted.append(adapted.squeeze())
        images_adapted = torch.stack(images_adapted)
        zero123plus_input['input_images'] = images_adapted.to(self.device)

        # input cameras from zero123plus
        cameras = list()
        dists_from_center = list()
        for i in range(bs):
            if original_scale_settings:
                cond_dist_from_center = 4
            else:
                cond_dist_from_center = torch.linalg.norm(batch['cond_c2ws'][i, :-1, 3]).cpu().numpy()
            dists_from_center.append(cond_dist_from_center)
            cameras.append(get_zero123plus_input_cameras(
                batch_size=1,
                radius=cond_dist_from_center).to(self.device).squeeze())
        cameras = torch.stack(cameras)
        lrm_generator_input['cameras'] = cameras.to(self.device)

        # render cameras to evaluate target views
        render_w2c0s = list()
        for dist in dists_from_center:
            render_w2c0s.append(get_input_image_camera(
                    batch_size=1, 
                    radius=dist, 
                    elevation=30.0, # 20 WHY!???!?!?!?!?!
                    is_flexicubes=True,))
        render_w2c0s = torch.stack(render_w2c0s)
        lrm_generator_input['render_w2c0s'] = render_w2c0s
        render_w2c0s = render_w2c0s.repeat((1, 1, v, 1, 1)).squeeze().to(self.device)
        rel_c02ct = torch.einsum('abik, akj -> abij', torch.linalg.inv(
            batch['target_c2ws']), batch['cond_c2ws'])
        render_w2cs = torch.einsum('abik, abkj -> abij', rel_c02ct, render_w2c0s)
        lrm_generator_input['render_cameras'] = render_w2cs.to(self.device)
        lrm_generator_input['render_size'] = self.render_size
        lrm_generator_input['render_w2c0s'] = torch.atleast_3d(
            lrm_generator_input['render_w2c0s'].squeeze())

        # evaluator input
        # get mesh
        evaluator_input['mesh'] = [trimesh.load(mesh_path) for mesh_path in batch['mesh_path']]

        # adapt images as for zero123++ input
        if original_scale_settings:
            target_images_adapted = list()
            _, _, _, h, w = batch['target_images'].shape
            for sample in batch['target_images']: # (B, 6, C, H, W)
                sample_images_adapted = list()
                for image in sample:
                    adapted_img = pil_to_tensor(self.get_adapted_img(
                        image,
                        rembg_session,
                        bgcolor=(255, 255, 255, 255),
                        original_scale_settings=original_scale_settings))
                    adapted_img = v2.functional.resize(
                        adapted_img, self.render_size, interpolation=3, antialias=True)[:3, :, :]
                    sample_images_adapted.append(adapted_img)
                target_images_adapted.append(torch.stack(sample_images_adapted))
            batch['target_images'] = torch.stack(target_images_adapted) / 255

        # images for lrm
        target_imgs = batch['target_images']  # (B, 6, C, H, W)
        target_imgs = v2.functional.resize(
            target_imgs, self.render_size, interpolation=3, antialias=True).clamp(0, 1)
        evaluator_input['target_imgs'] = target_imgs.to(self.device)

        # add camera positions
        evaluator_input['gt_c02ws'] = batch['cond_c2ws']
        evaluator_input['pred_c02ws'] = torch.linalg.inv(lrm_generator_input['render_w2c0s'])

        # object ids
        object_ids = [os.path.basename(os.path.dirname(mesh_path)) for mesh_path in batch['mesh_path']]

        # visualize every x object
        visualize = batch['visualize']

        return zero123plus_input, lrm_generator_input, evaluator_input, object_ids, dists_from_center, visualize
    
    def forward_lrm_generator(self, images, cameras, render_cameras, render_size=512):
        planes = torch.utils.checkpoint.checkpoint(
            self.lrm_generator.forward_planes, 
            images, 
            cameras, 
            use_reentrant=False,
        )
        out = self.lrm_generator.forward_geometry(
            planes, 
            render_cameras, 
            render_size,
        )
        return planes, out
    
    def forward(self, lrm_generator_input):
        images = lrm_generator_input['images']
        cameras = lrm_generator_input['cameras']
        render_cameras = lrm_generator_input['render_cameras']
        render_size = lrm_generator_input['render_size']
        planes_list = list()
        out_list = list()
        for i in range(lrm_generator_input['images'].shape[0]):
            planes, out = self.forward_lrm_generator(
                images[i].unsqueeze(0), cameras[i].unsqueeze(0), render_cameras[i].unsqueeze(0), render_size=render_size)
            planes_list.append(planes)
            out_list.append(out)
        return planes_list, out_list
    
    def write_im(self, tensor, name):
        tensor = tensor.detach().clone().cpu().numpy()
        if tensor.max() < 2:
            tensor = tensor * 255
        tensor = tensor.astype(np.uint8)
        imageio.imwrite(name, tensor)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        zero123plus_input, lrm_generator_input, evaluator_input, object_ids, dists_from_center, visualize = self.prepare_validation_batch_data(batch)
        output_images = list()
        for i, (object_id, input_image) in enumerate(zip(object_ids, zero123plus_input['input_images'])):
            if self.store_vis or visualize[i]:
                os.makedirs(f'outputs/{object_id}', exist_ok=True)
                self.write_im(input_image.permute(1, 2, 0), f'outputs/{object_id}/input_image.png')
            
            # get images from zero123plus
            output_image = self.zero123plus.pipeline(
                    to_pil_image(input_image), 
                    num_inference_steps=self.diffusion_steps, 
                ).images[0]
            
            output_image = np.asarray(output_image, dtype=np.float32) / 255.0
            output_image = torch.from_numpy(output_image).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
            if visualize[i]:
                self.write_im(output_image.permute(1, 2, 0), f'outputs/{object_id}/output_image.png')
            output_image = rearrange(output_image, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)
            output_image = v2.functional.resize(
                output_image, 320, interpolation=3, antialias=True).clamp(0, 1)
            output_images.append(output_image)

        output_images = torch.stack(output_images).squeeze().to(self.device)
        if len(output_images.shape) != 5:
            output_images = output_images.unsqueeze(0)

        # make input right format
        output_images = output_images.to(self.device)
        output_images = v2.functional.resize(output_images, 320, interpolation=3, antialias=True).clamp(0, 1)
        lrm_generator_input['images'] = output_images

        # lrm generator get planes and images
        planes_list, render_out_list = self.forward(lrm_generator_input)

        mesh_out_list = list()
        for i, (planes, object_id, render_w2c0) in enumerate(zip(planes_list, object_ids, lrm_generator_input['render_w2c0s'])):
            # get mesh
            vertices, faces, vertex_colors = self.lrm_generator.extract_mesh(
                planes,
                use_texture_map=self.export_texmap,
                texture_resolution=self.texture_resolution,)
            mesh_out = get_mesh(vertices, faces, vertex_colors)
            mesh_out_list.append(mesh_out)
            if self.export_texmap:
                vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
                if self.store_vis or visualize[i]:
                    save_obj_with_mtl(
                        vertices.data.cpu().numpy(),
                        uvs.data.cpu().numpy(),
                        faces.data.cpu().numpy(),
                        mesh_tex_idx.data.cpu().numpy(),
                        tex_map.permute(1, 2, 0).data.cpu().numpy(),
                        f'outputs/{object_id}/mesh.ply',
                    )
            else:
                if self.store_vis or visualize[i]:
                    save_obj(vertices, faces, vertex_colors, f'outputs/{object_id}/mesh.obj')

        render_images_list = list()
        for i, render_out in enumerate(render_out_list):
            # lrm generator get rendered images
            render_images = render_out['img']
            render_images_list.append(render_images)
            render_images = rearrange(render_images, 'b n c h w -> b c h (n w)')
            if visualize[i]:
                self.write_im(rearrange(
                    render_images.squeeze(), 'c h (n m w) -> c (n h) (m w)', n=3, m=2).permute(1, 2, 0), f'outputs/{object_id}/render_image.png')
                self.write_im(rearrange(
                    evaluator_input['target_imgs'][i].squeeze(), '(n m) c h w -> c (n h) (m w)', n=3, m=2).permute(1, 2, 0), f'outputs/{object_id}/targetlrm_image.png')
            self.validation_step_outputs.append(render_images)
                
        render_images = torch.stack(render_images_list).squeeze()
        if len(render_images.shape) != 5:
            render_images = render_images.unsqueeze(0)

        self.evaluator_lrm.evaluate(
            {'raw_imgs': render_images.to(self.device), 'object_mesh': mesh_out_list, 'c02ws': evaluator_input['pred_c02ws']},
            {'raw_imgs': evaluator_input['target_imgs'].to(self.device), 'object_mesh': evaluator_input['mesh'], 'object_id': object_ids, 'c02ws': evaluator_input['gt_c02ws']}
        )
        # generate rendering video
        if self.store_vis or visualize[i]:
            for i, (planes, object_id, dist) in enumerate(zip(planes_list, object_ids, dists_from_center)):
                video_path_idx = os.path.join('outputs', object_id, f'video.mp4')

                render_size = self.render_resolution
                render_cameras = get_render_cameras(
                    batch_size=1, 
                    M=120, 
                    radius=dist, 
                    elevation=30.0, # why 20 !?!?!?!
                    is_flexicubes=True,
                ).to(self.device)

                frames = render_frames(
                    self.lrm_generator, 
                    planes, 
                    render_cameras=render_cameras, 
                    render_size=render_size, 
                    chunk_size=20, 
                    is_flexicubes=True,
                )
                self.write_im(frames[0].permute(1, 2, 0), f'outputs/{object_id}/input_image_rendered.png')
                save_video(
                    frames,
                    video_path_idx, 
                    fps=30,
                    )
        
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        metrics_lrm = self.evaluator_lrm.get_metrics()
        self.log_dict(metrics_lrm, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        print(f"LRM metrics {metrics_lrm}")
        images = torch.cat(self.validation_step_outputs, dim=-1)

        all_images = self.all_gather(images)
        all_images = rearrange(all_images, 'r b c h w -> (r b) c h w')

        if self.global_rank == 0:
            image_path = os.path.join(self.logdir, 'images_val', f'val_{self.global_step:07d}.png')

            grid = make_grid(all_images, nrow=1, normalize=True, value_range=(0, 1))
            save_image(grid, image_path)
            print(f"Saved image to {image_path}")

        self.validation_step_outputs.clear()