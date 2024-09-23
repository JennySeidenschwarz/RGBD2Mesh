import os
import imageio
import rembg
import torch
import numpy as np
import PIL.Image
from PIL import Image
from typing import Any
import tqdm
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from typing import Any, List, Optional, Tuple, Union, cast
import trimesh


def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def get_input_image_camera(batch_size=1, radius=4.0, elevation=20.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=1, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)

    return cameras


def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    """
    Render frames from triplanes.
    """
    frames = []
    for i in tqdm.tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['img']
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)
    
    frames = torch.cat(frames, dim=1)[0]    # we suppose batch size is always 1
    return frames


def remove_background(image: PIL.Image.Image,
    rembg_session: Any = None,
    force: bool = False,
    **rembg_kwargs,
) -> PIL.Image.Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image


def apply_background_color(img: PIL.Image.Image, color: Tuple[int, int, int, int]) -> PIL.Image.Image:
    """
    Apply the specified background color to the image.

    Args:
        img (PILImage): The image to be modified.
        color (Tuple[int, int, int, int]): The RGBA color to be applied.

    Returns:
        PILImage: The modified image with the background color applied.
    """
    r, g, b, a = color
    colored_image = Image.new("RGBA", img.size, (r, g, b, a))
    colored_image.paste(img, mask=img)

    return colored_image


def resize_foreground(
    image: PIL.Image.Image,
    ratio: float,
) -> PIL.Image.Image:
    image = np.array(image)
    assert image.shape[-1] == 4
    alpha = np.where(image[..., 3] > 0)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )
    # crop the foreground
    fg = image[y1:y2, x1:x2]
    # pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )

    # compute padding according to the ratio
    new_size = int(new_image.shape[0] / ratio)
    # pad to size, double side
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    new_image = PIL.Image.fromarray(new_image)
    return new_image


def images_to_video(
    images: torch.Tensor, 
    output_path: str, 
    fps: int = 30,
) -> None:
    # images: (N, C, H, W)
    video_dir = os.path.dirname(output_path)
    video_name = os.path.basename(output_path)
    os.makedirs(video_dir, exist_ok=True)

    frames = []
    for i in range(len(images)):
        frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
            f"Frame shape mismatch: {frame.shape} vs {images.shape}"
        assert frame.min() >= 0 and frame.max() <= 255, \
            f"Frame value out of range: {frame.min()} ~ {frame.max()}"
        frames.append(frame)
    imageio.mimwrite(output_path, np.stack(frames), fps=fps, quality=10)


def save_video(
    frames: torch.Tensor,
    output_path: str,
    fps: int = 30,
) -> None:
    # images: (N, C, H, W)
    frames = [(frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for frame in frames]
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def is_vertex_occluded(vertices, camera_position, mesh):
    # Create a ray from the camera to the vertex
    ray_directions = vertices - camera_position
    ray_origins = np.repeat(
        np.expand_dims(camera_position, axis=0),
        ray_directions.shape[0],
        axis=0)
    
    # Create a ray object
    ray = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    
    # Check if the ray intersects with any part of the mesh
    locations, index_ray, index_triangle = ray.intersects_location(
        np.atleast_2d(ray_origins), np.atleast_2d(ray_directions))
    
    # Check if the intersection occurs before the vertex
    distances_mesh = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
    distances_cam = np.linalg.norm(ray_directions, axis=1)

    # check if any intersection of ray and mesh is in front of vertex
    is_mesh_between_vertex_and_cam = distances_mesh < distances_cam[index_ray]
    idx_vertex_where_mesh_between_vertex_and_cam = index_ray[is_mesh_between_vertex_and_cam]
    # check if at least one intersection of ray and mesh is in front of vertex
    occluded = np.isin(np.arange(vertices.shape[0]), idx_vertex_where_mesh_between_vertex_and_cam)

    return occluded


def is_vertex_in_fov(vertices, camera_position, view_direction, fov, near_clip, far_clip):
    # Vector from the camera to the vertex
    to_vertex = vertices - camera_position
    to_vertex_normalized = to_vertex / np.repeat(np.expand_dims(np.linalg.norm(to_vertex, axis=1), axis=1), 3, axis=1)

    # Compute angle between the view direction and the vector to the vertex
    angle = np.arccos(view_direction @ to_vertex_normalized.T) * 180 / np.pi
    # Check if within field of view and depth range
    return np.logical_and(
        angle <= fov / 2,
        np.logical_and(
            near_clip <= np.linalg.norm(to_vertex, axis=1),
            np.linalg.norm(to_vertex, axis=1) <= far_clip)
        )


def get_visible_vertices_with_occlusion(mesh, camera_transform, target_position=np.array([0, 0, 0]), fov=50, near_clip=0.1, far_clip=100, use_surface_points=False):
    if use_surface_points:
        points, _ = trimesh.sample.sample_surface(
            mesh, 99999, face_weight=None, sample_color=False)
    else:
        points = mesh.vertices
    
    camera_position = camera_transform[:3, 3]
    view_direction = target_position - camera_position
    view_direction /= np.linalg.norm(view_direction)
    # view_direction = np.dot(camera_transform[:3, :3], view_direction.T).T

    is_in_fov = is_vertex_in_fov(points, camera_position, view_direction, fov, near_clip, far_clip)
    occluded = is_vertex_occluded(points, camera_position, mesh)
    
    return np.logical_and(is_in_fov, ~occluded), points


def render(mesh, cam, image_size=(512, 512), fov=(50, 50), camera_position=None, target_position=np.array([0, 0, 0])):
    camera = trimesh.scene.cameras.Camera(
        resolution=image_size,
        fov=fov,

    )
    view_direction = target_position - camera_position
    view_direction /= np.linalg.norm(view_direction)

    # w2c --> c2w
    cam_transform = cam
    cam_transform = np.linalg.inv(cam_transform)

    # scene will have automatically generated camera and lights
    scene = mesh.scene(camera=camera, camera_transform=cam_transform)

    # convert the camera to rays with one ray per pixel
    origins, vectors, pixels = scene.camera_rays()

    # do the actual ray- mesh queries
    points, index_ray, index_tri = mesh.ray.intersects_location(
        origins, vectors, multiple_hits=False
    )

    # for each hit, find the distance along its vector
    depth = trimesh.util.diagonal_dot(points - origins[0], vectors[index_ray])

    # find pixel locations of actual hits
    pixel_ray = pixels[index_ray]

    # create a numpy array we can turn into an image
    # doing it with uint8 creates an `L` mode greyscale image
    a = np.zeros(scene.camera.resolution, dtype=np.uint8)

    # scale depth against range (0.0 - 1.0)
    depth_float = (depth - 0) / np.ptp(depth)

    # convert depth into 0 - 255 uint8
    depth_int = (depth_float * 255).round().astype(np.uint8)
    # assign depth to correct pixel locations
    a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int
    # a = a.T
    # create a PIL image from the depth queries
    img = PIL.Image.fromarray(a)

    # show the resulting image
    imageio.imwrite('depth.png', img)

    # create a raster render of the same scene using OpenGL
    # rendered = PIL.Image.open(trimesh.util.wrap_as_stream(scene.save_image()))
