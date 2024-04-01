import torch
import numpy as np
import gaussian_splatting.utils as utils
from gaussian_splatting.trainer import Trainer
import gaussian_splatting.utils.loss_utils as loss_utils
from gaussian_splatting.utils.data_utils import read_all
from gaussian_splatting.utils.camera_utils import to_viewpoint_camera
from gaussian_splatting.utils.point_utils import get_point_clouds
from gaussian_splatting.gauss_model import GaussModel
from gaussian_splatting.gauss_render import GaussRenderer

import contextlib

from torch.profiler import profile, ProfilerActivity

USE_GPU_PYTORCH = True
USE_PROFILE = False

colors = np.array([[0.0,0.0,0.0],
                   [0.501960813999176,0.501960813999176,0.250980406999588],
                   [0.0,0.501960813999176,0.0],
                   [0.501960813999176,0.501960813999176,0.0],
                   [0.0,0.0,0.501960813999176],
                   [0.501960813999176,0.0,0.501960813999176]])

class GSSTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = kwargs.get('data')
        self.gaussRender = GaussRenderer(**kwargs.get('render_kwargs', {}))
        self.lambda_dssim = 0.2
        self.lambda_depth = 0.75
        self.lambda_semantic = 1.0
        self.model = kwargs.get('model')
    
    def on_train_step(self):
        ind = np.random.choice(len(self.data['camera']))
        camera = self.data['camera'][ind]
        rgb = self.data['rgb'][ind]
        depth = self.data['depth'][ind]
        mask = (self.data['alpha'][ind] > 0.5)
        semantic = 255 * self.data['semantic'][ind]

        if USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(camera)

        if USE_PROFILE:
            prof = profile(activities=[ProfilerActivity.CUDA], with_stack=True)
        else:
            prof = contextlib.nullcontext()

        with prof:
            out = self.gaussRender(pc=self.model, camera=camera)

        if USE_PROFILE:
            print(prof.key_averages(group_by_stack_n=True).table(sort_by='self_cuda_time_total', row_limit=20))

        # Photometric losses
        l1_loss = loss_utils.l1_loss(out['render'], rgb)
        depth_loss = loss_utils.l1_loss(out['depth'][..., 0][mask], depth[mask])
        ssim_loss = 1.0-loss_utils.ssim(out['render'], rgb)

        # Semantic losses
        sem_loss = loss_utils.sem_loss(out['semantic'], semantic, num_classes=self.gaussRender.num_classes)

        # Total loss
        total_loss = ((1-self.lambda_dssim) * l1_loss +
                     self.lambda_dssim * ssim_loss +
                     depth_loss * self.lambda_depth +
                     sem_loss * self.lambda_semantic)

        psnr = utils.img2psnr(out['render'], rgb)
        log_dict = {'total': total_loss,'l1':l1_loss, 'ssim': ssim_loss, 'depth': depth_loss, 'psnr': psnr, 'semantic': sem_loss}

        return total_loss, log_dict

    def on_evaluate_step(self, **kwargs):
        import matplotlib.pyplot as plt
        ind = np.random.choice(len(self.data['camera']))
        camera = self.data['camera'][ind]
        if USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(camera)

        # Render rgb image
        rgb = self.data['rgb'][ind].detach().cpu().numpy()
        out = self.gaussRender(pc=self.model, camera=camera)
        rgb_pd = out['render'].detach().cpu().numpy()
        image = np.concatenate([rgb, rgb_pd], axis=1)

        # Render depth image
        depth_pd = out['depth'].detach().cpu().numpy()[..., 0]
        depth = self.data['depth'][ind].detach().cpu().numpy()
        depth = np.concatenate([depth, depth_pd], axis=1)
        depth = (1 - depth / depth.max())
        depth = plt.get_cmap('jet')(depth)[..., :3]

        # Render semantic image
        sem = 255 * self.data['semantic'][ind].detach().cpu().numpy()
        sem_pd = np.argmax(out['semantic'].detach().cpu().numpy(), axis=2)
        sem = np.concatenate([sem, sem_pd], axis=1)
        sem = colors[sem.astype(int),:]   

        # Concatenate outputs
        image = np.concatenate([image, depth, sem], axis=0)

        utils.imwrite(str(self.results_folder / f'image-{self.step}.png'), image)

        self.model.save_ply(path='/result/Replica/gs_model.ply')


if __name__ == "__main__":
    device = 'cuda'
    # folder = './B075X65R3X'
    folder = './Replica'
    data = read_all(folder, resize_factor=0.5)
    data = {k: v.to(device) for k, v in data.items()}
    data['depth_range'] = torch.Tensor([[1,3]]*len(data['rgb'])).to(device)

    points = get_point_clouds(data['camera'], data['depth'], data['alpha'], data['rgb'], data['semantic'])
    raw_points = points.random_sample(2**16)
    gaussModel = GaussModel(sh_degree=4, debug=False)
    gaussModel.create_from_pcd(pcd=raw_points)

    render_kwargs = {
        'white_bkgd': True,
    }

    trainer = GSSTrainer(model=gaussModel, 
        data=data,
        train_batch_size=1, 
        train_num_steps=25000,
        i_image =100,
        train_lr=1e-3, 
        amp=False,
        fp16=True,
        results_folder='result/test',
        render_kwargs=render_kwargs,
    )

    trainer.on_evaluate_step()
    trainer.train()

    gaussModel.save_ply(path='/result/Replica/gs_model.ply')