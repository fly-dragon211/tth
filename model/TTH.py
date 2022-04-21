import math, numbers, pdb
import numpy as np
import PIL
from torchvision.transforms import Compose, Resize, CenterCrop, TenCrop, Lambda, ToTensor, Normalize, RandomResizedCrop

import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F


def preprocess_clip_toTensor(shape=[224, 224], path='/data/hf/VisualSearch/flickr30k/ads/exported_qrcode_image.png'):
    preprocess = Compose([
            Resize(shape),
            lambda image: image.convert("RGB"),
            ToTensor(),
        ])
    return preprocess(PIL.Image.open(path))


# targeted mismatch attack with a patch
def tth_patch(networks, scales, target_tensor, carrier_img, mode='normal',
              num_steps=100, lr=1.0, lam=0.0, sigma_blur=0.0, verbose=True, seed=155, device=torch.device("cpu"),
              target_type='image', patch_ratio=0.2):
    # if seed is not None:   # uncomment to reproduce the results of the ICCV19 paper - still some randomness though
    #     reproduce(seed)
    target_tensor = target_tensor.to(device)

    carrier_img = carrier_img.to(device)
    patch_tensor1 = torch.rand_like(carrier_img)  # parameters to be learned is a patch
    M = torch.zeros_like(carrier_img)
    patch_length, patch_width = int(carrier_img.shape[-1]*patch_ratio), int(carrier_img.shape[-2]*patch_ratio)
    M[:, :, -patch_length:, -patch_width:] = 1

    patch_tensor1 = M.mul(patch_tensor1)
    patch_tensor = nn.Parameter(patch_tensor1.data)

    carrier_tensor = (1-M).mul(carrier_img) + M.mul(patch_tensor)
    carrier_org = carrier_img.clone()                 # to compute distortion
    optimizer = optim.Adam([patch_tensor], lr=lr)
    lr_schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95),
                     torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                                patience=2)]

    bin_centers_fixed = torch.DoubleTensor(np.arange(0, 1.001, 0.05)).to(device)   # for histograms only
    scales = np.array(scales)
    sigma_blur_all = sigma_blur / np.array(scales)
    kernel_size_all = 2*np.floor((np.ceil(6*sigma_blur_all)/2))+1

    # pre-compute all target global-descriptors / histograms / tensors
    targets, norm_factors = {}, {}
    for network in networks:  # optimize for all networks
        network.eval()

        m = torch.FloatTensor(network.meta['mean']).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        s = torch.FloatTensor(network.meta['std']).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        for scale in scales:  # optimize for all scales
            si = (scales==scale).nonzero()[0].item()
            if sigma_blur > 0.0:
                GS = GaussianSmoothing(channels = 3, kernel_size = kernel_size_all[si], sigma = sigma_blur_all[si]).to(device)
            else:
                GS = nn.Sequential()  # identity function

            # normalize (mean-std), re-scale, and feed to the network
            if target_type == 'image':
                xl = network.visual_features(nn.functional.interpolate((GS(target_tensor) - m) / s, scale_factor=scale, mode='bilinear', align_corners=False))
            elif target_type == 'embedding':
                xl = target_tensor
            elif target_type == 'embeddings':
                xl = [each.unsqueeze(0) for each in target_tensor]

            if not isinstance(xl, list): xl = [xl]  # to support optimization for multi targets too
            for l in range(len(xl)):
                x = xl[l]
                if mode == 'global':
                    # global descriptors
                    targets[network.meta['architecture'], str(scale), 'layer'+str(l)] = network.norm(x).squeeze().detach()
                elif mode == 'hist':  # activation histogram
                    nf = x.max().detach()
                    norm_factors[network.meta['architecture'], str(scale), 'layer'+str(l)] = nf
                    targets[network.meta['architecture'],str(scale), 'layer'+str(l)] = hist_per_channel((x / nf).clamp(0,1), bin_centers_fixed).detach()
                elif mode == 'tensor':  # activation tensor
                    nf = (0.1*x.max()).detach()  # 0.1 ???
                    norm_factors[network.meta['architecture'],str(scale), 'layer'+str(l)] = nf
                    targets[network.meta['architecture'], str(scale), 'layer'+str(l)] = (x / nf).detach()

    # for convergence checks
    globals()['converged'] = True
    globals()['loss_perf_min'] = 1e+9
    globals()['loss_perf_converged'] = 1e-4
    globals()['convergence_safe'] = False

    print('Optimizing..')
    itr = [0]
    while itr[0] <= num_steps:

        def closure():
            optimizer.zero_grad()
            loss_perf = torch.Tensor(1).to(device)*0.0
            n = 0  # counter for loss summands
            for network in networks:  # optimize for all networks
                network.eval()
                m = torch.FloatTensor(network.meta['mean']).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                s = torch.FloatTensor(network.meta['std']).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                for scale in scales:  # optimize for all scales
                    si = (scales==scale).nonzero()[0].item()
                    if sigma_blur > 0.0:
                        GS = GaussianSmoothing(channels=3, kernel_size=kernel_size_all[si], sigma=sigma_blur_all[si]).to(device)
                    else:
                        GS = nn.Sequential() # identity function

                    # normalize (mean-std), re-scale, and feed to the network
                    carrier_x = network.visual_features(nn.functional.interpolate((GS(carrier_tensor) - m) / s, scale_factor=scale, mode='bilinear', align_corners=False))


                    if mode == 'global': # global descriptors
                        for l in range(len(targets)):
                            ref = network.norm(carrier_x).squeeze()
                            target = targets[network.meta['architecture'], str(scale), 'layer'+str(l)]
                            loss_perf += 1 - (ref).dot(target)   # add loss over networks and scales
                            n+= 1

            # compute loss
            if lam > 0:
                loss_distort = (carrier_tensor-carrier_org).pow(2.0).sum() / (carrier_tensor.size(-1)*carrier_tensor.size(-2))
            else:
                loss_distort = torch.Tensor(1).to(device)*0.0
            loss_perf = loss_perf / n  # divide by number of summands (networks, scales, poolings)
            total_loss = loss_perf + lam * loss_distort

            # check for convergence (hacky!)
            if loss_perf < globals()['loss_perf_min']:
                globals()['loss_perf_min'] = loss_perf.clone()

            if loss_perf < globals()['loss_perf_converged']:
                globals()['convergence_safe'] = True

            if globals()['converged'] and (loss_perf-globals()['loss_perf_min']) > 1*globals()['loss_perf_min'] and globals()['convergence_safe'] == False:
                globals()['converged'] = False
                print("Iter {:5d}, Loss_perf = {:6f} Loss_distort = {:6f} Loss_total = {:6f}".format(itr[0], loss_perf.item(), loss_distort.item(), total_loss.item()))
                print('Did not converge')

            total_loss.backward(retain_graph=True)

            if verbose == True and itr[0] % 5 == 0:
                print("Iter {:5d}, lr: {:3.3e} Loss_perf = {:6f} Loss_distort = {:6f}, Loss_total = {:6f}".format(
                    itr[0], optimizer.param_groups[0]['lr'], loss_perf.item(), loss_distort.item(), total_loss.item()))
            globals()['loss_perf'] = loss_perf
            globals()['loss_distort'] = loss_distort
            itr[0] += 1
            return total_loss

        if not globals()['converged']: return carrier_tensor.data, 0, 0, False

        patch_tensor.data.clamp_(0, 1)  # correct pixels values
        carrier_tensor = (1 - M).mul(carrier_tensor) + M.mul(patch_tensor)
        closure()
        optimizer.step()
        lr_schedulers[0].step()
        # lr_schedulers[1].step(globals()['loss_perf_min'])

    carrier_tensor.data.clamp_(0, 1)  # pixel value correction
    return carrier_tensor.data, globals()['loss_perf'], globals()['loss_distort'], globals()['converged']

# targeted mismatch attack with a universarial patch
def tth_patch_multi_carrier(networks, scales, target_tensor, carrier_imgs, mode='normal',
                            num_steps=100, lr=1.0, lam=0.0, sigma_blur=0.0, verbose=True, seed=155, device=torch.device("cpu"),
                            target_type='image', patch_ratio=0.2):
    # if seed is not None:   # uncomment to reproduce the results of the ICCV19 paper - still some randomness though
    #     reproduce(seed)
    target_tensor = target_tensor.to(device)

    carrier_imgs = carrier_imgs.to(device)
    patch_tensor_org = torch.rand_like(carrier_imgs[0])  # parameters to be learned is a patch
    M = torch.zeros_like(carrier_imgs[0])
    patch_length, patch_width = int(carrier_imgs.shape[-1] * patch_ratio), int(carrier_imgs.shape[-2] * patch_ratio)
    M[:, 0:patch_length, -patch_width:] = 1
    patch_tensor_org = M.mul(patch_tensor_org)
    patch_tensor_org[:, 0:patch_length, -patch_width:] = preprocess_clip_toTensor((patch_length, patch_width))
    patch_tensor_org = torch.where(patch_tensor_org > 0.5, 1, 0).float()
    # M = torch.where(patch_tensor_org > 0.5, 1, 0).float()  # 仅仅对黑色部分进行攻击


    patch_tensor = nn.Parameter(patch_tensor_org.clone().data)

    carriers_tensor = (1-M).mul(carrier_imgs) + M.mul(patch_tensor)
    carrier_org = carrier_imgs.clone()                 # to compute distortion
    optimizer = optim.Adam([patch_tensor], lr=lr)
    lr_schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95),
                     torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                                patience=2)]

    bin_centers_fixed = torch.DoubleTensor(np.arange(0, 1.001, 0.05)).to(device)   # for histograms only
    scales = np.array(scales)
    sigma_blur_all = sigma_blur / np.array(scales)
    kernel_size_all = 2*np.floor((np.ceil(6*sigma_blur_all)/2))+1

    # pre-compute all target global-descriptors / histograms / tensors
    targets, norm_factors = {}, {}
    for network in networks:  # optimize for all networks
        network.eval()

        m = torch.FloatTensor(network.meta['mean']).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        s = torch.FloatTensor(network.meta['std']).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        for scale in scales:  # optimize for all scales
            si = (scales==scale).nonzero()[0].item()
            if sigma_blur > 0.0:
                GS = GaussianSmoothing(channels = 3, kernel_size = kernel_size_all[si], sigma = sigma_blur_all[si]).to(device)
            else:
                GS = nn.Sequential()  # identity function

            # normalize (mean-std), re-scale, and feed to the network
            if target_type == 'image':
                xl = network.visual_features(nn.functional.interpolate((GS(target_tensor) - m) / s, scale_factor=scale, mode='bilinear', align_corners=False))
            elif target_type == 'embedding':
                xl = target_tensor
            elif target_type == 'embeddings':
                xl = [each.unsqueeze(0) for each in target_tensor]

            if not isinstance(xl, list): xl = [xl]  # to support optimization for multi targets too
            for l in range(len(xl)):
                x = xl[l]
                if mode == 'global':
                    # global descriptors
                    targets[network.meta['architecture'], str(scale), 'layer'+str(l)] = network.norm(x).squeeze().detach()
                elif mode == 'hist':  # activation histogram
                    nf = x.max().detach()
                    norm_factors[network.meta['architecture'], str(scale), 'layer'+str(l)] = nf
                    targets[network.meta['architecture'],str(scale), 'layer'+str(l)] = hist_per_channel((x / nf).clamp(0,1), bin_centers_fixed).detach()
                elif mode == 'tensor':  # activation tensor
                    nf = (0.1*x.max()).detach()  # 0.1 ???
                    norm_factors[network.meta['architecture'],str(scale), 'layer'+str(l)] = nf
                    targets[network.meta['architecture'], str(scale), 'layer'+str(l)] = (x / nf).detach()

    # for convergence checks
    globals()['converged'] = True
    globals()['loss_perf_min'] = 1e+9
    globals()['loss_perf_converged'] = 1e-4
    globals()['convergence_safe'] = False

    print('Optimizing..')
    itr = [0]
    while itr[0] <= num_steps:

        def closure():
            optimizer.zero_grad()
            loss_perf = torch.Tensor(1).to(device)*0.0
            n = 0  # counter for loss summands
            for network in networks:  # optimize for all networks
                network.eval()
                m = torch.FloatTensor(network.meta['mean']).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                s = torch.FloatTensor(network.meta['std']).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                for scale in scales:  # optimize for all scales
                    si = (scales==scale).nonzero()[0].item()
                    if sigma_blur > 0.0:
                        GS = GaussianSmoothing(channels=3, kernel_size=kernel_size_all[si], sigma=sigma_blur_all[si]).to(device)
                    else:
                        GS = nn.Sequential() # identity function

                    # normalize (mean-std), re-scale, and feed to the network
                    carrier_x = network.visual_features(nn.functional.interpolate((GS(carriers_tensor) - m) / s, scale_factor=scale, mode='bilinear', align_corners=False))


                    if mode == 'global': # global descriptors
                        for l in range(len(targets)):
                            ref = network.norm(carrier_x).squeeze()
                            target = targets[network.meta['architecture'], str(scale), 'layer'+str(l)]
                            loss_perf += (1 - ref.mm(target.repeat((ref.shape[0],1)).T)).mean()  # add loss over networks and scales
                            n+= 1

            # compute loss
            if lam > 0:
                loss_distort = (patch_tensor-patch_tensor_org).pow(2.0).sum() / (patch_tensor.size(-1)*patch_tensor.size(-2))
            else:
                loss_distort = torch.Tensor(1).to(device)*0.0
            loss_perf = loss_perf / n  # divide by number of summands (networks, scales, poolings)
            total_loss = loss_perf + lam * loss_distort

            # check for convergence (hacky!)
            if loss_perf < globals()['loss_perf_min']:
                globals()['loss_perf_min'] = loss_perf.clone()

            if loss_perf < globals()['loss_perf_converged']:
                globals()['convergence_safe'] = True

            if globals()['converged'] and (loss_perf-globals()['loss_perf_min']) > 1*globals()['loss_perf_min'] and globals()['convergence_safe'] == False:
                globals()['converged'] = False
                print("Iter {:5d}, Loss_perf = {:6f} Loss_distort = {:6f} Loss_total = {:6f}".format(itr[0], loss_perf.item(), loss_distort.item(), total_loss.item()))
                print('Did not converge')

            total_loss.backward(retain_graph=True)

            if verbose == True and itr[0] % 5 == 0:
                print("Iter {:5d}, lr: {:3.3e} Loss_perf = {:6f} Loss_distort = {:6f}, Loss_total = {:6f}".format(
                    itr[0], optimizer.param_groups[0]['lr'], loss_perf.item(), loss_distort.item(), total_loss.item()))
            globals()['loss_perf'] = loss_perf
            globals()['loss_distort'] = loss_distort
            itr[0] += 1
            return total_loss

        if not globals()['converged']: return carriers_tensor.data, 0, 0, False

        patch_tensor.data.clamp_(0, 1)  # correct pixels values
        carriers_tensor = (1 - M).mul(carriers_tensor) + M.mul(patch_tensor)
        closure()
        optimizer.step()
        lr_schedulers[0].step()
        # lr_schedulers[1].step(globals()['loss_perf_min'])

    carriers_tensor.data.clamp_(0, 1)  # pixel value correction
    carriers_tensor = GS(carriers_tensor)
    return carriers_tensor.data, globals()['loss_perf'], globals()['loss_distort'], globals()['converged']


def hist_per_channel(x, bin_centers, sigma = 0.1):
    x = x.squeeze(0)
    N = x.size()[1]*x.size()[2]
    xflat = x.flatten().unsqueeze(1)
    expx = torch.exp(
        -torch.add(xflat.type(torch.cuda.DoubleTensor), -1.0*bin_centers.unsqueeze(0)).pow(2.0) / (2*sigma**2)
    ).type(torch.cuda.FloatTensor)
    nf = expx.sum(1).unsqueeze(1)
    nf[nf==0] = 1
    xh = torch.div(expx, nf)
    xh = xh.reshape(x.size(0), N, xh.size(-1))
    hists = xh.sum(1) / (x.size(1)*x.size(2))
    
    return hists


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).

    function implemented by Adrian Sahlman https://tinyurl.com/y2w8ktp5
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.kernel_size = kernel_size

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=(int((self.kernel_size[0]-1)/2),int((self.kernel_size[0]-1)/2)))
