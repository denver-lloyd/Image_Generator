__author__ = "Denver Lloyd"
__copyright__ = "TBD"

import numpy as np
import stats
import pdb


class Image_Generator:
    def __init__(self,
                 rows=100,
                 cols=100,
                 offset=168,
                 cfpn=0,
                 rfpn=0,
                 pfpn=0,
                 ctn=0,
                 rtn=0,
                 ptn=0,
                 L=20,
                 noise_func=np.random.normal):
        """
        class for generating a single image stack (L=#frames, N=rows, M=cols)
        with noise components plus offset
        Keyword Arguments:
        Returns:
        TODO: generate an FPN frame
        """

        self.rows = rows
        self.cols = cols
        self.cfpn = cfpn
        self.rfpn = rfpn
        self.pfpn = pfpn
        self.ctn = ctn
        self.rtn = rtn
        self.ptn = ptn
        self.L = L
        self.noise_func = noise_func

        np.random.seed(1)

        self.imgs = np.zeros((self.L, self.rows, self.cols))
        self.imgs += offset

    def noise_image(self):
        """
        add all noise sources to image taking residual
        temporal noise (additional FPN) into account
        for an accurate noise image
        """

        self.imgs = self.gen_temporal_noise(self.imgs)

    def gen_fpn(self, imgs):
        """
        generate fpn on a stack of frames
        """

        if self.cfpn:
            imgs += self.add_noise(noise_scale=self.cfpn,
                                   noise_size=(1, self.cols),
                                   noise_tile_size=(self.rows, 1))

        if self.rfpn:
            imgs += self.add_noise(noise_scale=self.rfpn,
                                   noise_size=(self.cols, 1),
                                   noise_tile_size=(1, self.cols))

        if self.pfpn:
            imgs += self.add_noise(noise_scale=self.pfpn,
                                   noise_size=(self.rows, self.cols),
                                   noise_tile_size=0)

        return self

    def update_noise(self, imgs, key='col_var_temp'):
        """
        """

        stat_vals = stats.get_stats(imgs, std=False)
        val = stat_vals[key]

        # make sure calculation is noise power
        pdb.set_trace()
        if key == 'col_var_temp':
            val = np.sqrt(self.ctn**2 - val)
        else:
            val = np.sqrt(self.rtn**2 - val)
        return val

    def gen_temporal_noise(self, imgs):
        """
        generate temporal noise on a stack of frames
        """

        # start with pix temporal noise, this will also contribute
        # to row and col component so need to update col and row
        # contributions
        pdb.set_trace()
        if self.ptn:
            imgs += self.add_noise(noise_scale=self.ptn,
                                   noise_size=(self.L, 1, 1),
                                   noise_tile_size=(1, self.rows, self.cols))

        # update col temporal noise contribution
        ctn = self.update_noise(imgs)
        if self.ctn:
            imgs += self.add_noise(noise_scale=ctn,
                                   noise_size=(self.L, 1, self.cols),
                                   noise_tile_size=(1, self.rows, 1))

        rtn = self.update_noise(imgs, key='row_var_temp')

        if self.rtn:
            imgs += self.add_noise(noise_scale=rtn,
                                   noise_size=(self.L, self.rows, 1),
                                   noise_tile_size=(1, 1, self.cols))

        return imgs

    def add_noise(self, noise_scale, noise_size, noise_tile_size=0):
        """
        generically generate comonent wise noise metrics
        """

        dist = self.noise_func(loc=0, scale=noise_scale,
                               size=noise_size)
        pdb.set_trace()
        if noise_tile_size:
            dist = np.tile(dist, noise_tile_size)

        return dist
