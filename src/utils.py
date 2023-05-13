import os
import sys

import urllib.request
from tqdm import tqdm


class download_progress_bar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def maybe_download_model(url, model_dir):
    model_name = os.path.basename(url)
    output_path = os.path.join(model_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(output_path):
        with download_progress_bar(unit='B', unit_scale=True,
                                   miniters=1, desc='Downloading model weights ' + url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

    return output_path
