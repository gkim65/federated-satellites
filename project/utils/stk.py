import os
import gdown

from project.utils.datasets import build_dataHomeFolder

folder_name = build_dataHomeFolder()


if not os.path.exists(folder_name+'landsat'):
    os.makedirs(folder_name+'landsat')
    # url = "https://drive.google.com/file/d/1zEiHCmMmx_qz17nSmrDzgqHlheYTbsvF/view?usp=sharing"
    url = "https://drive.google.com/file/d/1ab37NCbS1EUx5cqDaMv2V7Fk_g5chhlv/view?usp=sharing"
    gdown.download(url, folder_name+'landsat/10s_10c_s_landsat_star.csv', fuzzy=True)