import os
import gdown

from project.utils.datasets import build_dataHomeFolder

folder_name = build_dataHomeFolder()


if not os.path.exists(folder_name+'landsat'):
    os.makedirs(folder_name+'landsat')

if not os.path.exists(folder_name+'landsat/10s_10c_s_landsat_star.csv'):
    # url = "https://drive.google.com/file/d/1zEiHCmMmx_qz17nSmrDzgqHlheYTbsvF/view?usp=sharing"
    # url = "https://drive.google.com/file/d/1ab37NCbS1EUx5cqDaMv2V7Fk_g5chhlv/view?usp=sharing"
    url = "https://drive.google.com/file/d/1mD3wSA3ueNmoYHATWEYNBvT_09mJUspe/view?usp=sharing"
    gdown.download(url, folder_name+'landsat/10s_10c_s_landsat_star.csv', fuzzy=True)

if not os.path.exists(folder_name+'landsat/12s_12c_s_landsat_star.csv'):
    url = "https://drive.google.com/file/d/1Fo8A9yjzDiPZ0rYveSjC6tyutejpuUja/view?usp=sharing"
    gdown.download(url, folder_name+'landsat/12s_12c_s_landsat_star.csv', fuzzy=True)

if not os.path.exists(folder_name+'landsat/10s_4c_s_landsat_star_inter.csv'):
    url = "https://drive.google.com/file/d/1FdkDbSIxWKVtSgSUyeZOmhSjPMsiraBa/view?usp=sharing"
    gdown.download(url, folder_name+'landsat/10s_4c_s_landsat_star_inter.csv', fuzzy=True)

if not os.path.exists(folder_name+'landsat/10s_3c_s_landsat_star_inter.csv'):
    url = "https://drive.google.com/file/d/16ZUagV40WkNLSsWTardndF49dp9OmQqC/view?usp=sharing"
    gdown.download(url, folder_name+'landsat/10s_3c_s_landsat_star_inter.csv', fuzzy=True)