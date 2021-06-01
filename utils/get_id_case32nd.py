import glob
from tqdm import tqdm
from dataset.fetch_data_for_next_phase import add_image_id_to_pool

path_one32nd = "/dataset.local/all/hangd/dynamic_data/one32rd/"

for file in tqdm(glob.glob(path_one32nd+"imgs/*")):
    id = file[file.rfind("/"):]
    add_image_id_to_pool(id[1:][:-4], "../tmp/data_one32nd.json")
