import msgpack
import cv2
import os
import tempfile
import zipfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from .parsers import ProjectEncoder, ProjectDecoder, ensure_string_keys

def save_state_to_proj_file(comic_translate, file_name):
    """
    Saves the state of the comic_translate object to a msgpack file and a folder of unique images.

    Parameters:
        comic_translate: The comic_translate object containing the state.
        file_name (str): The path to the output msgpack file.
    """
    encoder = ProjectEncoder()

    # Create a temporary directory for unique images
    with tempfile.TemporaryDirectory() as temp_dir:
        unique_images_dir = os.path.join(temp_dir, "unique_images")
        os.makedirs(unique_images_dir, exist_ok=True)

        # Initialize the state dictionary
        state = {
            'current_image_index': comic_translate.curr_img_idx,
            'image_states': comic_translate.image_states,
            'unique_images': {},  # Will store id to file_name mapping
            'image_data_references': {},
            'image_files_references': {},
            'in_memory_history_references': {},
            'image_history_references': {},
            'original_image_files': comic_translate.image_files,
            'current_history_index': comic_translate.current_history_index,
            'displayed_images': list(comic_translate.displayed_images),
            'loaded_images': comic_translate.loaded_images
        }

        image_id_counter = 0
        image_path_to_id = {}
        unique_images = {}

        # Helper function to copy image and assign ID
        def copy_and_assign(file_path):
            nonlocal image_id_counter
            if file_path not in image_path_to_id:
                image_id = image_id_counter
                image_id_counter += 1
                image_path_to_id[file_path] = image_id
                
                bname = os.path.basename(file_path)
                new_file_path = os.path.join(unique_images_dir, bname)
                
                # Copy file from disk
                shutil.copy2(file_path, new_file_path)
                unique_images[image_id] = bname
            
            return image_path_to_id[file_path]

        # Process in_memory_history
        for file, history in comic_translate.in_memory_history.items():
            state['in_memory_history_references'][file] = []
            for idx, img in enumerate(history):
                path = comic_translate.image_history[file][idx]
                if idx == comic_translate.current_history_index.get(file, 0):
                    # Link to image_data
                    image_id = copy_and_assign(path)
                    state['image_data_references'][file] = image_id
                else:
                    image_id = copy_and_assign(path)
                state['in_memory_history_references'][file].append(image_id)

        # Process image_files and image_history
        for file_path in comic_translate.image_files:
            if file_path in comic_translate.image_history:
                state['image_history_references'][file_path] = []
                history_paths = comic_translate.image_history[file_path]
                for path in history_paths:
                    image_id = copy_and_assign(path)
                    state['image_history_references'][file_path].append(image_id)
            else:
                # This file hasn't been loaded before
                image_id = copy_and_assign(file_path)
                state['image_files_references'][file_path] = image_id

        state['unique_images'] = ensure_string_keys(unique_images)

        # Serialize the state to msgpack
        msgpack_file = os.path.join(temp_dir, "state.msgpack")
        with open(msgpack_file, 'wb') as file:
            msgpack.pack(state, file, default=encoder.encode, use_bin_type=True)

        # Create a zip file containing the msgpack file and unique_images folder
        zip_file_name = file_name
        with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_STORED) as zipf:
            zipf.write(msgpack_file, arcname="state.msgpack")
            for root, _, files in os.walk(unique_images_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname=arcname)

def load_state_from_proj_file(comic_translate, file_name):
    decoder = ProjectDecoder()

    if not hasattr(comic_translate, 'temp_dir'):
        comic_translate.temp_dir = tempfile.mkdtemp()

    # Use local references to attributes for faster access
    image_data = comic_translate.image_data
    temp_dir = comic_translate.temp_dir
    in_memory_history = comic_translate.in_memory_history
    image_history = comic_translate.image_history
    original_to_temp = {}

    image_states = {}
    current_history_index = {}
    displayed_images = set()
    loaded_images = []

    if file_name.lower().endswith( '.ctpr'):
        with zipfile.ZipFile(file_name, 'r') as archive:
            for file in archive.namelist():
                archive.extract(file, temp_dir)

    msgpack_file = os.path.join(temp_dir, "state.msgpack")
    unique_images_dir = os.path.join(temp_dir, "unique_images")

    with open(msgpack_file, 'rb') as file:
        unpacker = msgpack.Unpacker(file, object_hook=decoder.decode, strict_map_key=True)
        try:
            map_size = unpacker.read_map_header()
        except ValueError:
            raise ValueError("Invalid msgpack format: Expected a map at the root.")

        state = {}
        for _ in range(map_size):
            key = unpacker.unpack()
            value = unpacker.unpack()
            state[key] = value

    # Build img_id_to_usage mapping
    img_id_to_usage = {}

    # Helper function to process references
    def process_references(ref_dict, ref_type):
        for file_path, img_id_str in ref_dict.items():
            img_id = int(img_id_str)
            if ref_type in ['in_memory_history', 'image_history']:
                for idx, img_id_sub_str in enumerate(img_id_str):
                    img_id_sub = int(img_id_sub_str)
                    img_id_to_usage.setdefault(img_id_sub, []).append((ref_type, file_path, idx))
            else:
                img_id_to_usage.setdefault(img_id, []).append((ref_type, file_path))

    # Process different references
    process_references(state.get('image_data_references', {}), 'image_data')
    process_references(state.get('image_files_references', {}), 'image_files')

    # For in_memory_history_references and image_history_references
    for ref_type in ['in_memory_history_references', 'image_history_references']:
        for file_path, img_id_list in state.get(ref_type, {}).items():
            for idx, img_id_str in enumerate(img_id_list):
                img_id = int(img_id_str)
                usage_type = 'in_memory_history' if ref_type == 'in_memory_history_references' else 'image_history'
                img_id_to_usage.setdefault(img_id, []).append((usage_type, file_path, idx))

    unique_images = state.get('unique_images', {})

    # Helper function to process a single image
    def process_image(img_file_name, usages, image_data,
                     original_to_temp, in_memory_history, image_history):
        img_path = os.path.join(unique_images_dir, img_file_name)

        for usage in usages:
            usage_type = usage[0]
            if usage_type == 'image_data':
                file_path = usage[1]
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_data[file_path] = img
            elif usage_type == 'in_memory_history':
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                file_path, idx = usage[1], usage[2]
                in_memory_history.setdefault(file_path, [])
                history = in_memory_history[file_path]
                while len(history) <= idx:
                    history.append(None)
                history[idx] = img
            elif usage_type == 'image_files':
                file_path = usage[1]
                temp_path = img_path
                original_to_temp[file_path] = temp_path
            elif usage_type == 'image_history':
                file_path, idx = usage[1], usage[2]
                image_history.setdefault(file_path, [])
                history = image_history[file_path]
                while len(history) <= idx:
                    history.append(None)
                history[idx] = img_path

    # Process images in parallel
    with ThreadPoolExecutor() as executor:
        future_to_img_id = {
            executor.submit(process_image, img_file_name, img_id_to_usage[int(img_id)],
                            image_data, original_to_temp,
                            in_memory_history, image_history): img_id
            for img_id, img_file_name in unique_images.items() if int(img_id) in img_id_to_usage
        }

        for future in as_completed(future_to_img_id):
            img_id = future_to_img_id[future]
            try:
                future.result()
            except Exception as exc:
                print(f'Image ID {img_id} generated an exception: {exc}')

    # Finalize loading by updating comic_translate attributes
    comic_translate.curr_img_idx = state.get('current_image_index', 0)

    original_image_files = state.get('original_image_files', [])
    comic_translate.image_files = [
        original_to_temp.get(file, file) for file in original_image_files
    ]

    image_states = state.get('image_states', {})
    comic_translate.image_states = {
        original_to_temp.get(file, file): img_state for file, img_state in image_states.items()
    }

    current_history_index = state.get('current_history_index', {})
    comic_translate.current_history_index = {
        original_to_temp.get(file, file): index for file, index in current_history_index.items()
    }

    displayed_images = state.get('displayed_images', [])
    comic_translate.displayed_images = set(
        original_to_temp.get(i, i) for i in displayed_images
    )
    loaded_images = state.get('loaded_images', [])
    comic_translate.loaded_images = [
        original_to_temp.get(file, file) for file in loaded_images
    ]

    comic_translate.image_data = {
        original_to_temp.get(file, file): img 
        for file, img in image_data.items()
    }

    comic_translate.in_memory_history = {
        original_to_temp.get(file, file): imgs 
        for file, imgs in in_memory_history.items()
    }

    comic_translate.image_history = {
        original_to_temp.get(file_path, file_path): history_list
        for file_path, history_list in image_history.items()
    }




