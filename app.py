# ... (keep imports and other functions as they are) ...
import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import random
import traceback

# --- Add ZSE-SBIR directory to Python path ---
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
except NameError:
    project_root = r"D:\ZS SBIR\ZSE-SBIR"
    print(f"[Warning] __file__ not defined. Using hardcoded project_root: {project_root}")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"[i] Added {project_root} to sys.path")
else:
    print(f"[i] {project_root} already in sys.path")

# --- Import necessary components ---
original_argv = sys.argv
try:
    sys.argv = [original_argv[0]]
    from options import Option
    from model.model import Model
finally:
    sys.argv = original_argv

# --- Constants and Configuration ---
CHECKPOINT_PATH = r"D:\ZS SBIR\ZSE-SBIR\checkpoint\Sketchy-ext\best_checkpoint.pth"
GALLERY_ROOT = r"D:\ZS SBIR\ZSE-SBIR\datasets\Sketchy\256x256\photo"
SKETCH_ROOT = r"D:\ZS SBIR\ZSE-SBIR\datasets\Sketchy\256x256\sketch\tx_000000000000_ready"
CACHE_DIR = os.path.join(project_root, "cache")
FEATURES_PATH = os.path.join(CACHE_DIR, "gallery_features.npy")
PATHS_PATH = os.path.join(CACHE_DIR, "gallery_paths.pkl")
TOP_K = 5
NUM_EXAMPLE_SKETCHES = 10 # Keep this number or adjust if needed
EXAMPLE_SKETCH_COLUMNS = 5 # Adjust columns for gallery layout
COMPONENT_HEIGHT = "250px" # Define height for scrollable components
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[i] Using device: {DEVICE}")
print(f"[i] Target Checkpoint path: {CHECKPOINT_PATH}")
print(f"[i] Target Gallery root: {GALLERY_ROOT}")
print(f"[i] Target Sketch root: {SKETCH_ROOT}")
print(f"[i] Cache directory: {CACHE_DIR}")

# Global variables
gallery_feats = None
gallery_paths = None
model = None
preprocess = None
args = None
available_classes = []
available_classes_markdown = ""
example_sketch_paths = []

# --- Function to Load Precomputed Features ---
# ... (load_cached_features function remains the same) ...
def load_cached_features():
    global gallery_feats, gallery_paths
    if os.path.exists(FEATURES_PATH) and os.path.exists(PATHS_PATH):
        print(f"[i] Found cached files...")
        try:
            print("[i] Loading gallery features from cache...")
            gallery_feats = np.load(FEATURES_PATH)
            print("[i] Loading gallery paths from cache...")
            with open(PATHS_PATH, 'rb') as f:
                gallery_paths = pickle.load(f)

            if isinstance(gallery_feats, np.ndarray) and isinstance(gallery_paths, list):
                print(f"[i] Successfully loaded {gallery_feats.shape[0]} features and {len(gallery_paths)} paths from cache.")
                if gallery_feats.shape[0] != len(gallery_paths):
                     print("[Warning] Mismatch between cached features/paths. Recomputing.")
                     gallery_feats, gallery_paths = None, None
                     return False
                return True # Success
            else:
                print("[Warning] Cached files corrupted. Recomputing.")
                gallery_feats, gallery_paths = None, None
                return False
        except Exception as e:
            print(f"[Warning] Error loading from cache: {e}. Recomputing.")
            gallery_feats, gallery_paths = None, None
            return False
    else:
        print("[i] Cached features/paths not found.")
        return False


# --- Model Loading and Setup ---
# ... (load_model_and_setup function remains the same) ...
def load_model_and_setup():
    global model, preprocess, args
    if args is not None and model is not None and preprocess is not None:
         return

    print("[i] Parsing options using Option().parse()...")
    original_argv_setup = sys.argv
    current_cwd = os.getcwd()
    try:
        os.chdir(project_root)
        sys.argv = [original_argv_setup[0]]
        args = Option().parse()
    except SystemExit:
        print("[Warning] Option().parse() exited. Creating basic args.")
        class BasicArgs: pass
        args = BasicArgs()
    finally:
        sys.argv = original_argv_setup
        os.chdir(current_cwd)

    # Override/Ensure necessary args
    args.load = CHECKPOINT_PATH
    args.dataroot = GALLERY_ROOT # Option parser might use this, keep it
    args.dataset = 'Sketchy'
    args.batch = getattr(args, 'batch', 32)
    args.image_size = getattr(args, 'image_size', 224)
    args.choose_cuda = getattr(args, 'choose_cuda', "0")
    if not hasattr(args, 'cls_number'): args.cls_number = 125 # Default for Sketchy
    if not hasattr(args, 'd_model'): args.d_model = 768
    if not hasattr(args, 'head'): args.head = 12
    if not hasattr(args, 'number'): args.number = 6
    if not hasattr(args, 'd_ff'): args.d_ff = 3072
    if not hasattr(args, 'embed_dim'): args.embed_dim = 768
    if not hasattr(args, 'patch_size'): args.patch_size = 16

    print(f"[i] Using args: batch={args.batch}, image_size={args.image_size}, load={args.load}")
    print("[i] Initializing ZSE-SBIR model...")
    try:
        current_cwd = os.getcwd()
        os.chdir(project_root)
        model = Model(args).to(DEVICE)
    except AttributeError as e: print(f"[Error] Missing attribute during Model init: {e}"); raise e
    except Exception as e: print(f"[Error] Could not initialize Model: {e}"); raise e
    finally: os.chdir(current_cwd)

    print(f"[i] Loading checkpoint from: {args.load}")
    if not os.path.exists(args.load): raise FileNotFoundError(f"Checkpoint missing: {args.load}")

    state_dict_to_load = None
    try:
        ckpt = torch.load(args.load, map_location=DEVICE, weights_only=False)
        if 'model' in ckpt: state_dict_to_load = ckpt['model']
        elif 'state_dict' in ckpt: state_dict_to_load = ckpt['state_dict']
        else: state_dict_to_load = ckpt
    except Exception as e:
        print(f"[Warning] Failed loading checkpoint with weights_only=False: {e}. Trying weights_only=True...")
        try:
             ckpt = torch.load(args.load, map_location=DEVICE, weights_only=True)
             state_dict_to_load = ckpt
        except Exception as e_true:
            print(f"[Error] Failed loading checkpoint with weights_only=True as well: {e_true}")
            raise RuntimeError(f"Could not load checkpoint state dict from: {args.load}")

    if state_dict_to_load is None:
        raise RuntimeError("Failed to extract state_dict from checkpoint.")

    filtered_state_dict = {k: v for k, v in state_dict_to_load.items() if k in model.state_dict()}
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

    if missing_keys: print(f"[Warning] Missing keys while loading state_dict: {missing_keys}")
    if unexpected_keys: print(f"[Warning] Unexpected keys while loading state_dict: {unexpected_keys}")
    if not filtered_state_dict and state_dict_to_load: print("[Warning] State dictionary mismatch: No keys from the checkpoint matched the model.")
    elif not state_dict_to_load: print("[Warning] Loaded state dictionary was empty.")
    else: print(f"[i] Loaded {len(filtered_state_dict)} matching keys into model.")

    model.eval()
    print("[i] Model loaded successfully.")

    preprocess = transforms.Compose([
        transforms.Resize(args.image_size), transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("[i] Preprocessing transform defined.")

# --- Compute and Save Gallery Features ---
# ... (compute_and_save_gallery_features function remains the same) ...
def compute_and_save_gallery_features():
    global gallery_feats, gallery_paths, model, preprocess, args
    if model is None or preprocess is None or args is None:
        print("[Warning] Model setup incomplete. Running setup first.")
        load_model_and_setup()

    print(f"[i] Computing gallery features from: {GALLERY_ROOT}")
    if not os.path.exists(GALLERY_ROOT): raise FileNotFoundError(f"Gallery root missing: {GALLERY_ROOT}")

    local_available_classes = []
    try:
        ds = ImageFolder(GALLERY_ROOT, transform=preprocess)
        if not ds.classes: print(f"[Warning] No class folders found by ImageFolder in {GALLERY_ROOT}. Will rely on SKETCH_ROOT later.")
        else:
            local_available_classes = ds.classes
            print(f"[i] ImageFolder found {len(ds.classes)} classes in gallery: {', '.join(ds.classes[:10])}...")
    except Exception as e: print(f"[Error] Failed creating ImageFolder for gallery: {e}. Will rely on SKETCH_ROOT later.")

    try:
        if 'ds' not in locals() or not hasattr(ds, 'samples'): raise RuntimeError("Gallery ImageFolder failed, cannot compute features.")
        ldr = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
        gallery_paths = [p for p, _ in ds.samples]
        print(f"[i] Found {len(gallery_paths)} images in gallery for feature extraction.")
    except Exception as e: print(f"[Error] Failed creating DataLoader for gallery: {e}"); raise

    _gallery_feats_list = []
    dummy_sk = torch.zeros(1, 3, args.image_size, args.image_size).to(DEVICE)

    model.eval()
    with torch.no_grad():
        for i, (imgs, _) in enumerate(ldr):
            print(f"\r[i] Processing gallery batch {i+1}/{len(ldr)}...", end="")
            B = imgs.size(0); imgs = imgs.to(DEVICE)
            current_dummy_sk = dummy_sk.repeat(B, 1, 1, 1)
            try:
                output = model(imgs, current_dummy_sk, stage='test', only_sa=True)
                if isinstance(output, tuple): sa_feats = output[0]
                else: sa_feats = output
                _gallery_feats_list.append(sa_feats[:, 0].cpu())
            except Exception as e: print(f"\n[Error] processing gallery batch {i+1}: {e}"); continue

    print("\n[i] Gallery feature extraction complete.")
    if not _gallery_feats_list: raise RuntimeError("No features extracted from gallery.")
    gallery_feats = torch.cat(_gallery_feats_list, 0).numpy()
    print(f"[i] Gallery features computed. Shape: {gallery_feats.shape}")

    try:
        print(f"[i] Saving features to {FEATURES_PATH}"); os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(FEATURES_PATH, gallery_feats)
        print(f"[i] Saving paths to {PATHS_PATH}");
        with open(PATHS_PATH, 'wb') as f: pickle.dump(gallery_paths, f)
        print("[i] Cache saved successfully.")
    except Exception as e: print(f"[Error] Failed to save cache: {e}")

# --- Function to Load Example Sketches and Format Classes ---
# ... (load_example_sketches_and_classes function remains largely the same) ...
def load_example_sketches_and_classes():
    global example_sketch_paths, available_classes, available_classes_markdown, SKETCH_ROOT, NUM_EXAMPLE_SKETCHES

    print("[i] Preparing example sketches and class list...")
    if not os.path.isdir(SKETCH_ROOT):
        print(f"[Error] SKETCH_ROOT '{SKETCH_ROOT}' not found or not a directory.")
        available_classes = []
        available_classes_markdown = "**Error:** Sketch directory not found."
        example_sketch_paths = []
        return

    try:
        potential_classes = [d for d in os.listdir(SKETCH_ROOT) if os.path.isdir(os.path.join(SKETCH_ROOT, d)) and not d.startswith('.')]
        if potential_classes:
            available_classes = sorted(potential_classes)
            print(f"[i] Found {len(available_classes)} classes in {SKETCH_ROOT}: {', '.join(available_classes[:10])}...")
        else:
            print(f"[Error] No subdirectories (classes) found directly within {SKETCH_ROOT}.")
            available_classes = []
    except Exception as e: print(f"[Error] Could not list directories in {SKETCH_ROOT}: {e}"); available_classes = []

    if available_classes:
        # Keep Markdown formatting for bold title, etc.
        class_list_str = "\n".join([f"- `{cls_name}`" for cls_name in available_classes]) # Use backticks for inline code style
        available_classes_markdown = f"**Available Classes ({len(available_classes)}):**\n{class_list_str}"
        print(f"[i] Formatted class list markdown.")
    elif not available_classes_markdown:
        available_classes_markdown = "No classes found or error reading sketch directory."
        print("[Warning] Could not generate class list markdown.")

    example_sketch_paths = []
    if not available_classes:
        print("[Warning] No classes available. Cannot load example sketches.")
        return

    possible_classes = available_classes[:]
    random.shuffle(possible_classes)
    print(f"[i] Searching for up to {NUM_EXAMPLE_SKETCHES} example sketches...")
    found_count = 0
    attempts = 0
    max_attempts = len(possible_classes) * 3
    processed_classes = set()

    while found_count < NUM_EXAMPLE_SKETCHES and possible_classes:
        attempts += 1
        if attempts > max_attempts: print("[Warning] Reached max attempts searching for sketches."); break
        class_name = possible_classes.pop(0)
        processed_classes.add(class_name)
        class_path = os.path.join(SKETCH_ROOT, class_name)
        if not os.path.isdir(class_path): continue

        try:
            sketches_in_class = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            if sketches_in_class:
                chosen_sketch_file = random.choice(sketches_in_class)
                full_path = os.path.join(class_path, chosen_sketch_file)
                if full_path not in example_sketch_paths:
                    example_sketch_paths.append(full_path)
                    found_count += 1
                    # print(f"[Debug] Found example sketch: {full_path}") # Uncomment for verbose log
                elif class_name in processed_classes: possible_classes.append(class_name) # Re-add class if duplicate image found
            # else: print(f"[Debug] No sketch image files found in: {class_path}") # Uncomment for verbose log
        except Exception as e: print(f"[Warning] Error accessing sketch class directory {class_path}: {e}")

    if not example_sketch_paths: print("[Warning] Could not find any example sketches after searching.")
    else: print(f"[i] Successfully loaded {len(example_sketch_paths)} example sketch paths.")


# --- Gradio Interface Function ---
# ... (retrieve_images function remains the same) ...
def retrieve_images(sketch_input):
    global model, preprocess, gallery_feats, gallery_paths, TOP_K, DEVICE, args
    global example_sketch_paths, available_classes_markdown

    # Note: We don't need to return example_sketch_paths and available_classes_markdown
    # again here unless we want them to dynamically update based on the sketch input,
    # which is not the current requirement. They are set once at startup.
    # We only need to return the retrieved_gallery paths.

    if model is None or preprocess is None or gallery_feats is None or gallery_paths is None or args is None:
         print("[Error] App not fully initialized.")
         return ["Error: Application not ready. Please check logs."] # Return only gallery error

    if sketch_input is None:
        print("[i] Sketchpad cleared.")
        return [] # Return empty gallery

    sketch_np = None
    try:
        if isinstance(sketch_input, dict):
            if 'image' in sketch_input and isinstance(sketch_input['image'], np.ndarray): sketch_np = sketch_input['image']
            elif 'composite' in sketch_input and isinstance(sketch_input['composite'], np.ndarray): sketch_np = sketch_input['composite']
            else: raise TypeError(f"Sketchpad dictionary keys: {sketch_input.keys()}. No numpy array found.")
        elif isinstance(sketch_input, np.ndarray): sketch_np = sketch_input
        else: raise TypeError(f"Unexpected input type from Sketchpad: {type(sketch_input)}")
        if sketch_np is None: raise ValueError("Failed to extract numpy array from sketch input.")

        print("[i] Processing sketch...")
        if sketch_np.shape[2] == 4:
             bg = Image.new("RGB", (sketch_np.shape[1], sketch_np.shape[0]), (255, 255, 255))
             sketch_pil_rgba = Image.fromarray(sketch_np, 'RGBA')
             bg.paste(sketch_pil_rgba, (0, 0), sketch_pil_rgba)
             sketch_pil = bg
        elif sketch_np.shape[2] == 3: sketch_pil = Image.fromarray(sketch_np, 'RGB')
        else: raise ValueError(f"Unexpected number of channels in sketch: {sketch_np.shape[2]}")
    except Exception as e:
        print(f"[Error] Processing sketch input: {e}"); traceback.print_exc()
        return [f"Error processing sketch: {e}"] # Return gallery error

    try:
        sk_t = preprocess(sketch_pil).unsqueeze(0).to(DEVICE)
        dummy_img = torch.zeros(1, 3, args.image_size, args.image_size).to(DEVICE)
        print("[i] Extracting sketch features...")
        model.eval()
        with torch.no_grad():
            output = model(sk_t, dummy_img, stage='test', only_sa=True)
            if isinstance(output, tuple): sa_q = output[0]
            else: sa_q = output
            query_vec = sa_q[:, 0].cpu().numpy()

        print("[i] Calculating similarities...")
        sims = cosine_similarity(query_vec, gallery_feats)[0]
        print("[i] Finding top K images...")
        k = min(TOP_K, len(gallery_paths))
        if k <= 0: print("[Warning] No gallery paths available for retrieval."); top_paths = []
        else:
            ids = np.argsort(sims)[::-1][:k]
            top_paths = [gallery_paths[i] for i in ids]
            top_scores = [sims[i] for i in ids]
            print("[i] Top retrieved paths and scores:")
            for path, score in zip(top_paths, top_scores): print(f"- {os.path.basename(path)}: {score:.4f}")

        return top_paths # Return only the list of retrieved image paths

    except Exception as e:
        print(f"[Error] During feature extraction or retrieval: {e}"); traceback.print_exc()
        return [f"Error during retrieval: {e}"] # Return gallery error


# --- Main Execution Block ---
if __name__ == "__main__":
    setup_successful = False
    cache_loaded = load_cached_features()

    try:
        print("-" * 30); print("[i] Initializing model architecture...")
        load_model_and_setup()
        if cache_loaded and gallery_feats is not None and gallery_paths is not None: print("[i] Valid cache loaded for gallery features and paths.")
        else: print("[i] Cache miss or invalid. Computing gallery features..."); compute_and_save_gallery_features(); print("[i] Gallery feature computation complete.")
        load_example_sketches_and_classes() # Load examples/classes after setup
        if model is None or preprocess is None or gallery_feats is None or gallery_paths is None: raise RuntimeError("Essential components missing post-setup.")
        print("[i] Setup complete."); print("-" * 30); setup_successful = True
    except FileNotFoundError as e: print(f"[Critical Error - File Not Found] {e}"); available_classes_markdown = f"**Error:** Setup failed (File Not Found: {e})"
    except RuntimeError as e: print(f"[Critical Error - Runtime] {e}"); available_classes_markdown = f"**Error:** Setup failed (Runtime: {e})"
    except Exception as e: print(f"[Critical Error - Unexpected] Failed during setup: {e}"); traceback.print_exc(); available_classes_markdown = f"**Error:** Setup failed (Unexpected: {e})"

    # --- Define CSS for Scrollable Markdown ---
    # Target the specific markdown component using its elem_id
    # The actual scrollable container might be a child or parent, inspect element if needed
    # This targets the block containing the markdown component.
    COMPONENT_HEIGHT = "250px"
    custom_css = f"""
    #class-list-markdown {{ /* Use this ID in the Markdown component below */
        height: {COMPONENT_HEIGHT};
        overflow-y: auto !important; /* Use !important to try overriding other styles */
        display: block; /* Ensure it behaves like a block */
        border: 1px solid #333; /* Optional: adds a border */
        padding: 10px; /* Optional: adds some inner spacing */
    }}
    """

    # --- Setup Gradio Interface ---
    print("[i] Setting up Gradio interface...")
    # Use css argument in gr.Blocks
    with gr.Blocks(css=custom_css) as iface:
        gr.Markdown(
            """
            # Zero-Shot Sketch-Based Image Retrieval
            Draw a sketch and click Submit. Use a smaller brush size for better results.
            To clear your sketch, click the trashcan icon on the sketchpad.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                sketch_input = gr.Sketchpad(type="numpy", label="Draw your sketch here")
            with gr.Column(scale=2):
                retrieved_gallery = gr.Gallery(label="Retrieved Images", columns=TOP_K, object_fit="contain", height=300) # Main results gallery height

        submit_button = gr.Button("Submit Sketch")
        clear_button = gr.Button("Clear All")
        gr.Markdown("---")

        with gr.Row():
             with gr.Column(scale=2):
                 # Set height directly on the Gallery component
                 example_gallery = gr.Gallery(
                     value=example_sketch_paths,
                     label="Example Sketches",
                     columns=EXAMPLE_SKETCH_COLUMNS,
                     object_fit="contain",
                     height=COMPONENT_HEIGHT, # Set fixed height
                     elem_id="example-sketch-gallery-wrapper"
                 )
             with gr.Column(scale=1):
                 # Assign elem_id for CSS targeting
                class_list_display = gr.Markdown(
                     value=available_classes_markdown,
                     label="Available Classes",
                     elem_id="class-list-markdown" # Make sure this ID matches the CSS selector
                 )


        # Define interaction: Button click updates only the retrieved_gallery
        submit_button.click(
            fn=retrieve_images,
            inputs=sketch_input,
            outputs=[retrieved_gallery] # Only update this output
        )
        def clear_all_outputs():
            # Return None for sketchpad (clears it)
            # Return empty list for gallery (clears it)
            return None, []

        clear_button.click(
            fn=clear_all_outputs,         # Function to call
            inputs=None,                  # No inputs needed for this function
            outputs=[sketch_input, retrieved_gallery] # The components to clear
        )

    print("[i] Launching Gradio app...")
    # Launch regardless of setup success to show potential error messages
    iface.launch(share=False)
    if not setup_successful:
        print("[Hint] Setup failed. Check console logs and the error message in the Gradio UI.")