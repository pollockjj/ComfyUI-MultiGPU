import torch

# --- Configuration ---
# Target VRAM in GiB
TARGET_GIB = 17

# --- Do not edit below this line ---
BYTES_PER_GIB = 1024**3
target_bytes = int(TARGET_GIB * BYTES_PER_GIB)

# --- Script Execution ---
if not torch.cuda.is_available():
    print("CUDA is not available. Exiting.")
    exit()

device_count = torch.cuda.device_count()
if device_count == 0:
    print("No CUDA devices found. Exiting.")
    exit()

# List available devices for user selection
print("Available CUDA Devices:")
for i in range(device_count):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")

# Prompt for device selection with input validation
selected_device_index = -1
while True:
    try:
        choice = input(f"Enter the device number to use [0-{device_count - 1}]: ")
        selected_device_index = int(choice)
        if 0 <= selected_device_index < device_count:
            break
        else:
            print(f"Error: Invalid selection. Please enter a number between 0 and {device_count - 1}.")
    except ValueError:
        print("Error: Invalid input. Please enter a numerical index.")

device = f'cuda:{selected_device_index}'
print("-" * 30)
print(f"Selected device: {torch.cuda.get_device_name(selected_device_index)} ({device})")
print(f"Attempting to allocate {TARGET_GIB} GiB of VRAM...")

try:
    # Allocate a tensor of 'target_bytes' size on the selected device.
    # torch.uint8 is used as it is 1 byte per element.
    tensor = torch.empty(target_bytes, dtype=torch.uint8, device=device)
    
    allocated_gib = tensor.nbytes / BYTES_PER_GIB
    print(f"Successfully allocated {allocated_gib:.2f} GiB on '{device}'.")
    print("Allocation is active. Press CTRL+C or Enter to exit and release VRAM.")
    
    # Keep the script alive to hold the VRAM allocation
    input()

except torch.cuda.OutOfMemoryError:
    print(f"Error: CUDA out of memory on '{device}'. Could not allocate {TARGET_GIB} GiB.")
    print("This may be due to insufficient total VRAM or existing VRAM usage on this specific device.")
    
except Exception as e:
    print(f"An unexpected error occurred: {e}")

finally:
    print("Exiting script and releasing VRAM.")
