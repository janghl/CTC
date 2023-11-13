import torch
from ctc_model import CTCEncoder

device = 'cuda'

# Load pre-trained model
model = CTCEncoder().to(device)
model.load_state_dict(torch.load('ctc.pth'))
model.eval() 

# Define input and output buffers
input_buffer = []  
output_buffer = bytearray()

while True:
  # Get new frame
  frame = get_input_frame()  

  # Add new frame to input buffer
  input_buffer.append(frame)

  # If input buffer is full, start encoding
  if len(input_buffer) == buffer_size:

    # Convert input buffer to Tensor
    inputs = torch.stack(input_buffer, dim=0).to(device)  

    # Forward pass for encoding
    with torch.no_grad():
      latent, bits = model(inputs)
    
    # Add encoding result to output buffer
    output_buffer += bits.cpu().numpy().tobytes()
    
    # Clear input buffer 
    input_buffer = []

  # Write to file every N frames    
  if len(output_buffer) > chunk_size:
    write_to_file(output_buffer)
    output_buffer = bytearray()
