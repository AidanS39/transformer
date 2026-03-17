import torch
import time
import warnings

from model import TransformerCheckpoint
from model import Transformer
from model import generate_response

from utils import get_device
from utils import slow_print

def main():
    # torch.manual_seed(42)
    warnings.filterwarnings('ignore', category=UserWarning)
    torch.set_float32_matmul_precision('high')
    
    device = get_device()
    
    # TODO: allow user to pick any model from models folder
    checkpoint = TransformerCheckpoint.load("./models/model_1024_16_12_2048.pt")

    slow_print(
        r"""
 ________________________________________________________________________________
|       __________  ___    _   _______ __________  ____  __  _____________       |
|      /_  __/ __ \/   |  / | / / ___// ____/ __ \/ __ \/  |/  / ____/ __ \      |
|       / / / /_/ / /| | /  |/ /\__ \/ /_  / / / / /_/ / /|_/ / __/ / /_/ /      |
|      / / / _, _/ ___ |/ /|  /___/ / __/ / /_/ / _, _/ /  / / /___/ _, _/       |
|     /_/ /_/ |_/_/  |_/_/ |_//____/_/    \____/_/ |_/_/  /_/_____/_/ |_|        |
|________________________________________________________________________________|
""", char_delay=0.01, word_delay=0)
    
    slow_print("\nWelcome to the TRANSFORMER Program!\n")

    slow_print("""
To generate a response: type out a prompt, then press ENTER.
To exit the program: press ENTER without submitting a prompt, or type `exit` then press ENTER.
"""
    )

    encoder = checkpoint.encoder
    config = checkpoint.config
    model_state = checkpoint.model_state

    model = Transformer(config).to(device)
    model.load_state_dict(model_state)
    model = torch.compile(model)

    keep_prompting = True

    while keep_prompting:
        prompt = input("Write the beginning of a story: ")

        start_time = time.time()
        response = generate_response(model, encoder=encoder, device=device, prompt=prompt, temp=1.5, k=3)

        if response == None:
            slow_print("\nGoodbye!\n")
            keep_prompting = False
        else:
            print(f"\n\ngenerated sequence in {time.time() - start_time} seconds\n")

if __name__ == "__main__":
    main()