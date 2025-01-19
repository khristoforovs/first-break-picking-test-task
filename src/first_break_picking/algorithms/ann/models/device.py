
import torch
from termcolor import colored

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
color = "green" if torch.cuda.is_available() else "red"
print(colored(f"\nRunning on {str(device).upper()}", color), end="\n\n")


pass